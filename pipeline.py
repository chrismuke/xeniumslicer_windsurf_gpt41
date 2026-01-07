import os
import argparse
import numpy as np
import pandas as pd
import tifffile
import pyarrow.parquet as pq
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from PIL import Image

# Constants
MICRON_PER_PIXEL = 0.2125  # from doc, may be overridden by metadata


def load_dapi_image(image_path, fast_single_plane=True):
    """Load DAPI image from Zarr if available, otherwise from OME-TIFF. Always use only first channel/plane for speed. No conversion or caching performed."""
    import logging, tifffile, zarr, numpy as np, os
    zarr_path = os.path.splitext(image_path)[0] + '.zarr'
    if os.path.exists(zarr_path):
        logging.info(f"Zarr file found: {zarr_path}. Loading DAPI image from Zarr...")
        z = zarr.open(zarr_path, mode='r')
        logging.info(f"Zarr type: {type(z)}")
        if isinstance(z, zarr.Group):
            arr_keys = list(z.array_keys())
            if not arr_keys:
                # Try to search subgroups for arrays (Xenium convention)
                group_keys = list(z.group_keys())
                for gk in group_keys:
                    subgroup = z[gk]
                    if hasattr(subgroup, 'array_keys'):
                        sub_arr_keys = list(subgroup.array_keys())
                        if sub_arr_keys:
                            logging.info(f"Found array(s) in subgroup '{gk}': {sub_arr_keys}")
                            img = subgroup[sub_arr_keys[0]][:]
                            logging.info(f"Loaded array from subgroup '{gk}'")
                            return img
                raise ValueError(f"No arrays found in Zarr group or its subgroups: {zarr_path}")
            img = z[arr_keys[0]][:]
        else:
            img = z[:]
        logging.info(f"Loaded flat array from Zarr")
        return img
    else:
        logging.info(f"Zarr file not found, loading OME-TIFF: {image_path}")
        img = tifffile.imread(image_path)
        logging.info(f"Loaded TIFF image shape: {img.shape}")
        # Always use only first channel/plane for speed
        if img.ndim == 3:
            img = img[0]
            logging.info(f"Selected first plane/channel, shape: {img.shape}")
        return img



def count_total_cells(boundary_path):
    """Count number of unique cell_ids for ETA estimation (if feasible)."""
    import pyarrow.dataset as ds
    import pandas as pd
    if boundary_path.endswith('.csv') or boundary_path.endswith('.gz'):
        try:
            ids = set()
            for chunk in pd.read_csv(boundary_path, usecols=['cell_id'], chunksize=100000):
                ids.update(chunk['cell_id'].unique())
            return len(ids)
        except Exception:
            return None
    elif boundary_path.endswith('.parquet'):
        try:
            dataset = ds.dataset(boundary_path, format="parquet")
            cell_ids = set()
            for batch in dataset.to_batches(columns=['cell_id'], batch_size=100000):
                df = batch.to_pandas()
                cell_ids.update(df['cell_id'].unique())
            return len(cell_ids)
        except Exception:
            return None
    else:
        return None

def cell_boundary_chunks(boundary_path, chunksize=10000):
    """Yield cell boundary groups in batches from parquet or CSV, never loading full file into memory."""
    import pyarrow.dataset as ds
    if boundary_path.endswith('.csv') or boundary_path.endswith('.gz'):
        for chunk in pd.read_csv(boundary_path, chunksize=chunksize):
            yield from chunk.groupby('cell_id')
    elif boundary_path.endswith('.parquet'):
        # Use pyarrow.dataset for streaming
        dataset = ds.dataset(boundary_path, format="parquet")
        batch = []
        last_cell_id = None
        for record_batch in dataset.to_batches(batch_size=chunksize):
            df = record_batch.to_pandas()
            for cell_id, group in df.groupby('cell_id'):
                if last_cell_id is not None and cell_id == last_cell_id:
                    batch[-1] = pd.concat([batch[-1], group], ignore_index=True)
                else:
                    batch.append(group)
                last_cell_id = cell_id
            if len(batch) >= chunksize:
                for g in batch:
                    yield g['cell_id'].iloc[0], g
                batch = []
        for g in batch:
            yield g['cell_id'].iloc[0], g
    else:
        raise ValueError(f"Unsupported file type: {boundary_path}")


def load_cell_stats(stats_path):
    """Load cell stats from parquet or CSV."""
    if stats_path.endswith('.parquet'):
        df = pq.read_table(stats_path).to_pandas()
    else:
        df = pd.read_csv(stats_path)
    return df


def get_polygon_pixel_coords(micron_coords):
    """Convert list of (x, y) micron coordinates to pixel coordinates."""
    return [(x / MICRON_PER_PIXEL, y / MICRON_PER_PIXEL) for x, y in micron_coords]


def extract_patch(image, polygon, pad=10, cell_id=None, log_debug=False):
    """Extract a rectangular patch from image containing the polygon. Returns None if degenerate. Logs debug info if log_debug."""
    minx, miny, maxx, maxy = polygon.bounds
    minx_i = max(int(minx) - pad, 0)
    miny_i = max(int(miny) - pad, 0)
    maxx_i = min(int(maxx) + pad, image.shape[1])
    maxy_i = min(int(maxy) + pad, image.shape[0])
    # Check for degenerate bbox
    if maxx_i <= minx_i or maxy_i <= miny_i:
        if log_debug:
            import logging
            logging.warning(f"cell_id={cell_id}: Degenerate bbox: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy} -> indices: {minx_i}, {miny_i}, {maxx_i}, {maxy_i}")
        return None, (minx_i, miny_i, maxx_i, maxy_i)
    if minx_i >= image.shape[1] or miny_i >= image.shape[0] or maxx_i <= 0 or maxy_i <= 0:
        if log_debug:
            import logging
            logging.warning(f"cell_id={cell_id}: Polygon bbox is completely outside image bounds. Image shape: {image.shape}, bbox indices: {minx_i}, {miny_i}, {maxx_i}, {maxy_i}")
        return None, (minx_i, miny_i, maxx_i, maxy_i)
    patch = image[miny_i:maxy_i, minx_i:maxx_i]
    # Ensure patch is 2D
    if patch.ndim > 2:
        patch = np.squeeze(patch)
    if patch.ndim != 2:
        if log_debug:
            import logging
            logging.warning(f"cell_id={cell_id}: Patch is not 2D after squeeze. Patch shape: {patch.shape}")
        return None, (minx_i, miny_i, maxx_i, maxy_i)
    if patch.size == 0:
        if log_debug:
            import logging
            logging.warning(f"cell_id={cell_id}: Patch is empty. Indices: {minx_i}, {miny_i}, {maxx_i}, {maxy_i}")
        return None, (minx_i, miny_i, maxx_i, maxy_i)
    return patch, (minx_i, miny_i, maxx_i, maxy_i)


def save_patch(patch, output_dir, cell_id, label):
    os.makedirs(output_dir, exist_ok=True)
    # Convert to uint8 for saving
    if patch.dtype != np.uint8:
        patch = ((patch - patch.min()) / (np.ptp(patch) + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(patch)
    patch_path = os.path.join(output_dir, f"cell_{cell_id}.png")
    img.save(patch_path)
    return patch_path


def main(data_dir, output_dir, batch_size=1000, chunk_size=10000):
    import logging
    import time
    import traceback
    # Ensure output directory exists before logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Set up logging to both console and file
    log_path = os.path.join(output_dir, 'pipeline.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode='w')])
    logging.info(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Data dir: {data_dir}, Output dir: {output_dir}, Batch size: {batch_size}, Chunk size: {chunk_size}")

    # File paths
    dapi_path = os.path.join(data_dir, "morphology.ome.tif")
    boundaries_path = os.path.join(data_dir, "cell_boundaries.parquet")
    stats_path = os.path.join(data_dir, "cells.parquet")

    # Load DAPI image
    t0 = time.time()
    logging.info(f"Loading DAPI image from {dapi_path}")
    dapi_img_full = load_dapi_image(dapi_path)
    logging.info(f"Loaded DAPI image shape: {dapi_img_full.shape} (elapsed: {time.time()-t0:.2f}s)")
    if dapi_img_full.ndim == 3:
        dapi_img = np.max(dapi_img_full, axis=0)
        logging.info(f"Using max-projected DAPI image shape: {dapi_img.shape}")
    else:
        dapi_img = dapi_img_full

    # Load cell stats
    t0 = time.time()
    logging.info(f"Loading cell stats from {stats_path}")
    cell_stats = load_cell_stats(stats_path)
    logging.info(f"Loaded cell stats shape: {cell_stats.shape} (elapsed: {time.time()-t0:.2f}s)")

    # Prepare output
    t0 = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    summary_csv = os.path.join(output_dir, 'samples.csv')
    logging.info(f"Output summary CSV: {summary_csv}")
    header_written = False
    batch = []

    # Count total cells for ETA
    t0 = time.time()
    logging.info("Counting total cells for ETA estimation (if feasible)...")
    total_cells = count_total_cells(boundaries_path)
    logging.info(f"Counted total cells in {time.time()-t0:.2f}s")
    if total_cells is not None:
        logging.info(f"Estimated total cells: {total_cells}")
    else:
        logging.info("Could not estimate total number of cells. ETA will be approximate.")

    start_time = time.time()
    processed = 0
    log_interval = batch_size
    try:
        t_load_boundaries = time.time()
        logging.info(f"Starting patch extraction using boundaries from {boundaries_path}")
        for i, (cell_id, group) in enumerate(cell_boundary_chunks(boundaries_path, chunksize=chunk_size)):
            batch_start_time = time.time()
            poly_coords = list(zip(group['vertex_x'], group['vertex_y']))
            n_vertices = len(poly_coords)
            pixel_coords = get_polygon_pixel_coords(poly_coords)
            polygon = Polygon(pixel_coords)
            bounds = polygon.bounds
            patch, bbox = extract_patch(dapi_img, polygon, cell_id=cell_id, log_debug=(i < 20))
            patch_shape = patch.shape if patch is not None else None
            if i < 20:
                logging.info(f"cell_id={cell_id}, n_vertices={n_vertices}, bounds={bounds}, bbox_indices={bbox}, patch_shape={patch_shape}, image_shape={dapi_img.shape}")
            if n_vertices < 3:
                if i < 20:
                    logging.warning(f"cell_id={cell_id}: Skipping degenerate polygon (n_vertices={n_vertices})")
                continue
            if patch is None:
                if i < 20:
                    logging.warning(f"cell_id={cell_id}: Patch extraction returned None. See previous logs for details.")
                continue
            label = group['label_id'].iloc[0] if 'label_id' in group else None
            if label is None and 'label' in cell_stats.columns:
                label_row = cell_stats[cell_stats['cell_id'] == cell_id]
                label = label_row['label'].values[0] if not label_row.empty else 'unknown'
            patch_path = save_patch(patch, output_dir, cell_id, label)
            batch.append({'cell_id': cell_id, 'label': label, 'patch_path': patch_path})
            processed += 1
            if processed % log_interval == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                if total_cells is not None:
                    percent = 100 * processed / total_cells
                    remaining = total_cells - processed
                    eta = remaining / speed if speed > 0 else -1
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta)) if eta > 0 else 'unknown'
                    logging.info(f"Processed {processed}/{total_cells} cells ({percent:.2f}%) in {elapsed:.1f}s ({speed:.2f} cells/sec), ETA: {eta_str}")
                else:
                    logging.info(f"Processed {processed} cells in {elapsed:.1f}s ({speed:.2f} cells/sec)")
            if len(batch) >= batch_size:
                batch_elapsed = time.time() - batch_start_time
                logging.info(f"Writing batch of {len(batch)} samples to CSV (batch took {batch_elapsed:.2f}s)")
                pd.DataFrame(batch).to_csv(summary_csv, mode='a', header=not header_written, index=False)
                header_written = True
                batch = []
        # Write any remaining
        if batch:
            pd.DataFrame(batch).to_csv(summary_csv, mode='a', header=not header_written, index=False)
        total_time = time.time() - start_time
        avg_speed = processed / total_time if total_time > 0 else 0
        logging.info(f"Saved {processed} samples to {output_dir} in {total_time:.1f}s (avg speed: {avg_speed:.2f} cells/sec)")
        logging.info(f"Pipeline finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xenium DAPI Patch Extraction Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='Input data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for patches and CSV')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of samples to write to CSV at once (default: 1000)')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for reading boundaries (default: 10000)')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, batch_size=args.batch_size, chunk_size=args.chunk_size)
