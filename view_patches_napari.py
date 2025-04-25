import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import napari


def load_patches(summary_csv):
    """Load patch paths and labels from summary CSV."""
    df = pd.read_csv(summary_csv)
    return df


def main(output_dir):
    summary_csv = os.path.join(output_dir, 'samples.csv')
    if not os.path.exists(summary_csv):
        print(f"Summary CSV not found: {summary_csv}")
        sys.exit(1)
    df = load_patches(summary_csv)
    if df.empty:
        print("No patches found in summary CSV.")
        sys.exit(1)
    # Load all patches into memory (can optimize for huge sets)
    images = []
    labels = []
    cell_ids = []
    for _, row in df.iterrows():
        patch_path = row['patch_path']
        label = row['label']
        cell_id = row['cell_id']
        if os.path.exists(patch_path):
            img = np.array(Image.open(patch_path))
            images.append(img)
            labels.append(str(label))
            cell_ids.append(str(cell_id))
    if not images:
        print("No patch images found.")
        sys.exit(1)
    # Stack images for napari (N, H, W)
    image_stack = np.stack(images)
    # Start napari viewer
    viewer = napari.Viewer()
    viewer.add_image(image_stack, name="DAPI patches")
    # Add labels as text overlay
    viewer.add_labels(np.arange(len(labels)), name="Cell Index")
    viewer.text_overlay = [f"cell_id: {cid}, label: {lbl}" for cid, lbl in zip(cell_ids, labels)]
    print("Use the layer controls and text overlay to browse patches.")
    napari.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Napari viewer for DAPI patches.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing samples.csv and patch PNGs')
    args = parser.parse_args()
    main(args.output_dir)
