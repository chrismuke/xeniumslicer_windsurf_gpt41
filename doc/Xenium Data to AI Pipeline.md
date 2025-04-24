# **A Performant Pipeline for Generating Expression-Labeled DAPI Training Samples from Xenium Data for AI-Based Cell Classification**

Abstract  
This report details a comprehensive strategy and Python-based pipeline for generating labeled training data suitable for an Artificial Intelligence (AI) model designed to segment and classify cells based solely on DAPI staining patterns of the nucleus and perinuclear region. Leveraging the rich multi-modal information from 10x Genomics Xenium spatial transcriptomics datasets, the proposed approach utilizes spatial gene expression profiles to programmatically derive cell classification labels. These labels serve as ground truth for training an AI model on corresponding DAPI image patches. The report covers the structure of Xenium data outputs, methods for expression-based cell classification, a detailed Python pipeline for sample generation, relevant deep learning architectures, visualization techniques using Napari, and best practices for Machine Learning Operations (MLOps) including data versioning and experiment tracking. The goal is to provide a robust and performant framework for researchers aiming to correlate subtle DAPI morphological features with underlying molecular cell states identified through spatial transcriptomics.  
**1\. Understanding Xenium Data for DAPI-Based AI Training**

1.1. Overview of Xenium Data Generation  
The 10x Genomics Xenium platform performs in situ spatial profiling, enabling the high-resolution mapping of RNA transcripts directly within tissue sections.1 The core technology involves sequential cycles of fluorescent probe hybridization, high-resolution imaging, and probe removal, generating optical signatures that are decoded to identify specific RNA transcripts and their precise locations within the tissue architecture.1 This process achieves subcellular resolution, with transcript localization precision typically below 30 nm in XY and 100 nm in Z.3 The accompanying morphology images, including the DAPI stain for nuclei, are captured at high resolution, with a pixel size of approximately 0.2125 µm/pixel, facilitating detailed subcellular analysis.6 The platform is compatible with both fresh frozen (FF) and formalin-fixed paraffin-embedded (FFPE) tissues, offering flexibility for various sample types.2 The onboard analysis system processes terabytes of raw imaging data to produce readily usable outputs, including transcript locations, cell segmentation boundaries, and morphology images, often within days of starting a run.3  
1.2. Key Output Files and Formats  
A typical Xenium experiment generates a bundle of output files containing morphology images, segmentation results, transcript information, and summary statistics. Understanding the structure and content of these files is crucial for building the data processing pipeline. The experiment.xenium file, a JSON manifest, provides metadata and relative paths to other data files within the output directory.10 Key files relevant to this project include:

* **Morphology Images (DAPI):** The DAPI stain, visualizing cell nuclei, is typically stored as an OME-TIFF file. In recent Xenium onboard analysis (XOA) versions, this might be morphology\_focus\_0000.ome.tif within the morphology\_focus/ directory, especially when multimodal segmentation is used.10 Older or different configurations might output a single morphology.ome.tif or morphology\_focus.ome.tif.11 It is essential to consult the experiment.xenium manifest or associated metadata to confirm the exact file corresponding to the DAPI channel. These OME-TIFF files are 16-bit grayscale and contain a pyramid of image resolutions, tiled for efficient loading and visualization in compatible software.10 This DAPI image serves as the primary input for the AI model. Critically, the pipeline must ensure it selects *only* the DAPI channel image, as other morphology images representing different stains (e.g., boundary markers, RNA interior stains) may also be present in the output.10 Using non-DAPI information would violate the project's constraint of classifying based solely on DAPI patterns.  
* **Cell and Nucleus Segmentation Boundaries:** Xenium analysis provides segmentation boundaries for both nuclei and cells. These are available as polygon vertices (X, Y coordinates in µm) in gzipped CSV (nucleus\_boundaries.csv.gz, cell\_boundaries.csv.gz) or Apache Parquet (nucleus\_boundaries.parquet, cell\_boundaries.parquet) formats.10 Parquet format is generally recommended for better I/O performance compared to gzipped CSV.7 Some outputs might also include a zipped Zarr store (cells.zarr.zip) containing segmentation masks directly, which could simplify loading.11 Each polygon is linked to a unique cell\_id. The vertices are typically listed in clockwise order, with the first point repeated at the end to denote a closed polygon.7 The availability of *both* nucleus and cell boundaries is a key feature of the Xenium output, as it directly enables the analysis of the *perinuclear region*, defined geometrically as the area within the cell boundary but outside the nucleus boundary. Loading and processing both sets of boundaries is therefore essential for extracting image patches corresponding to the user's specific region of interest.  
* **Cell Summary Data:** Files like cells.csv.gz or cells.parquet provide summary information for each segmented cell.10 This includes the unique cell\_id, centroid coordinates (x\_centroid, y\_centroid in µm), cell and nucleus area (cell\_area, nucleus\_area in µm²), transcript counts (total and control probes), and segmentation method used.10 This file links segmentation polygons to expression data and provides basic cell metrics useful for quality control.  
* **Transcript Data:** Detailed information for each decoded transcript is stored in files like transcripts.csv.gz, transcripts.parquet, or transcripts.zarr.zip.10 Each row typically contains the transcript's location (x\_location, y\_location, z\_location in µm), the cell\_id it was assigned to (or \-1 if unassigned 13), the feature\_name (gene or control probe), a Phred-scaled quality value (qv), a flag indicating if it overlaps the nucleus (overlaps\_nucleus), and the distance to the nearest nucleus boundary (nucleus\_distance).10 While this raw transcript data isn't directly used as input for the final DAPI classification model, it forms the basis for the cell-feature matrix used to generate the training labels. The qv score is crucial for filtering low-confidence transcripts (a default threshold of Q20 is often applied for matrix generation).7  
* **Cell-Feature Matrix:** This matrix quantifies the number of transcripts per gene (or control feature) detected within each cell's segmented boundary. It is provided in multiple formats: a directory containing files in sparse Matrix Market Exchange (MTX) format (cell\_feature\_matrix/ with matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz), a Hierarchical Data Format 5 (HDF5) file (cell\_feature\_matrix.h5), or a zipped Zarr store (cell\_feature\_matrix.zarr.zip).6 The HDF5 and Zarr formats are generally more efficient for loading and handling large sparse matrices compared to the MTX format.7 This matrix is the primary input for the expression-based cell classification step that generates the ground truth labels for the DAPI image patches. Note that the matrix typically includes counts from control probes (e.g., negative control probes targeting non-biological sequences, negative control codewords) which should be identified using the features.tsv.gz file and excluded during biological analysis.7

1.3. Coordinate Systems and Data Linking  
The spatial data within Xenium outputs (segmentation vertices, transcript locations, cell centroids) are primarily reported in micrometers (µm).10 The coordinate system origin is typically defined as the top-left corner of the image.7 To relate these µm coordinates to the pixel grid of the DAPI image, division by the pixel size (approximately 0.2125 µm/pixel) is necessary.7 Maintaining consistency between these coordinate systems (µm for spatial data, pixels for image arrays) is critical throughout the pipeline. The central identifier linking all these data modalities – cell summaries, nucleus boundaries, cell boundaries, the cell-feature matrix columns, and transcript assignments – is the cell\_id.10  
1.4. Accessing and Loading Key Data Components in Python  
A performant Python pipeline requires efficient loading of these potentially large data files. Several libraries are well-suited for this:

* **Images (OME-TIFF):** tifffile is a robust library for reading various TIFF formats, including the multi-resolution OME-TIFFs generated by Xenium.7 aicsimageio is another excellent option built on tifffile with additional conveniences for bioimaging formats. Napari itself can load OME-TIFFs directly.  
* **Tabular Data (CSV, Parquet):** pandas is the standard library for handling tabular data. For improved performance with Parquet files, pandas should be used in conjunction with an engine like pyarrow or fastparquet.15 pyarrow provides highly optimized readers for Parquet. Using Parquet or Zarr formats where available is strongly recommended over gzipped CSV for faster data loading, which contributes significantly to overall pipeline performance.10  
* **Matrices (HDF5, Zarr, MTX):** anndata is the de facto standard for storing and manipulating single-cell and spatial transcriptomics data in Python, including the cell-feature matrix (adata.X), cell metadata (adata.obs), and gene metadata (adata.var). It can read HDF5 (.h5ad) and Zarr formats efficiently. For direct HDF5 interaction, h5py can be used. Zarr stores can be accessed using the zarr library. MTX files can be read using scipy.io.mmread and then loaded into an AnnData object.  
* **Specialized Libraries:** Packages like spatialdata and its associated I/O library spatialdata-io offer dedicated readers for Xenium data, parsing multiple file types into a unified SpatialData object, which can simplify data management and integration.17 While Bioconductor packages like XeniumIO exist for R 15, the focus here is on Python solutions.

Handling large files, particularly the multi-gigabyte OME-TIFF images and potentially large transcript lists or boundary files, requires careful memory management. Techniques like lazy loading or memory mapping for images (supported by tifffile and Napari) and using memory-efficient data types in pandas (e.g., appropriate integer sizes, categorical types) are important considerations.

**Table 1: Key Xenium Output Files for DAPI-Based AI Training**

| File Pattern/Directory | Format(s) | Description | Key Content/Columns | Relevance to Pipeline |
| :---- | :---- | :---- | :---- | :---- |
| morphology\_focus\_0000.ome.tif *or similar* | OME-TIFF | DAPI nucleus stain image | Pixel intensities, multi-resolution pyramid, metadata | Primary input image data for AI model |
| nucleus\_boundaries.parquet | Parquet (or CSV.gz) | Polygon boundaries for segmented nuclei | cell\_id, vertex\_x (µm), vertex\_y (µm) | Defines nucleus region for patch extraction/masking |
| cell\_boundaries.parquet | Parquet (or CSV.gz) | Polygon boundaries for segmented cells | cell\_id, vertex\_x (µm), vertex\_y (µm) | Defines cell region for patch extraction/perinuclear mask |
| cells.parquet | Parquet (or CSV.gz) | Summary statistics for each cell | cell\_id, x\_centroid, y\_centroid, nucleus\_area, cell\_area, transcript\_counts | Links IDs, provides centroids and QC metrics |
| transcripts.parquet | Parquet (or CSV.gz) | Decoded transcript locations and assignments | transcript\_id, cell\_id, feature\_name, x\_location, y\_location, z\_location, qv, overlaps\_nucleus | Source data for cell-feature matrix (label generation) |
| cell\_feature\_matrix.h5 | HDF5 (or Zarr, MTX) | Gene expression counts per cell (sparse matrix) | Rows: features (genes/controls), Columns: cell\_id, Values: transcript counts (typically Q20 filtered) | Input data for expression-based cell classification |

*Note: Exact filenames and availability of formats (Parquet, Zarr) may vary depending on the Xenium Onboard Analysis software version and experimental setup. Always consult the experiment.xenium manifest for definitive paths.*

**2\. Cell Classification Strategy Using Gene Expression**

2.1. Rationale  
The core idea of this project is to train an AI model to recognize cell states based only on DAPI staining, using ground truth labels derived from gene expression. This approach is predicated on the hypothesis that different cellular states, characterized by distinct gene expression profiles, may manifest subtle but learnable differences in nuclear morphology, chromatin texture, or DAPI signal distribution in the perinuclear space. Gene expression provides a powerful, objective measure of cell identity and state, potentially capturing biological distinctions (e.g., activation status, cell cycle phase, differentiation trajectory) that are not readily apparent from morphology alone. By using expression-derived labels, the aim is to guide the AI model to detect these subtle DAPI correlates, effectively leveraging the "invisible" mRNA information mentioned in the user query to interpret the visible DAPI stain.  
2.2. Leveraging the Cell-Feature Matrix  
The starting point for deriving expression-based labels is the cell-feature matrix (e.g., cell\_feature\_matrix.h5, cell\_feature\_matrix.zarr.zip, or the MTX directory).7 This matrix represents the quantified gene expression for each segmented cell. In the Python ecosystem, this data is typically loaded into an AnnData object using the scanpy library.17 The AnnData structure conveniently stores the count matrix (adata.X), cell-level metadata (adata.obs, where classification labels will be stored), and gene-level metadata (adata.var). Other frameworks like Seurat (often used in R, but accessible via tools like rpy2 20\) or Giotto 14 also provide functionalities for handling and analyzing such data.  
2.3. Preprocessing Gene Expression Data  
Raw gene expression counts require preprocessing to remove noise and technical artifacts before classification. Standard steps, commonly implemented using scanpy 17, include:

1. **Quality Control (QC) Filtering:** Remove cells with very low total transcript counts or an unusually high proportion of control probes/codewords (indicating potential technical issues). Filter out genes detected in too few cells to be informative. It is crucial to identify and exclude negative control features (probes targeting non-biological sequences like ERCC, or unused codewords) listed in the features.tsv.gz file associated with the MTX format, as these do not represent biological signal.7  
2. **Normalization:** Correct for differences in sequencing depth or capture efficiency between cells. Common methods include counts-per-million (CPM) scaling or using scanpy.pp.normalize\_total.17  
3. **Transformation:** Stabilize variance and make distributions more symmetrical. Logarithmic transformation (log(counts \+ 1)) using scanpy.pp.log1p is standard.17 Alternatively, a square-root transformation has sometimes been recommended for count data.19  
4. **Feature Selection:** Identify Highly Variable Genes (HVGs) using methods like scanpy.pp.highly\_variable\_genes. This focuses downstream analysis on genes exhibiting significant biological variation across cells, reducing noise and computational burden.  
5. **Dimensionality Reduction:** Reduce the high-dimensional gene space to a lower-dimensional representation while retaining major axes of variation. Principal Component Analysis (PCA) applied to the selected HVGs is a standard first step (scanpy.pp.pca).17

2.4. Methods for Expression-Based Cell Typing  
Once the data is preprocessed, several strategies can be employed to classify cells based on their expression profiles:

* **Unsupervised Clustering:** This data-driven approach identifies groups of cells with similar expression patterns without prior knowledge of cell types. Graph-based algorithms like Leiden (scanpy.tl.leiden 17) or Louvain (scanpy.tl.louvain 19) are widely used. These algorithms first build a neighborhood graph connecting cells with similar expression profiles (typically in PCA space, using scanpy.pp.neighbors 17) and then partition this graph into communities or clusters. The biological meaning of these clusters must then be inferred by identifying marker genes (genes preferentially expressed in specific clusters, using scanpy.tl.rank\_genes\_groups) and comparing them to known markers for the tissue context. Visualization techniques like UMAP (scanpy.tl.umap, scanpy.pl.umap 17) are essential for exploring cluster separation and relationships. This method is powerful for discovering novel or unexpected cell states within the data but requires careful biological interpretation of the resulting clusters.  
* **Supervised Classification / Label Transfer:** If a well-annotated reference dataset (e.g., single-cell RNA-seq from the same tissue type) is available, supervised methods can be used to assign cell type labels to the Xenium data. Tools like scmap, SingleR, or scanpy.tl.ingest compare the expression profiles of Xenium cells to the reference profiles and transfer the most likely labels. This approach leverages existing biological knowledge but its accuracy depends heavily on the quality, relevance, and completeness of the reference dataset. It might fail to identify cell states unique to the spatial context or not present in the reference.  
* **Marker Gene-Based Scoring:** Cell types can sometimes be identified by scoring each cell based on the expression of known canonical marker gene sets using functions like scanpy.tl.score\_genes. This is simpler than full clustering or classification but relies on pre-defined, potentially incomplete or context-dependent marker lists.

The choice between these methods carries significant implications. Unsupervised clustering is exploratory but requires post-hoc annotation and may be influenced by technical factors. Supervised methods provide direct biological labels but are constrained by the reference data. Marker scoring is straightforward but limited by the chosen markers. The optimal strategy may depend on the specific biological question, the tissue context, the quality of the Xenium data, and the availability of suitable reference datasets. It's also important to consider that the Xenium gene panel itself, whether pre-designed or custom 1, fundamentally determines the available expression information. If the panel lacks genes crucial for distinguishing the cell states of interest, no classification method will be able to resolve them effectively, limiting the potential biological relevance of the labels generated for the DAPI model training.

2.5. Assigning Class Labels to Cells/Nuclei  
Regardless of the method chosen, the outcome is a categorical class label assigned to each cell\_id. This label (e.g., 'Cluster 1', 'T-cell', 'High\_MarkerX\_Score') is typically stored as a new column in the adata.obs DataFrame of the AnnData object (e.g., adata.obs\['leiden\_labels'\]). These expression-derived labels constitute the ground truth for training the subsequent DAPI image classification model. It is worth noting that the cell-feature matrix used for this classification is typically generated using transcripts passing a default quality threshold (e.g., Q20 7). While usually sufficient, the raw transcript file contains all decoded transcripts, including lower-quality ones.7 Advanced users could, in principle, regenerate the matrix with different filtering criteria if specific hypotheses warrant exploring information from lower-confidence transcripts, though this adds considerable complexity.  
**3\. Python Pipeline for Training Sample Generation**

3.1. Overall Workflow Logic  
The core task is to extract DAPI image patches corresponding to each cell (specifically, its nucleus and perinuclear region) and pair them with the expression-derived class label. A performant Python pipeline can achieve this through the following steps:

1. **Load Data:** Load the necessary input files identified in Section 1: the DAPI OME-TIFF image, nucleus boundary polygons, cell boundary polygons, and the AnnData object containing the expression-derived labels in adata.obs linked by cell\_id. Prioritize loading tabular data from Parquet or Zarr formats for speed.  
2. **Iterate Cells:** Loop through each cell\_id that is present in both the segmentation data (possessing nucleus and cell boundaries) and the classified AnnData object (having an assigned label).  
3. **Retrieve Polygons:** For the current cell\_id, fetch its corresponding nucleus and cell boundary polygon vertices (in µm coordinates).  
4. **Convert Coordinates:** Transform the polygon vertex coordinates from µm to pixel coordinates using the known pixel size of the DAPI image (e.g., pixel\_coords \= micron\_coords / 0.2125).7  
5. **Define Region of Interest (ROI):**  
   * Determine the bounding box encompassing the cell polygon in pixel coordinates. This defines the maximum extent of the patch.  
   * Crucially, define the precise nucleus and perinuclear regions. This requires geometric operations: represent the nucleus and cell boundaries as polygon objects (e.g., using shapely). The perinuclear region is the geometric difference: perinuclear\_polygon \= cell\_polygon.difference(nucleus\_polygon). This operation requires careful implementation, as real segmentations can result in complex shapes, multi-part polygons, or areas where nucleus and cell boundaries touch or slightly overlap. Robust geometric libraries are essential here.  
   * Rasterize these polygons (nucleus, perinuclear, or combined) into binary masks matching the image grid within the bounding box (e.g., using skimage.draw.polygon or cv2.fillPoly).  
6. **Extract Image Patch:** Crop the DAPI image array using the calculated bounding box coordinates. Apply the generated binary mask(s) if precise nucleus/perinuclear pixel isolation is desired, setting background pixels to zero or another constant. Alternatively, extract a fixed-size patch centered on the nucleus centroid.  
7. **Retrieve Label:** Look up the expression-derived class label associated with the current cell\_id from the adata.obs DataFrame.  
8. **Data Augmentation (Optional but Recommended):** Apply on-the-fly data augmentations to the extracted patch. For microscopy images, suitable augmentations include rotations, flips, scaling, brightness/contrast adjustments, Gaussian noise, and elastic deformations. This artificially increases dataset size and variability, helping the model generalize better and reducing overfitting, especially important if the number of cells per class is limited. Libraries like albumentations provide a wide range of relevant transformations.  
9. **Save Sample:** Store the processed image patch (potentially along with its mask) and its corresponding class label. The storage format should be optimized for efficient loading during deep learning model training. Options include:  
   * Saving individual image files (e.g., PNG, TIFF) into class-specific subdirectories. Simple but potentially slow for large datasets due to I/O overhead during training.  
   * Appending patches and labels to larger container files like HDF5 (h5py) or Zarr (zarr).  
   * Using specialized data loading libraries like webdataset, which store samples in tar archives, offering good performance for large datasets, especially in distributed training settings. Container formats generally offer better I/O performance during training compared to millions of small files.

3.2. Recommended Python Libraries  
Implementing this pipeline involves leveraging several powerful Python libraries:  
**Table 2: Recommended Python Libraries for the Pipeline**

| Library | Purpose | Key Functions/Features | Relevance |
| :---- | :---- | :---- | :---- |
| tifffile | Reading OME-TIFF images | TiffFile(), asarray(), multi-resolution access, metadata reading | Loading the primary DAPI image input |
| pandas | Handling tabular data (CSV, metadata) | DataFrame, read\_csv(), read\_parquet() (with engine) | Loading cell summaries, boundary coordinates (CSV) |
| pyarrow | High-performance Parquet I/O | Used as engine for pandas.read\_parquet() | Fast loading of Parquet boundary/transcript/cell summary files |
| anndata | Storing/manipulating expression data | AnnData object, read\_h5ad(), read\_zarr(), .X, .obs, .var attributes | Managing cell-feature matrix and derived labels |
| h5py / zarr | Reading HDF5 / Zarr files | Direct access to HDF5/Zarr structures | Loading matrix/boundaries if not using anndata/pandas, saving patches |
| shapely | Geometric operations on polygons | Polygon(), .buffer(), .difference(), .centroid, .bounds | Defining nucleus/cell shapes, calculating perinuclear region |
| geopandas | Geospatial operations on tabular data | GeoDataFrame, spatial indexing, integrates shapely | Alternative for managing polygon data linked to cell metadata |
| numpy | Numerical computing, array manipulation | ndarray, array slicing, mathematical operations | Core library for image patch manipulation, coordinate calculations |
| scikit-image | Image processing | io.imread, draw.polygon, transform.resize, morphology operations | Image loading (alternative), polygon rasterization, basic processing |
| OpenCV (cv2) | Computer vision, image processing | imread(), fillPoly(), warpAffine(), optimized image operations | Alternative/complement to scikit-image, potentially faster functions |
| PIL (Pillow) | Basic image manipulation and saving | Image.open(), Image.save(), format conversion | Saving extracted patches as standard image files (e.g., PNG) |
| scanpy | Single-cell/spatial expression analysis | pp.\* (preprocessing), tl.\* (tools like clustering, UMAP), pl.\* (plotting) | Generating expression-derived class labels (Section 2\) |
| albumentations | Image augmentation | Large library of diverse augmentation techniques suitable for bioimages | Applying data augmentation to training patches |
| joblib/multiprocessing | Parallel processing | Parallel(), delayed(), Pool() | Speeding up patch extraction by using multiple CPU cores |
| webdataset | Efficient deep learning data loading | Reading data from tar archives | Recommended format for storing/loading large numbers of patches |

**3.3. Implementation Details & Code Snippets (Conceptual)**

* **Loading Data:**  
  Python  
  import tifffile  
  import pandas as pd  
  import anndata as ad  
  import pyarrow.parquet as pq

  \# Load DAPI image (lazy loading might be needed for large images)  
  with tifffile.TiffFile('path/to/morphology\_focus\_0000.ome.tif') as tif:  
      dapi\_image \= tif.asarray(level=0) \# Load highest resolution level  
      metadata \= tif.ome\_metadata  
      \# Extract pixel size from metadata (example, actual parsing needed)  
      pixel\_size\_x \= float(metadata\['Image'\]\['Pixels'\])  
      pixel\_size\_y \= float(metadata\['Image'\]\['Pixels'\])  
      \# Assuming square pixels for simplicity  
      pixel\_size \= pixel\_size\_x

  \# Load boundaries (using pyarrow for speed)  
  nucleus\_boundaries\_df \= pq.read\_table('path/to/nucleus\_boundaries.parquet').to\_pandas()  
  cell\_boundaries\_df \= pq.read\_table('path/to/cell\_boundaries.parquet').to\_pandas()

  \# Load AnnData with labels  
  adata \= ad.read\_h5ad('path/to/classified\_adata.h5ad')

* **Coordinate Conversion & Polygon Creation (inside cell loop):**  
  Python  
  from shapely.geometry import Polygon

  cell\_id \= '...' \# Current cell ID  
  label \= adata.obs.loc\[cell\_id, 'expression\_label'\]

  \# Get vertices for this cell  
  nucleus\_verts\_um \= nucleus\_boundaries\_df\[nucleus\_boundaries\_df\['cell\_id'\] \== cell\_id\]\[\['vertex\_x', 'vertex\_y'\]\].values  
  cell\_verts\_um \= cell\_boundaries\_df\[cell\_boundaries\_df\['cell\_id'\] \== cell\_id\]\[\['vertex\_x', 'vertex\_y'\]\].values

  \# Convert to pixel coordinates (ensure correct axis order, typically Y, X for numpy)  
  nucleus\_verts\_px \= nucleus\_verts\_um / pixel\_size  
  cell\_verts\_px \= cell\_verts\_um / pixel\_size  
  nucleus\_verts\_px\_yx \= nucleus\_verts\_px\[:, ::-1\] \# Swap to Y, X order  
  cell\_verts\_px\_yx \= cell\_verts\_px\[:, ::-1\]

  \# Create Shapely polygons  
  nucleus\_poly \= Polygon(nucleus\_verts\_px\_yx)  
  cell\_poly \= Polygon(cell\_verts\_px\_yx)

  \# Define perinuclear region (handle potential errors/invalid geometry)  
  try:  
      perinuclear\_poly \= cell\_poly.difference(nucleus\_poly)  
  except Exception as e:  
      print(f"Warning: Geometry issue for cell {cell\_id}: {e}")  
      \# Handle error: skip cell, use cell poly only, etc.  
      perinuclear\_poly \= None \# Example handling

* **Patch Extraction & Masking:**  
  Python  
  import numpy as np  
  from skimage.draw import polygon as draw\_polygon

  \# Get bounding box of the cell  
  min\_y, min\_x, max\_y, max\_x \= \[int(c) for c in cell\_poly.bounds\]

  \# Ensure bounds are within image dimensions  
  min\_y, min\_x \= max(0, min\_y), max(0, min\_x)  
  max\_y, max\_x \= min(dapi\_image.shape, max\_y), min(dapi\_image.shape\[1\], max\_x)

  \# Extract patch  
  patch \= dapi\_image\[min\_y:max\_y, min\_x:max\_x\]

  \# Create masks (relative to patch coordinates)  
  nucleus\_mask \= np.zeros(patch.shape, dtype=np.uint8)  
  rr, cc \= draw\_polygon(nucleus\_verts\_px\_yx\[:, 0\] \- min\_y, nucleus\_verts\_px\_yx\[:, 1\] \- min\_x, shape=patch.shape)  
  nucleus\_mask\[rr, cc\] \= 1

  \# Similarly create perinuclear mask if needed...

  \# Apply mask (example: keep only nucleus pixels)  
  \# masked\_patch \= patch \* nucleus\_mask

* **Saving (Example: Individual Files):**  
  Python  
  from PIL import Image  
  import os

  output\_dir \= f'./training\_patches/{label}'  
  os.makedirs(output\_dir, exist\_ok=True)  
  patch\_filename \= os.path.join(output\_dir, f'{cell\_id}\_patch.png')  
  \# Convert patch to appropriate format (e.g., scale to 8-bit for PNG)  
  patch\_img \= Image.fromarray(np.uint8(patch / patch.max() \* 255)) \# Example scaling  
  patch\_img.save(patch\_filename)

3.4. Performance Considerations  
Generating potentially millions of patches requires attention to performance:

* **Parallel Processing:** The patch extraction process is highly parallelizable across cells. Use Python's multiprocessing module or libraries like joblib to distribute the workload across multiple CPU cores, significantly reducing processing time.  
* **Efficient I/O:** As mentioned, use performant file formats (Parquet, Zarr, HDF5, WebDataset) for both input data and output patches. Minimize redundant disk reads/writes within the processing loop.  
* **Memory Management:** For very large DAPI images that exceed available RAM, load the image in chunks or use memory mapping (memmap option in tifffile or numpy.memmap). Ensure data structures used within the loop (e.g., pandas DataFrames) are memory-efficient. Release memory for objects that are no longer needed.

**4\. Deep Learning Models for DAPI Image Analysis**

4.1. Task Definition  
The user query mentions "segment and classify cells only based on the dapi staining". This phrasing could imply two distinct goals:

1. **Classification:** Using the provided Xenium cell/nucleus segmentations to define input image patches (containing nucleus and perinuclear DAPI signal) and classifying these patches according to the expression-derived labels.  
2. **Segmentation \+ Classification:** Performing *de novo* segmentation of cells/nuclei using *only* the DAPI image, followed by classification of the segmented objects.

Given that Xenium already provides high-quality segmentations which are used to generate the expression labels, the more direct interpretation and likely primary goal is **classification** of patches derived from these existing segmentations. Performing accurate *de novo* cell segmentation based solely on DAPI is challenging, as DAPI primarily stains the nucleus, making the cytoplasmic boundary difficult to delineate without additional markers (which Xenium *does* use for its default segmentation 1). This report will primarily focus on the classification task, assuming patches are extracted using Xenium segmentations, but will also mention architectures relevant for segmentation if that becomes a necessary or alternative goal.

4.2. Input Data Representation  
The input to the AI model will be 2D image patches extracted from the DAPI channel. Key considerations include:

* **Patch Size:** Fixed size patches (e.g., 64x64, 128x128, 256x256 pixels) are standard for CNNs. The optimal size depends on the scale of relevant features and available GPU memory.  
* **Content:** Should the patch encompass the entire cell based on the cell boundary? Or be centered on the nucleus?  
* **Masking/Channels:** Should the input be the raw DAPI intensity within the patch? Or should masks be used to explicitly separate nucleus and perinuclear regions? Options include:  
  * Single-channel input: Raw DAPI intensity in the patch.  
  * Masked input: DAPI intensity only where the nucleus or cell mask is active (background zeroed out).  
  * Multi-channel input: e.g., Channel 1 \= Nucleus DAPI, Channel 2 \= Perinuclear DAPI. This explicitly provides regional information to the model. Experimentation may be needed to determine the most informative input representation.

4.3. Architectures for Classification  
Standard Convolutional Neural Networks (CNNs) are well-suited for image classification tasks:

* **ResNet Variants (Residual Networks):** Architectures like ResNet18, ResNet34, ResNet50 are robust and widely used baselines. Deeper models capture more complex features but risk overfitting, especially if the number of distinct DAPI patterns correlating with expression labels is limited. Starting with smaller variants (ResNet18/34) is advisable.  
* **EfficientNet:** A family of models designed to balance accuracy and computational efficiency, potentially offering good performance with fewer parameters.  
* **Other CNNs:** VGG, DenseNet, etc., could also be considered.

These architectures need adaptation for single-channel grayscale input (modifying the first convolutional layer) unless a multi-channel input strategy (e.g., nucleus/perinuclear) is adopted.

The core challenge lies in the biological hypothesis: can subtle differences in DAPI staining (nuclear texture, shape, perinuclear signal) reliably distinguish between cell states defined by potentially complex gene expression signatures? DAPI primarily reflects DNA content and chromatin condensation. While these correlate with some cellular processes (e.g., cell cycle, apoptosis), the link to arbitrary expression-defined states might be weak or inconsistent. The model might inadvertently learn spurious correlations related to segmentation quality, cell density, or image artifacts if not carefully trained and validated. Rigorous testing and interpretability methods (e.g., saliency maps) are crucial to ensure the model learns biologically relevant DAPI features.

4.4. Architectures for Segmentation (If needed)  
If the goal shifts to performing segmentation based only on DAPI (a harder task), architectures designed for pixel-level prediction are required:

* **U-Net and Variants:** The U-Net architecture is the standard for biomedical image segmentation due to its encoder-decoder structure with skip connections, which effectively combines contextual and localization information. Variants like Attention U-Net or U-Net++ offer potential improvements.  
* **nnU-Net:** A framework that automatically configures U-Net architectures (including preprocessing, network topology, and training schemes) based on the dataset characteristics, often achieving state-of-the-art results.  
* **Cellpose:** While often used as a pre-trained tool, Cellpose employs a U-Net style architecture.22 It could potentially be fine-tuned on DAPI-specific data, although its original training included cytoplasmic signals.

Training such models requires pixel-level ground truth masks. If using Xenium segmentations as ground truth, the model would learn to reproduce those segmentations from DAPI alone – a potentially useful but distinct task from the primary classification goal.

4.5. Training Considerations  
Standard deep learning training practices apply:

* **Loss Function:** Categorical Cross-Entropy for multi-class classification. For segmentation, Dice loss, Focal loss, or combinations are common, especially for handling class imbalance.  
* **Optimizer:** Adam or AdamW with appropriate learning rates are standard choices.  
* **Learning Rate Scheduling:** Techniques like ReduceLROnPlateau (decrease LR when validation metric plateaus) or CosineAnnealingLR can improve convergence.  
* **Regularization:** Dropout, Batch Normalization (integral to many modern CNNs), and Weight Decay help prevent overfitting.  
* **Transfer Learning:** Using models pre-trained on large datasets like ImageNet can sometimes accelerate convergence, but the domain gap between natural images and DAPI microscopy is significant, requiring careful fine-tuning. Pre-training on other large microscopy datasets (e.g., cell painting) might be more beneficial, although finding weights suitable for *DAPI-only classification linked to expression states* is unlikely. Architectures known for texture analysis could also be explored.

**Table 3: Overview of Deep Learning Architectures for DAPI Image Analysis**

| Architecture Type | Example Models | Strengths for DAPI Analysis | Considerations | Relevance to Task |
| :---- | :---- | :---- | :---- | :---- |
| CNN Classifier | ResNet18/34/50, EfficientNet | Proven for image classification, capture hierarchical features | Requires input patch definition (size, content), sensitive to DAPI variability | Primary choice for classifying patches based on expression labels |
| Segmentation | U-Net, nnU-Net, Cellpose | Pixel-level prediction, captures spatial context | Requires pixel-level ground truth masks, DAPI alone is weak for cell boundaries | Needed if *de novo* segmentation from DAPI is the goal/sub-task |
| Texture-Focused CNN | (Specialized variants) | Potentially better at capturing subtle chromatin patterns | Less standard, may require custom implementation or specific pre-training | Potential optimization if standard CNNs struggle with DAPI texture |

**5\. Visualization with Napari**

5.1. Why Napari?  
Napari is a fast, interactive, multi-dimensional image viewer designed for the scientific Python ecosystem.22 Its key strength lies in its ability to easily overlay different data types (images, points, shapes, labels, vectors) in a shared coordinate space. This makes it an ideal tool for visualizing and exploring the complex, multi-modal data generated by Xenium and the outputs of the analysis pipeline, facilitating debugging, quality control, and biological interpretation.  
5.2. Visualizing Inputs  
Napari can be used to interactively inspect the raw data inputs:

* **DAPI Image:** Load the large OME-TIFF DAPI image. Napari's multi-scale rendering allows smooth panning and zooming from whole-slide overview down to subcellular detail without loading the entire high-resolution image into memory at once.  
* **Segmentation Boundaries:** Load nucleus and cell boundary polygons (from Parquet/CSV files, converted to lists of coordinates). Display these as Napari Shapes layers. Boundaries can be color-coded by cell\_id, or assigned default colors to distinguish nucleus vs. cell outlines.  
* **Cell Centroids:** Load coordinates from cells.parquet and display as a Points layer, providing quick spatial references for cells.  
* **Transcripts (Optional):** Load transcript locations from transcripts.parquet. Display as a Points layer, potentially coloring points by gene name (feature\_name) or quality score (qv) to visualize gene expression patterns at the single-molecule level.

5.3. Visualizing Expression-Derived Labels  
A crucial step is visualizing the spatial distribution of the cell classes derived from gene expression:

* **Labeled Segmentations:** Create a Shapes layer using the cell or nucleus boundaries. Use the expression-derived labels stored in adata.obs to color-code each polygon according to its assigned class. This provides an immediate visual map of how different expression-defined cell types are organized within the tissue context.  
* **Linked UMAP/Spatial View:** While requiring more custom code or plugins, it's possible to link the spatial view in Napari with other plots, such as a UMAP embedding of the cells colored by cluster label. Selecting cells in the UMAP could highlight them in the spatial view, and vice-versa, aiding exploration.

5.4. Visualizing Training Samples  
For quality control, develop a simple Napari script or widget to load and display the generated DAPI image patches alongside their assigned expression-derived class label. This allows quick visual verification that the patch extraction and labeling process is working correctly.  
5.5. Visualizing Model Outputs  
Napari is invaluable for evaluating the performance of the trained AI model:

* **Predicted Segmentations:** If the model performs segmentation, overlay the predicted masks as a Napari Labels layer (integer masks where each integer represents a cell instance). Compare visually with the original Xenium segmentations or the DAPI image.  
* **Predicted Classifications:** Color-code the Xenium segmentation polygons (or predicted segmentations) based on the *model's predicted class labels*. Display this side-by-side or overlaid with the ground truth expression-derived labels to visually identify agreements and disagreements (errors).  
* **Interpretability:** Display saliency or attention maps from the classification model as an Image layer overlaid on the corresponding DAPI patch. This helps understand which pixels or regions within the DAPI image the model is focusing on to make its predictions, assessing if it's learning relevant biological features or artifacts.

5.6. Integration  
Integrating data into Napari typically involves converting data loaded with libraries like pandas, numpy, anndata, shapely, etc., into the data structures expected by Napari's layer types (e.g., NumPy arrays for Images/Labels, lists of coordinate arrays for Shapes, Nx2 or Nx3 arrays for Points).

Python

import napari  
import numpy as np  
\# Assuming dapi\_image, nucleus\_polygons\_px, cell\_polygons\_px, labels\_dict are loaded

viewer \= napari.Viewer()

\# Add DAPI image  
viewer.add\_image(dapi\_image, name='DAPI', colormap='gray')

\# Add nucleus boundaries (example: list of Nx2 numpy arrays)  
viewer.add\_shapes(nucleus\_polygons\_px, shape\_type='polygon', edge\_color='blue',  
                  face\_color='transparent', name='Nuclei')

\# Add cell boundaries colored by expression label  
\# Prepare properties dictionary and face\_color cycle mapping labels to colors  
properties \= {'label': \[labels\_dict.get(cell\_id, 'Unknown') for cell\_id in cell\_ids\_in\_order\]}  
\# Define a color map or cycle  
from napari.utils.colormaps import label\_colormap  
num\_labels \= len(set(properties\['label'\]))  
cmap \= label\_colormap(num\_labels=num\_labels)  
\# Need mapping from label string to integer index for cmap  
label\_to\_int \= {label: i for i, label in enumerate(set(properties\['label'\]))}  
face\_colors \= \[cmap.map(label\_to\_int.get(l, 0)) for l in properties\['label'\]\]

viewer.add\_shapes(cell\_polygons\_px, shape\_type='polygon', properties=properties,  
                  face\_color=face\_colors, \# Use mapped colors  
                  edge\_color='yellow', name='Cells (Expression Label)')

napari.run()

Frameworks like spatialdata aim to simplify this process by providing unified data structures and built-in plotting functions that directly interface with Napari.18 Using spatialdata, which includes Xenium readers 17, could streamline the loading and visualization workflow, ensuring consistent handling of coordinate systems across different data modalities. While basic visualization is straightforward, creating highly interactive tools (e.g., clicking a cell to show its patch and label) typically requires developing custom Napari widgets or plugins, involving additional software development effort.

**6\. MLOps Framework for Robust Development**

6.1. Importance of MLOps  
Machine Learning Operations (MLOps) encompasses practices, tools, and principles aimed at building, deploying, and maintaining machine learning models reliably and efficiently. For a complex project involving large datasets (Xenium outputs, generated patches), multi-step processing pipelines, and iterative model development, adopting MLOps practices is crucial for ensuring:

* **Reproducibility:** Tracking data, code, parameters, and environments to ensure experiments can be replicated.  
* **Scalability:** Managing large datasets and computationally intensive tasks efficiently.  
* **Collaboration:** Facilitating teamwork (if applicable) by standardizing workflows and tracking changes.  
* **Reliability:** Automating processes to reduce manual errors and ensure consistent results.

6.2. Data Version Control (DVC)  
Standard version control systems like Git are excellent for code but unsuitable for large data files (like OME-TIFFs or millions of image patches). Tools like DVC (Data Version Control) or Pachyderm address this challenge.

* **DVC:** Works alongside Git. Git tracks the code and small DVC metadata files (.dvc files). These metadata files contain information (like hashes) about the actual data files, which DVC stores in a separate cache or remote storage (e.g., AWS S3, Google Cloud Storage, SSH server, local directory). This allows versioning of datasets and models, linking specific data versions to specific code commits.  
  Bash  
  \# Example DVC usage  
  \# Initialize DVC (once per project)  
  dvc init

  \# Configure remote storage (e.g., S3)  
  dvc remote add \-d myremote s3://my-bucket/dvc-store

  \# Track a data directory (e.g., raw Xenium output)  
  dvc add data/xenium\_output

  \# Track generated patches  
  dvc add data/training\_patches

  \# Track a trained model  
  dvc add models/dapi\_classifier.pth

  \# Commit changes to Git (tracks.dvc files and.gitignore)  
  git add data/.dvc data/training\_patches.dvc models/dapi\_classifier.pth.dvc.gitignore  
  git commit \-m "Add initial data and model tracking"

  \# Push data to remote storage  
  dvc push

  \# To retrieve data associated with a specific commit later:  
  \# git checkout \<commit\_hash\>  
  \# dvc pull

* **Pachyderm:** A more comprehensive, container-centric platform that provides data versioning based on Git-like semantics but also integrates pipelining and data provenance tracking. It might be more powerful for complex, automated pipelines but has a steeper learning curve than DVC.

Implementing DVC requires careful planning of the storage backend. Storing multiple versions of large datasets consumes significant space. Cloud storage offers scalability but incurs costs, while local storage requires sufficient capacity and robust backup strategies. A pragmatic approach is often to version key raw inputs, final generated training sets, and final models, rather than every intermediate file.

6.3. Workflow Management Tools  
The project involves a multi-step pipeline (expression analysis \-\> label generation \-\> patch extraction \-\> model training \-\> evaluation). Manually running these steps is error-prone and difficult to reproduce. Workflow management tools automate and orchestrate these pipelines.

* **Snakemake:** A popular Python-based workflow manager. Workflows are defined using rules that specify inputs, outputs, and the shell command or Python script to execute. Snakemake automatically determines the execution order based on dependencies and supports parallel execution, cluster submission, and Conda environment integration.  
* **Nextflow:** Another widely used workflow manager, particularly in bioinformatics, using a Groovy-based domain-specific language (DSL). It has strong support for containerization (Docker, Singularity) and seamless execution across different platforms (local, HPC, cloud).

Defining the pipeline in Snakemake or Nextflow makes it more robust, reproducible, and easier to modify or re-run with different parameters.

6.4. Experiment Tracking  
Training deep learning models involves experimenting with different architectures, hyperparameters, and data variations. Keeping track of these experiments manually is challenging. Experiment tracking platforms log crucial information for each run.

* **MLflow:** An open-source platform with components for tracking experiments (parameters, metrics, code versions), packaging code (MLflow Projects), managing models (MLflow Models), and a model registry. It can be run locally or on a server. Integration involves adding logging calls to the training script.  
  Python  
  import mlflow

  \# Start an MLflow run  
  with mlflow.start\_run():  
      \# Log parameters  
      mlflow.log\_param("learning\_rate", lr)  
      mlflow.log\_param("batch\_size", batch\_size)  
      mlflow.log\_param("architecture", "ResNet18")

      \# \--- Training loop \---  
      for epoch in range(num\_epochs):  
          \#... train...  
          val\_loss, val\_accuracy \= evaluate(model, val\_loader)  
          \# Log metrics  
          mlflow.log\_metric("val\_loss", val\_loss, step=epoch)  
          mlflow.log\_metric("val\_accuracy", val\_accuracy, step=epoch)  
      \# \--- End Training loop \---

      \# Log the trained model  
      mlflow.pytorch.log\_model(model, "dapi\_model")

* **Weights & Biases (W\&B):** A popular commercial platform (with free tiers for academic use) offering comprehensive experiment tracking, interactive visualizations, hyperparameter optimization sweeps, dataset versioning, model registry, and collaboration features. Integration is similarly achieved via library calls in the training script.

Using MLflow or W\&B provides a systematic way to compare runs, identify best-performing models, and ensure traceability from results back to the exact code, data, and parameters used. While implementing a full MLOps stack can have significant overhead, adopting tools incrementally is often effective. Starting with DVC for data versioning and MLflow/W\&B for experiment tracking typically provides the most immediate benefits for this type of project. Integrating workflow managers with experiment tracking (e.g., having Snakemake rules trigger training scripts that log to MLflow) offers further automation but requires careful configuration.

6.5. Continuous Integration/Continuous Deployment (CI/CD) \- Optional  
For more mature projects or team settings, CI/CD pipelines (e.g., using GitHub Actions, GitLab CI) can automate testing (unit tests, data validation tests), potentially trigger retraining on new data, and manage model deployment, further enhancing reliability and development velocity.  
**Table 4: Comparison of MLOps Tools for Bioimaging AI**

| Category | Tool Examples | Key Features | Pros for Bioimaging | Cons/Considerations |
| :---- | :---- | :---- | :---- | :---- |
| **Data Versioning** | DVC | Git integration, storage agnostic, data caching | Handles large files (OME-TIFF, patches), links data+code | Requires separate storage backend, storage costs/management |
|  | Pachyderm | Container-based, data provenance, integrated pipelines | Full pipeline versioning, scalable | Steeper learning curve, more infrastructure overhead |
| **Workflow Mgmt** | Snakemake | Python-based rules, dependency management, Conda envs | Good Python integration, popular in bioinformatics | Can become complex for very large workflows |
|  | Nextflow | Groovy DSL, strong container/cloud support | Excellent scalability, robust execution | Less Python-native, requires learning Groovy |
| **Experiment Track** | MLflow | Open-source, tracks params/metrics/models, registry | Easy local setup, flexible components | UI less polished than W\&B, requires self-hosting server |
|  | Weights & Biases | Cloud-based, rich UI, sweeps, collaboration | Excellent visualization, easy hyperparameter tuning | Primarily cloud-based (local option exists), potential cost |

**7\. References and Further Reading**

**7.1. 10x Genomics Documentation**

* Xenium Onboard Analysis Output Files: 10  
* Xenium Analysis Guides (including community tools, segmentation): 22  
* Xenium Analyzer Specifications & User Guides: 3  
* Xenium Platform Overview: 2

**7.2. Key Software Libraries**

* Scanpy: [https://scanpy.readthedocs.io/](https://scanpy.readthedocs.io/)  
* AnnData: [https://anndata.readthedocs.io/](https://anndata.readthedocs.io/)  
* Napari: [https://napari.org/](https://napari.org/)  
* SpatialData: [https://spatialdata.scverse.org/](https://spatialdata.scverse.org/) 18  
* SpatialData-IO (incl. Xenium reader): [https://spatialdata.scverse.org/projects/io/](https://spatialdata.scverse.org/projects/io/) 17  
* Tifffile: [https://github.com/cgohlke/tifffile](https://github.com/cgohlke/tifffile)  
* Scikit-image: [https://scikit-image.org/](https://scikit-image.org/)  
* Shapely: [https://shapely.readthedocs.io/](https://shapely.readthedocs.io/)  
* DVC (Data Version Control): [https://dvc.org/](https://dvc.org/)  
* MLflow: [https://mlflow.org/](https://mlflow.org/)  
* Weights & Biases: [https://wandb.ai/](https://wandb.ai/)  
* Snakemake: [https://snakemake.readthedocs.io/](https://snakemake.readthedocs.io/)  
* Nextflow: [https://www.nextflow.io/](https://www.nextflow.io/)  
* Albumentations: [https://albumentations.ai/](https://albumentations.ai/)

**7.3. Relevant Publications & Preprints**

* Janesick A et al. High resolution mapping of the breast cancer tumor microenvironment using integrated single cell, spatial and in situ analysis of FFPE tissue. bioRxiv 2022\. doi: [https://doi.org/10.1101/2022.10.06.510405](https://doi.org/10.1101/2022.10.06.510405) 3 (Example of integrated analysis, potentially relevant context)  
* *Search PubMed/bioRxiv/arXiv for recent papers on: "Xenium data analysis", "spatial transcriptomics cell classification", "deep learning DAPI morphology", "MLOps computational biology".*

**7.4. Tutorials**

* Analyzing Xenium Data (Examples):  
  * Squidpy Tutorial: 17  
  * Giotto Suite Tutorial: 14  
  * Seurat Vignette (includes Xenium): 20  
  * stLearn Tutorial: 19  
  * SpatialData Xenium Example: 18  
* Napari Tutorials: [https://napari.org/stable/tutorials/index.html](https://napari.org/stable/tutorials/index.html)  
* MLOps Tool Tutorials: Refer to documentation links in 7.2.  
* 10x Genomics Analysis Guides (Post-Xenium analysis, community tools): 9

Conclusion  
This report outlines a performant Python-based pipeline to generate training data for an AI model aimed at classifying cells based solely on DAPI staining, using ground truth labels derived from Xenium spatial transcriptomics data. The approach leverages the rich multi-modal outputs of the Xenium platform, specifically linking gene expression profiles (via the cell-feature matrix) to cell morphology (via DAPI images and segmentation boundaries). Key steps involve understanding the Xenium data structure, implementing robust expression-based cell classification using tools like Scanpy, carefully extracting corresponding DAPI image patches (including nucleus and perinuclear regions) using libraries like Shapely and Scikit-image, and training suitable deep learning models (e.g., ResNet). Visualization with Napari is crucial for quality control, debugging, and interpreting results across different data modalities. Furthermore, adopting MLOps practices, particularly data versioning with DVC and experiment tracking with MLflow or W\&B, is strongly recommended for ensuring reproducibility and managing the complexity of the project.  
While technically feasible, the success of this approach hinges on the underlying biological hypothesis: that subtle, learnable patterns in DAPI staining correlate reliably with expression-defined cell states. This correlation may not always hold true or might be confounded by technical factors. Therefore, rigorous validation, careful model interpretation, and consideration of the limitations imposed by the chosen gene panel and the DAPI staining itself are essential throughout the project. By combining state-of-the-art spatial transcriptomics, computational analysis, and machine learning, this pipeline provides a framework for exploring the intricate relationships between molecular state and cellular morphology at subcellular resolution.

#### **Works cited**

1. 10x Genomics Xenium Analyzer \- OSTR \- National Cancer Institute, accessed April 23, 2025, [https://ostr.ccr.cancer.gov/emerging-technologies/spatial-biology/xenium/](https://ostr.ccr.cancer.gov/emerging-technologies/spatial-biology/xenium/)  
2. Xenium In Situ Platform \- 10x Genomics, accessed April 23, 2025, [https://www.10xgenomics.com/platforms/xenium](https://www.10xgenomics.com/platforms/xenium)  
3. Xenium Analyzer \- 10x Genomics, accessed April 23, 2025, [https://cdn.10xgenomics.com/image/upload/v1670300547/support-documents/CG000630\_XeniumAnalyzer\_Specifications\_RevB.pdf](https://cdn.10xgenomics.com/image/upload/v1670300547/support-documents/CG000630_XeniumAnalyzer_Specifications_RevB.pdf)  
4. Xenium Workshop, accessed April 23, 2025, [https://genomics.uci.edu/wp-content/uploads/sites/30/PDF\_UCI\_GRT\_Hub\_Xenium\_Workshop\_20240118-120fec8a1ce5134f.pdf](https://genomics.uci.edu/wp-content/uploads/sites/30/PDF_UCI_GRT_Hub_Xenium_Workshop_20240118-120fec8a1ce5134f.pdf)  
5. Xenium Analyzer, accessed April 23, 2025, [https://biotech.illinois.edu/sites/biotech.illinois.edu/files/uploads/XeniumAnalyzer\_Specifications\_0.pdf](https://biotech.illinois.edu/sites/biotech.illinois.edu/files/uploads/XeniumAnalyzer_Specifications_0.pdf)  
6. Xenium Analyzer \- 10x Genomics, accessed April 23, 2025, [https://www.10xgenomics.com/instruments/xenium-analyzer](https://www.10xgenomics.com/instruments/xenium-analyzer)  
7. Xenium File Format Documentation \- 10x Genomics, accessed April 23, 2025, [https://cf.10xgenomics.com/supp/xenium/xenium\_documentation.html](https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html)  
8. Xenium Analyzer \- 10x Genomics, accessed April 23, 2025, [https://cdn.10xgenomics.com/image/upload/v1729792773/support-documents/CG000584\_Xenium\_Analyzer\_UserGuide\_RevG.pdf](https://cdn.10xgenomics.com/image/upload/v1729792773/support-documents/CG000584_Xenium_Analyzer_UserGuide_RevG.pdf)  
9. Getting started with Xenium In Situ, accessed April 23, 2025, [https://cumming.ucalgary.ca/sites/default/files/teams/392/PDFs/10x\_LIT000215\_Xenium-Getting%20Started%20Guide\_8.5x11\_Letter.pdf](https://cumming.ucalgary.ca/sites/default/files/teams/392/PDFs/10x_LIT000215_Xenium-Getting%20Started%20Guide_8.5x11_Letter.pdf)  
10. Understanding Xenium Outputs \- Official 10x Genomics Support, accessed April 23, 2025, [https://www.10xgenomics.com/jp/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-understanding-outputs](https://www.10xgenomics.com/jp/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-understanding-outputs)  
11. Xenium Output Files at a Glance \- Official 10x Genomics Support, accessed April 23, 2025, [https://www.10xgenomics.com/jp/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-at-a-glance](https://www.10xgenomics.com/jp/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-at-a-glance)  
12. Xenium Output Files at a Glance \- Official 10x Genomics Support, accessed April 23, 2025, [https://www.10xgenomics.com/support/software/xenium-onboard-analysis/1.9/analysis/xoa-output-at-a-glance](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/1.9/analysis/xoa-output-at-a-glance)  
13. tile-xenium/README.md at main \- GitHub, accessed April 23, 2025, [https://github.com/maximilian-heeg/tile-xenium/blob/main/README.md](https://github.com/maximilian-heeg/tile-xenium/blob/main/README.md)  
14. 10x Xenium Human Breast Cancer Pre-Release \- Giotto Suite 3.3 documentation, accessed April 23, 2025, [https://giottosuite.readthedocs.io/en/latest/subsections/datasets/xenium\_breast\_cancer.html](https://giottosuite.readthedocs.io/en/latest/subsections/datasets/xenium_breast_cancer.html)  
15. XeniumIO: Import 10X Genomics Xenium Analyzer Data \- Bioconductor, accessed April 23, 2025, [https://bioconductor.org/packages/devel/bioc/vignettes/XeniumIO/inst/doc/XeniumIO.html](https://bioconductor.org/packages/devel/bioc/vignettes/XeniumIO/inst/doc/XeniumIO.html)  
16. XeniumIO: Import and represent Xenium data from the 10X Xenium Analyzer \- Bioconductor, accessed April 23, 2025, [https://bioconductor.org/packages/devel/bioc/manuals/XeniumIO/man/XeniumIO.pdf](https://bioconductor.org/packages/devel/bioc/manuals/XeniumIO/man/XeniumIO.pdf)  
17. Analyze Xenium data — squidpy documentation \- Read the Docs, accessed April 23, 2025, [https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial\_xenium.html](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_xenium.html)  
18. Technology focus: Xenium \- spatialdata \- scverse, accessed April 23, 2025, [https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/technology\_xenium.html](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/technology_xenium.html)  
19. Xenium data analysis with spatial trajectories inference — stLearn 0.4.11 documentation, accessed April 23, 2025, [https://stlearn.readthedocs.io/en/latest/tutorials/Xenium\_PSTS.html](https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)  
20. MargoKapustina/XeniumSpatialAnalysis: R-based Xenium Spatial Analysis Toolkit to assess gene expression gradients \- GitHub, accessed April 23, 2025, [https://github.com/MargoKapustina/XeniumSpatialAnalysis](https://github.com/MargoKapustina/XeniumSpatialAnalysis)  
21. Analysis of Image-based Spatial Data in Seurat \- Satija Lab, accessed April 23, 2025, [https://satijalab.org/seurat/articles/seurat5\_spatial\_vignette\_2](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2)  
22. Introduction to 10x Genomics Xenium In Situ Data Analysis Tools: Continuing Your Journey after Xenium Analyzer, accessed April 23, 2025, [https://www.10xgenomics.com/analysis-guides/continuing-your-journey-after-xenium-analyzer](https://www.10xgenomics.com/analysis-guides/continuing-your-journey-after-xenium-analyzer)  
23. Downstream Analysis of Single Cell RNA-Seq Data, accessed April 23, 2025, [https://genomics.uci.edu/wp-content/uploads/sites/30/Introduction-to-10x-Visium-and-Xenium-spatial-platform-data-analysis-workflow-and-analytical-tools-1.pdf](https://genomics.uci.edu/wp-content/uploads/sites/30/Introduction-to-10x-Visium-and-Xenium-spatial-platform-data-analysis-workflow-and-analytical-tools-1.pdf)