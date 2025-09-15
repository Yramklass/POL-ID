# POL-ID: Deep Learning for Pollen Detection and Classification in the Authentication of South African Honey🍯🔬

**POL-ID** is an end-to-end deep learning pipeline designed to automate the process of melissopalynology (the study of pollen in honey). By analyzing microscopic images of honey slides, this project detects and classifies pollen grains to determine the honey's floral and geographical origin, providing a powerful tool for quality control and authentication.

The system leverages state-of-the-art computer vision models to provide fast, accurate, and reproducible results, significantly reducing the time and expertise required compared to traditional manual methods.



---

## Key Features

* **🎯 High-Performance Pollen Detection:** Utilizes object detection models like **YOLOv8** and **DETR** to accurately locate and bound individual pollen grains on a microscopic slide.
* **🧠 Advanced Pollen Classification:** Employs a novel **parallel fusion architecture** combining a **Swin Transformer** and a **ConvNeXt** model for robust and highly accurate species classification.
* **🤖 Smart Clustering of Unknowns:** Integrates **UMAP** for dimensionality reduction and **HDBSCAN** to intelligently group low-confidence pollen grains, helping to identify rare types or highlight potential new species not present in the training data.
* **📊 Comprehensive Reporting:** Generates a detailed final report including total pollen counts, species ratios (percentages), and a final honey classification (e.g., monofloral, multifloral).

---

## Data Preprocessing 

The raw data used for this project required significant preprocessing to ensure model training was effective and evaluation was fair. Key challenges included:

* **Inconsistent Labeling:** The raw detection data had multiple class labels for pollen, which were standardized to a single class (`0`). The classification labels also had some inconsistencies (e.g. misspellings, incorrect/inconsistent class labels). These were handled through a combination of automated and manual adjustments.
* **Stack-Level Splitting:** To prevent data leakage between training and testing sets, the classification data was split at the "stack" level (multiple images of the same pollen grain at different focal depths).

For a complete and detailed explanation of all preprocessing steps, the rationale behind them, and instructions for handling new data, please see the **[`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)** file.

---

## The Pipeline

The POL-ID pipeline is a sequential process that transforms a raw microscope image into a final authentication report.

**Image Input → [ 1. Detection ] → Cropped Grains → [ 2. Classification ] → Confident & Unconfident Grains → [ 3. Clustering ] → Final Counts → [ 4. Composition Analysis ] → Report Output**

<br>

### 1. Pollen Detection (YOLO / DETR)

A slide image is first fed into an object detection model (**YOLOv8** in the primary pipeline, with a **DETR** version also available). The model is trained to identify and draw bounding boxes around every pollen grain present in the image. The output is a collection of cropped images, each containing a single pollen grain.

### 2. Pollen Classification (Swin Transformer + ConvNeXt Fusion)

Each cropped pollen grain is simultaneously processed by two powerful classification models:

* **Swin Transformer:** A vision transformer that captures global features and complex spatial relationships.
* **ConvNeXt:** A modern convolutional neural network that excels at extracting hierarchical features.

The feature outputs from both models are fused before the final classification layer. This parallel approach leverages the strengths of both architectures, leading to superior accuracy and robustness compared to a single model. Each grain is assigned a species label and a confidence score.

### 3. Low-Confidence Clustering (UMAP + HDBSCAN)

Grains with a classification confidence score below a set threshold are not discarded. Instead, their features are passed through a dimensionality reduction step using **UMAP**. These lower-dimensional points are then fed to the **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm. HDBSCAN groups these uncertain grains based on their visual similarity, creating clusters of "unknown" types. This crucial step ensures that even rare or unclassified pollen contributes to the final analysis.

### 4. Final Composition Analysis

The system aggregates the results from the high-confidence classifications and the low-confidence clusters to generate the final report. The output provides:

* The predicted honey type (e.g., Monofloral Eucalyptus, Multifloral Fynbos).
* A complete list of identified pollen species and their absolute counts.
* The percentage ratio of each pollen species.
* Counts for each cluster of unidentified pollen.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yramklass/POL-ID.git](https://github.com/yramklass/POL-ID.git)
    cd POL-ID
    ```

2.  **Create and activate Conda environments:** This project requires separate environments for detection, classification and full pipeline tasks.
    ```bash
    # Create the detection environment
    conda env create -f detection_environment.yaml
    
    # Create the classification environment
    conda env create -f classification_environment.yaml 

    # Create the full_pipeline environment
    conda env create -f full_pipeline_environment.yaml
    ```
    *Activate the appropriate environment before running scripts (e.g., `conda activate detection`)*.

3.  **Download the Pre-Trained Models** 📂

    The official pre-trained models for this project are available on the [**GitHub Releases page**](https://github.com/yramklass/POL-ID/releases/latest).

    * Navigate to the latest release and download the following files from the **Assets** section:
        * `best_yolov8_model.pt`
        * `best_parallel_fusion_model.pth`

    * Create a new folder named `models` in the root directory of this project and place the downloaded files inside it. The final structure should look like this:
        ```
        POL-ID/
        ├── models/
        │   ├── best_yolov8_model.pt
        │   └── best_parallel_fusion_model.pth
        ├── full_pipeline.py
        └── ... (other project files)
        ```

---

## Usage

The main analysis is run using the `full_pipeline.py` script.

**Important:** Before running, you must **configure the file paths** (for input data, model checkpoints, and output directories) directly within the script itself.

1.  **Activate the correct Conda environment:**
    ```bash
    conda activate full_pipeline_env
    ```
2.  **Run the pipeline on a directory of slide images:**
    ```bash
    python full_pipeline.py /path/to/your/slides_directory
    ```
3.  **Batch Processing:** To run the analysis on all honey samples at once, use the `run_all_pipelines.py` script.
    ```bash
    python run_all_pipelines.py
    ```

For users on a High-Performance Computing (HPC) cluster with a Slurm workload manager, several `.sbatch` scripts are provided to submit jobs for training and analysis (e.g., `run_all_pipelines.sbatch`).

---

## Project Structure

This repository contains scripts for the core pipeline, model training, data preparation, and results analysis.

### 📂 Core Pipeline Scripts
* `full_pipeline.py`: The main end-to-end pipeline using the **YOLOv8** detector.
* `full_pipeline_detr.py`: An alternative end-to-end pipeline using a **DETR** detector.
* `run_all_pipelines.py`: A wrapper script to execute the full pipeline for all honey samples sequentially.

### 🧠 Model Training Scripts
* `yolo_model.py`: Trains the YOLOv8 object detection model.
* `train_rf_detr_model.py` / `deimkit_training.py`: Scripts for training DETR-based object detection models.
* `parallel_fusion_model.py`: Trains the parallel (ConvNeXt + Swin) classification model using a 2-phase strategy (head training, then full fine-tuning).
* `parallel_fusion_model_3_phase.py`: An extension of the above with a 3-phase training strategy.
* `parallel_fusion_raytune.py`: Performs hyperparameter optimization for the parallel fusion model using Ray Tune.
* `sequential_fusion_model.py`: Trains a `CoAtNet` model as an alternative classifier.

### 🛠️ Data Preparation & Utilities
* `prepare_detection_data.py`: Splits the detection dataset into train/val/test sets in YOLO format.
* `crop_pollen.py`: Crops individual pollen grains from images using existing labels.
* `extract_expert_compositions.py`: Parses expert honey composition data from Excel files into a standard CSV format.
* `fix_class_folder_names.py` / `fix_detection_labels.py`: Utility scripts for dataset cleaning.
* `select_samples_by_class_limit.py`: A script to select a subset of taxa for training based on specific criteria.

### 📈 Analysis & Visualization Scripts
* `plot_comparisons.py`: Generates plots comparing the model's output compositions against the expert-defined compositions.
* `plot_expert_compositions.py`: Visualizes the ground-truth honey compositions from the expert data.
* `plot_training_loss.py`: Plots training loss and validation mAP for DETR models.
* `taxon_data_analysis.py`: Plots F1-score vs. number of image stacks for each pollen taxon.

---

## How to Cite

If you use POL-ID in your research, please cite this work:
```bibtex
@misc{polid2025,
  author       = {Yash Ramklass},
  title        = {POL-ID: Deep Learning for Pollen Detection and Classification in
the Authentication of South African Honey},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/yramklass/POL-ID](https://github.com/yramklass/POL-ID)}}
}