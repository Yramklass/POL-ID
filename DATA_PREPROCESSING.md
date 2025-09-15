# Data Preprocessing Guide for the POL-ID Project

This document details the challenges encountered with the raw data and the steps taken to clean and prepare it for use in the POL-ID pipeline. It is intended for future researchers who wish to reproduce these results or for data annotators to improve future data collection.

---

## 1. Detection Dataset Issues

### Problem: Inconsistent Class Labels
The raw detection dataset contained multiple numeric labels for what should have been a single "pollen" class. For example, labels like `0`, `1`, `2` were used across different annotation files, all referring to pollen.

### Solution: Label Standardization
The `fix_detection_labels.py` script was created to solve this. It iterates through all `.txt` label files in the dataset and overwrites the class index on each line to `0`, ensuring a single, consistent class for the detector to learn.

---

## 2. Classification Dataset Issues

### Problem: Data Leakage from Image Stacks
The classification dataset consists of "stacks" of images, where each stack contains multiple pictures of the *same pollen grains* taken at different focal depths. A standard random split would be invalid, as it could place images of the same grain in both the training and testing sets, artificially inflating performance metrics.

### Solution: Stack-Level Splitting and the `_box` Convention ⚠️
To prevent this, we implemented a **stack-level split**. This ensures that all images from a single stack are assigned exclusively to one set (either train, validation, or test).

To identify the boundaries of a stack, we relied on a specific file naming convention from the original annotators:
* The last image of every annotated stack was named with a `_box` suffix (e.g., `IMG_1234_box.jpg`). This was ensured through manual inspection and renaming/rearranging files where necessary.
* The `crop_pollen.py` script uses the presence of a `_box` file to confirm the end of one stack and the beginning of the next.

**For supplementary data that did not have this `_box` file**, we had to artificially recreate it. This was done by copying the last image in a stack and renaming the copy to include the `_box` suffix. This is a critical but non-elegant step required to make new data compatible with our existing scripts.

---

## 3. General Data Cleaning (Manual and Scripted)

### Problem: Inconsistent Taxon Names
There were numerous inconsistencies in the data, including:
* Typos in class folder/label names (e.g., "Campanulaceae" vs. "Camanulaceae").
* Inconsistencies in labelled and expert class names (e.g. "Vahlia" vs "Vahlia-Type")
* Differing syntax between expert classifications and class labels (e.g. "Eucalyptus Sp. 1" vs "Eucalyptus_sp_1")
* Supplementary data class label inconsistencies (i.e. class labels containing the "Supp_" prefix)

### Solution: Standardization and Renaming
A significant effort was undertaken to standardize all class names. This involved steps like removing the "Supp_" prefix in processing, and replacing characters and using regular expressions to normalize class labels in expert vs model comparisons. Fixes to label files and folder names were made manually.