
QFA-Based Multi-Level Image Annotation Framework
Overview
This repository presents an experimental framework for evaluating the effectiveness of a QFA-based multi-level image annotation approach. The framework focuses on image segmentation quality, annotation accuracy, and label ranking performance using benchmark datasets derived from the Corel Photo CD collection.

Datasets
The experiments utilize two widely used benchmark datasets:

Corel A Dataset: 203 images

Corel B Dataset: 291 images

Both datasets are subsets of the larger Corel Photo CD collection, which consists of:

5,000 images

50 diverse semantic categories

100 images per category

Dataset source:
https://archive.ics.uci.edu/dataset/119/corel+image+features

Data Partitioning
Each dataset is divided into training and testing sets using a 3:1 ratio.

Image Segmentation
Images are segmented using a modified Otsu thresholding method enhanced by the proposed QFA (Quantum Firefly Algorithm).

Insignificant regions are removed by filtering out smaller segments.

Each image is typically represented by 2 to 5 regions, prioritized based on region size.

Feature Extraction
Each segmented region is described using a 12-dimensional feature vector.

These feature vectors serve as the basis for image annotation and classification.

Annotation Vocabulary
The annotation process uses a vocabulary of 38 frequently occurring semantic keywords.

Experimental Evaluation
The proposed framework enables a comprehensive evaluation in terms of:

Segmentation quality

Annotation accuracy

Label ranking performance

Implementation Details
Development Environment
Platform: Google Colab

Programming Language: Python 2.7

Local Testing Environment
The model was also tested locally for verification using the following setup:

System: Dell Inspiron N5110

Operating System: Windows 10 (64-bit)

Processor: Intel Core i7-2410M @ 2.30 GHz

Memory: 16 GB RAM

Summary
This repository provides a reproducible experimental setup for validating a QFA-enhanced multi-level image annotation system using standard benchmark datasets, ensuring robust and fair performance assessment.


