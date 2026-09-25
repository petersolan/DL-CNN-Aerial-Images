# Fallen Tree Detection from Aerial Imagery using YOLOv3

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Model-YOLOv3_Darknet-orange.svg)](https://pjreddie.com/darknet/yolo/)
[![Domain](https://img.shields.io/badge/Domain-Geospatial_%2F_Remote_Sensing-green.svg)](#)

> **Master’s Thesis Research:** Automated object detection of storm-damaged trees in urban environments using Deep Learning and Convolutional Neural Networks (CNNs).

---

## 📌 Overview

This repository details the end-to-end computer vision pipeline developed for my Master's dissertation. The project evaluates the viability, speed, and accuracy of automated fallen tree detection from high-resolution aerial imagery in post-disaster urban areas.

![Image Alt text](https://github.com/petersolan/DL-CNN-Aerial-Images/assets/59766852/701994db-dc31-4687-9388-cc0c1b443d80)

to this:

![Image Alt text](https://github.com/petersolan/DL-CNN-Aerial-Images/assets/59766852/fac1fca2-6094-466f-88dc-4ba0f90cd4f6)

Beyond model training, this repository demonstrates how deep learning-based spatial classification can be integrated into automated **Data Engineering pipelines** to accelerate post-disaster geospatial data ingestion and rapid response workflows.

---

## 🛠️ Methodology & Architecture

The object detection pipeline was built using the **YOLOv3** architecture (trained via Darknet on Google Colab GPU instances). 

### Pipeline Workflow
1. **Data Ingestion & Split:** Dataset sourced from NOAA aerial surveys and partitioned using a standard **60 / 20 / 20** split (Train / Validation / Test).
2. **Model Training:** Parameter tuning across batch sizes, iteration steps, and learning schedules on GPU acceleration.
3. **Threshold Calibration:** Validation tuning on Confidence and Intersection over Union (IoU) metrics to optimize precision-recall tradeoffs.
4. **Testing & Inference:** Evaluating final metrics across independent test sets using confusion matrices and Precision-Recall Gain curves.

---

## 📊 Data Source

* **Source:** High-Resolution Post-Disaster Aerial Imagery via [NOAA InPort (Item 65224)](https://www.fisheries.noaa.gov/inport/item/65224)
* **Dataset Partitioning:**
  * **Training Set:** 60% (~600 images)
  * **Validation Set:** 20% (~200 images / labels)
  * **Test Set:** 20% (~200 images / labels)

---

## ⚡ Implementation & Experiments

### 1. Training Phase
* **Environment:** Darknet framework on Google Colab (Tesla GPU runtime).
* **Batch Size:** `64`
* **Dataset Size:** 600 annotated images.
* **Execution Time:** ~10 hours across continuous GPU sessions.
* **Artifact Output:** Serialized `.weights` file capturing optimized network layers.

### 2. Validation & Parameter Calibration
Validation was conducted on 200 labeled images (190 True Positive instances, 10 True Negative background samples) to determine optimal detection thresholds for **Confidence** and **Intersection over Union (IoU)**.

**Selected Thresholds:**
* **Confidence Threshold:** `0.25`
* **IoU Threshold:** `0.50`

| Validation Metric | Value |
| :--- | :--- |
| **Execution Time** | 306.04 seconds |
| **Accuracy** | 95.5% |
| **Precision** | 99.5% |
| **Recall** | 95.8% |
| **True Negative Rate (TNR)** | 90.0% |
| **False Positive Rate (FPR)** | 10.0% |
| **F1-Score** | 0.976 |

---

## 📈 Final Test Results

Evaluated on the held-out test set under calibrated parameters (Confidence `0.25`, IoU `0.50`):

### Confusion Matrix

| | Predicted: Yes | Predicted: No | Total |
| :--- | :---: | :---: | :---: |
| **Actual: Yes** | **183** *(TP)* | **7** *(FN)* | 190 |
| **Actual: No** | **0** *(FP)* | **10** *(TN)* | 10 |

### Performance Metrics Summary

| Metric | Test Result |
| :--- | :--- |
| **Processing Speed** | 270.01s (~1.35s / image) |
| **Accuracy** | **96.5%** |
| **Precision** | **100.0%** (0 False Positives) |
| **Recall** | **96.3%** |
| **True Negative Rate** | **100.0%** |
| **False Positive Rate** | **0.0%** |
| **F1-Score** | **0.981** |

---

## 💡 Data Engineering Application

Integrating computer vision models like YOLOv3 into spatial ETL pipelines enables automatic spatial feature extraction from aerial imagery. By transforming unorganised raster tiles into structured spatial vector datasets (e.g., GeoJSON / PostGIS geometries), emergency management teams can rapidly query and prioritise disaster response zones.
