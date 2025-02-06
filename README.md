# Object Detection Case Study – Underwater Dataset  

This project focuses on object detection for underwater creatures using **Faster R-CNN with a ResNet50 backbone**.  
The goal is to train a model capable of detecting various marine species in underwater environments.  

For a detailed report on the methodology, experiments, and results, check the **full report** [here](https://github.com/shani1610/underwater-object-detection/tree/main/report).  

<img src="./report/assets/data_sample1.png" alt="Underwater Dataset Sample" width="400"/>

---

## Installation

Clone the repository:  

```bash
git clone https://github.com/shani1610/underwater-object-detection.git
cd underwater-object-detection
```

Install the required dependencies:
```bash
pip install requirements.txt
```

## Dataset
The dataset should be structured as follows:

```
underwater-object-detection/
  ├── data/
  │   ├── test/
  │   │   ├── images/
  │   │   ├── labels/
  │   ├── train/
  │   ├── valid/
  ├── scripts/
```

### Option 1: Download Manually

1. Download the dataset from Kaggle: [Aquarium Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
   
2. Extract the ZIP file and place its contents inside the ```data/``` directory.

### Option 2: Download via Script

If you have Kaggle credentials set up, you can download the dataset directly using:

```bash
python scripts/download_data.py
```

Ensure you have your Kaggle API key.

## Checkpoints

Pre-trained model checkpoints can be downloaded from:
[Dropbox Link](https://www.dropbox.com/scl/fo/56nxq2px1pie7yiuu2cgu/AJrBZE0-ZeFG96e8VuOVAE0?rlkey=j7n706cq2jdpoykhacmyk22do&st=49vxi0ip&dl=0)

Place the downloaded checkpoint inside the checkpoints/ directory.


