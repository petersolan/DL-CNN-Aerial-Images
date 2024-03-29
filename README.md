# Welcome to the DL CNN repository.
This repo explains study completed for the purpose of my Master's dissertation topic:  
Detection of fallen trees on aerial images for urban areas affected by natural disasters.

Briefly speaking, I explain how to go from this:

![Image Alt text](https://github.com/petersolan/DL-CNN-Aerial-Images/assets/59766852/701994db-dc31-4687-9388-cc0c1b443d80)

to this:

![Image Alt text](https://github.com/petersolan/DL-CNN-Aerial-Images/assets/59766852/fac1fca2-6094-466f-88dc-4ba0f90cd4f6)

I explain every step of this CNN application in this Deep Learning model, starting from data acquisition, through data preparation, training, validation and finally testing. I am going to explain how this DL model could be applied in Data Engineering process of data pipeline improvement.

1. Methodology  

The main object of this research is to detect correctly as many fallen trees as possible using an automated method. CNN is applied to this problem to study the efficiency of an automated solution related to quality and time required to commit such a kind of classification. I used YOLO v3 for model training. Model feasibility was evaluated with bounding boxes prediction parameters, confusion matrix, calculated metrics, and precision-recall gain curve.

2. Data source  

All the aerial images used for model training are freely available in the following location: https://www.fisheries.noaa.gov/inport/item/65224
I split data as follows: 60/20/20 (t/v/t)

3. Implementation:  

a) training 

For the model training YOLOv3 convolutional neural network I used Darknet/Google Colab.
For the training stages there are a few parameters to tune such as:
batch size (number of images in one batch),
number of epochs (number of times all of the training data has gone through the network)
and total numbers of images used in training.
The batch parameter indicates the batch size used during training, for this study I applied 64. The whole training process was accomplished through external processing via Google Colab, which allows for 12h sessions using GPU. The training was completed in less than 10 hours for a set of 600 images.

![Image Alt text](https://github.com/petersolan/DL-CNN-Aerial-Images/assets/59766852/52f592a5-ba78-4c3a-aef0-306078159d4a)

Overall loss during the training phase, cropped to 500 iterations

The final output of this step is a file in dedicated “weights” format containing the results of the training stage, which are previously mentioned weights.

b) validation  

Weights obtained in the previous step were applied during the validation process. 200 labels used for validation stage. This dataset  consists of 190 True Positive labels and 10 empty images (True Negative).
The main task was to evaluate and statistically prove the most suitable threshold values of Confidence and Intersection over Union (IoU) for model.

After validating results for different parameters author of this research decided to apply thresholds of 0.25 for confidence and 0.5 for IoU to test in the last step.

| Time[s] | 306.04 |
|--------|--------|
| Accuracy | 0.955 |
| Precision | 0.995 |
| Recall | 0.958 |
| True Negative Rate | 0.9 |
| False Positive Rate | 0.1 |
| F1-score | 0.976 |

c) test  

Confusion matrix

| Class |   Y   |   N   |
|-------|-------|-------|
|   Y   |  183  |   7   |
|   N   |   0   |   10  |

Summary of test results
| Threshold Confidence / IoU | 0.25/0.5 |
| - | - |
| Time[s] | 270.01 |
| Accuracy | 0.965 |
| True Positive | 183 |
| True Negative | 10 |
| False Positive | 0 |
| False Negative | 7 |
| Precision | 1 |
| Recall | 0.963 |
| True Negative Rate | 1 |
| False Positive Rate | 0 |
| F1-score | 0.981 |
