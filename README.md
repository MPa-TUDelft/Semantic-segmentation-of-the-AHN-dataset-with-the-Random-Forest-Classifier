# Semantic segmentation of the AHN dataset with the Random Forest Classifier

This gihub repository contains the code to train and test Random Forest models for point cloud classification as mention in the thesis of Manos Papageorgiou.
The thesis is submitted to the Delft Univerity of Technology in partial fulfillment of the requirements for the degree of Master of Science in Geomatics. Defended on September 14, 2021.

Supervisors:
1. Ravi Peters
2. Weixiao Gao

<img src="src/images/cover_front.JPG">


## How to run

Download the datasets from the link provided in the data folder.

Install dependencies with 

```bash
pip install -r requirements.txt
```

Run train_RFC_AHN3.py to train a Random Forest.

Run test_RFC_AHN3.py to test the Random Forest.
