# NT2211 - Máy học trong An toàn thông tin

## Assignment: Build Model from Dataset 
* Group 16
* Project struct:
```
├── dataset/
│   ├── train.csv
│   └── test.csv
├── Src/
│   ├── Group16.ipynb # Process build model (Pre-processing, Feature Selection, Tuning and Training)
│   ├── xgb_model.pkl # Model XGBoost
│   ├── feature_importance.png
|   └── . . .
├── requirements.txt
├── main.py # Evaluate from training model
└── . . .
```
*Note:* Model XGBoost is stored in [Google Drive](https://drive.google.com/file/d/1PNFglQ9GuYo95m_adW4uIFDV_Q505qKC/edit) (auto download from `main.py`) because its size is 104MB (exceed 100MB on Github).

## How to use
```
$ python3 main.py [-h] --dataset DATASET

Evaluate a XGBoost model on a dataset.

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Path to the dataset CSV file.

Ex: python3 main.py --dataset dataset\val.csv
```
![XGBoost_ex](assets\ex.png)


## References
*  yliang725: Anomaly Detection IoT23: [Github](https://github.com/yliang725/Anomaly-Detection-IoT23)