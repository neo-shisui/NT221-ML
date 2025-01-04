import os
import gzip
import gdown
from xgboost import XGBClassifier
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Specify the output file name
model_saved_path = os.path.abspath('Src/xgb_model.pkl')
model_compressed = os.path.abspath('Src/xgb_model.gz')

def load_model():
    """Load the XGBoost model."""
    try:
        if os.path.exists(model_saved_path) is False:
            # Check and extract zip file
            if os.path.exists(model_compressed) is False:
                # https://drive.google.com/file/d/1PNFglQ9GuYo95m_adW4uIFDV_Q505qKC/view?usp=sharing
                MODEL_ID = "1PNFglQ9GuYo95m_adW4uIFDV_Q505qKC"
                DRIVE_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

                # Download the file
                gdown.download(DRIVE_URL, model_compressed, quiet=False)

            # Decompress the model
            with gzip.open(model_compressed, 'rb') as f_in:
                with open(model_saved_path, 'wb') as f_out:
                    f_out.write(f_in.read())

        # Load the XGBoost model
        xgb_model_loaded = XGBClassifier()
        xgb_model_loaded.load_model(model_saved_path)
        print("[*] Load XGBoost model successfully!")
        return xgb_model_loaded
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def data_preprocessing(data):
    # 20 features + Label
    features = ['IAT', 'AVG', 'Header_Length', 'Magnitue', 'Protocol type', 'rst_count', 'flow_duration', 'Tot size', 'Max', 'syn_flag_number', 'urg_count', 'Tot sum', 'Min', 'Srate', 'Rate', 'syn_count', 'fin_flag_number', 'UDP', 'fin_count', 'Covariance', 'Label']
    result = data[features]

    # Fill NA value by mode
    select_features = ['IAT', 'AVG', 'Header_Length', 'Magnitue', 'Protocol type', 'rst_count', 'flow_duration', 'Tot size', 'Max', 'syn_flag_number', 'urg_count', 'Tot sum', 'Min', 'Srate', 'Rate', 'syn_count', 'fin_flag_number', 'UDP', 'fin_count', 'Covariance']
    
    # Use .loc to ensure you are modifying the correct slice of the DataFrame
    result.loc[:, select_features] = result[select_features].apply(
        lambda col: col.fillna(col.mode()[0])
    )
    return result

def load_dataset(dataset_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(dataset_path)
        result = data_preprocessing(data)
        print("[*] Load dataset successfully!")
        return result
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

def evaluate_model(model, data):
    """Evaluate the model on the dataset."""
    select_features = ['IAT', 'AVG', 'Header_Length', 'Magnitue', 'Protocol type', 'rst_count', 'flow_duration', 'Tot size', 'Max', 'syn_flag_number', 'urg_count', 'Tot sum', 'Min', 'Srate', 'Rate', 'syn_count', 'fin_flag_number', 'UDP', 'fin_count', 'Covariance']
    
    # Step 1: Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['Label'])

    # Step 2: Split dataset while preserving class ratios
    X = data[select_features]  # Features
    y = y_encoded  # Encoded target

    # Predict
    y_pred_proba = model.predict_proba(X)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Print evaluation metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    print("[*] Evaluation metrics:")
    print("  [+] Accuracy:", accuracy_score(y, y_pred))
    print("  [+] Classification Report:")
    print(classification_report(y, y_pred))

    # y_pred_proba = model.pre(dtest)
    # y_pred = np.argmax(y_pred_proba, axis=1)

    # # Print evaluation metrics
    # print("[*] Evaluation metrics:")
    # print("  [+] Accuracy:", accuracy_score(y, y_pred))
    # print("  [+] Classification Report:")
    # print(classification_report(y, y_pred))

def main():
    """Main function to load model, dataset, and evaluate."""
    parser = argparse.ArgumentParser(description="Evaluate a XGBoost model on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV file.")
    
    args = parser.parse_args()

    # Load the model and dataset
    model = load_model()
    dataset = load_dataset(args.dataset)

    # Evaluate the model
    evaluate_model(model, dataset)

if __name__ == "__main__":
    main()
