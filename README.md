# AI Based Diabetic Retinopathy Detection System
Capstone - AI based Diabetic Retinopathy Detction system
This repository contains my AI-powered capstone project designed to **detect and classify the severity of diabetic retinopathy** using retina images. The goal is to build an end-to-end detection system that can assist healthcare providers—especially in rural or under-resourced areas—by providing accurate, explainable predictions from fundus images.

##  Project Highlights

- Built using **EfficientNetB0** with transfer learning
- Trained on the **APTOS 2019 Blindness Detection Dataset**
- Includes data augmentation to improve generalization
- Evaluates performance using classification reports, confusion matrix, and accuracy plots
- Modularized Python code for clarity and reusability
- Ready for deployment via a lightweight Streamlit web app

##  Folder & File Structure

| File/Folder       | Description                                      |
|-------------------|--------------------------------------------------|
| `main.py`         | Orchestrates the full pipeline (train & evaluate) |
| `model.py`        | Contains the EfficientNetB0-based model builder |
| `data_loader.py`  | Loads and augments training/validation images    |
| `train.py`        | Training loop with support for multiple epochs   |
| `evaluate.py`     | Accuracy plots, confusion matrix, classification report |
| `app/`            | Streamlit-based web interface (optional UI)     |
| `data/`           | Directory for training and validation image sets |
| `outputs/`        | Saved models, metrics, and predictions           |

##  Dataset

This project uses the **APTOS 2019 Blindness Detection Dataset**, which includes thousands of labeled retina images.  
Dataset: [APTOS 2019 on Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

classDiagram
  class ImageSample {
    +filepath: str
    +raw_image
    +preprocessed
    +load()
    +preprocess()
  }

  class PredictionResult {
    +label: str
    +confidence: float
    +created_at: datetime
    +to_json()
    +to_csv()
  }

  class DataLoader {
    +build_train_loader()
    +build_val_loader()
    +augment()
    +normalize()
  }

  class EfficientNetModel {
    +build_model(input_shape)
    +load_weights(path)
    +predict(tensor) PredictionResult
  }

  class Trainer {
    +fit(train_data, val_data)
    +save_checkpoints()
    +callbacks
  }

  class Evaluator {
    +evaluate(test_data)
    +confusion_matrix()
    +classification_report()
    +plot_metrics()
  }

  class AppUI {
    +upload_images()
    +run_inference()
    +show_explanations()
    +export_report()
  }

  class MainOrchestrator {
    +run_train()
    +run_eval()
  }

  %% Relationships
  DataLoader --> ImageSample : yields
  ImageSample --> EfficientNetModel : preprocessed to tensor
  EfficientNetModel --> PredictionResult : returns
  Trainer --> EfficientNetModel : trains
  Trainer --> DataLoader : consumes
  Evaluator --> EfficientNetModel : loads/checkpoints
  Evaluator --> DataLoader : consumes test set
  AppUI --> EfficientNetModel : inference
  AppUI --> PredictionResult : displays/exports
  MainOrchestrator --> Trainer
  MainOrchestrator --> Evaluator


Note

This repository is temporarily public for capstone evaluation and review purposes. It will be made private after completion of the grading process.
