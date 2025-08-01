 main.py

from data_loader import create_data_generators
from model import build_model
from train import train_model
from evaluate import plot_accuracy, evaluate_model
import numpy as np
import os

BASE_PATH = "/content/drive/My Drive/capstone-3/milestone-3/aptos2019-blindness-detection"
MODEL_PATH_H5 = os.path.join(BASE_PATH, "dr_model.h5")
MODEL_PATH_KERAS = os.path.join(BASE_PATH, "dr_model.keras")
HISTORY_PATH = os.path.join(BASE_PATH, "training_history.npy")

# Load data
tg, vg = create_data_generators()

# Build model
model = build_model()

# Train model
history = train_model(model, tg, vg, epochs=10)

# Save model and history
model.save(MODEL_PATH_H5)
model.save(MODEL_PATH_KERAS)
np.save(HISTORY_PATH, history.history)
print("Model and training history saved.")

# Evaluate
plot_accuracy(history.history)
evaluate_model(model, vg)
