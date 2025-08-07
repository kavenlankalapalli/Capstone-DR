from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os


# -------------------------------
# Model Loading
# -------------------------------

#model_path = "/Users/kalyanlankalapalli/downloads/dr_model-2.keras/dr_model-2.keras"
#print(" File exists:", os.path.exists(model_path))


def load_model():
    model_path = '/Users/kalyanlankalapalli/downloads/dr_model-3.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Model load failed:")
        print(e)
        print("model error message abive")
    #model = tf.keras.models.load_model(model_path)
    # model = None  # Placeholder
    

model = load_model()
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def predict_diabetic_retinopathy(processed):
    if model is None:
        print(" Model is not loaded.")
    else:
        print("Model is loaded successfully.")
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[predicted_class] * 100
        st.success(f"Prediction: **{class_labels[predicted_class]}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    return class_labels[predicted_class], confidence
# -------------------------------
# Preprocessing Functions
# -------------------------------
def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB to make sure we always have 3 channels
    image = image.convert("RGB")
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to NumPy array
    image_np = np.array(image)
    
    # CLAHE enhancement (optional)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3 channels
    processed = cv2.merge([enhanced, enhanced, enhanced])
    processed = processed / 255.0  # Normalize
    
    return processed

def preprocess_image1(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")  # Force RGB mode
    image = image.resize(target_size)
    image_np = np.array(image) / 255.0
    return image_np

image_np = preprocess_image1("/Users/kalyanlankalapalli/documents/gcu/milestone-3/aptos2019-blindness-detection/images/validation/3/3b018e8b7303.png")
print(image_np.shape)
input_batch = np.expand_dims(image_np, axis=0)
prediction = model.predict(input_batch)


#image = Image.open("/Users/kalyanlankalapalli/documents/gcu/milestone-3/aptos2019-blindness-detection/images/validation/3/3b018e8b7303.png")
#processed = preprocess_image(image)  
#print("Processed shape:", processed.shape)
#input_batch = np.expand_dims(processed, axis=0)
#print("Input batch shape:", input_batch.shape)  # Should be (1, 224, 224, 3)

#pred_class, confidence = predict_diabetic_retinopathy(input_batch)



#print(pred_class, confidence)
