import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from google.colab import drive

def read_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def main():
    drive.mount('/content/drive')
    model_path = "/content/drive/MyDrive/Models/proj.h5"
    model = tf.keras.models.load_model(model_path)
    image_path = 'document.jpg'
    tensor = read_and_preprocess_image(image_path)
    prediction = model.predict(tensor)
    print("Prediction:", prediction)
    
if __name__ == "main":
    main()
