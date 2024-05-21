from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


class Item(BaseModel):
    text: str

def cv2_to_pil(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    image_pillow = Image.fromarray(img_cv2)
    return image_pillow

def pil_to_cv2(img_pil):
    tmp = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return tmp

def crop_img( ):
    chunk_size = 224
    img = Image.open('document_2.jpg')
    tmp = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, chans = tmp.shape
    image = tmp
    

    # Получение размеров изображения
    height, width = image.shape[:2]

    # Определение размера центральной части
    center_x, center_y = width // 2, height // 2

    crop_size_x = (height // 224 - 1) * 224
    crop_size_y = (width // 224 - 2) * 224

    # Определение координат для обрезки
    x1, y1 = center_x - crop_size_x // 1, center_y - crop_size_y // 1
    x2, y2 = center_x + crop_size_x // 1, center_y + crop_size_y // 1

    # Вырезание центральной части изображения
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def slice_image(image_PIL, chunk_size, tmp_set):
    image = image_PIL
    width, height = image.size

    for x in range(0, width, chunk_size):
        for y in range(0, height, chunk_size):
            box = (x, y, x + chunk_size, y + chunk_size)
            region = image.crop(box)
            #region.save(f'chunk_{x}_{y}.png')
            tmp_set.append(region)
    return tmp


chunk_size = 224

tmp = []

def read_and_preprocess_image(image):
    img = image
    #img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)


def load_model():
    return tf.keras.models.load_model("proj.h5")

    prediction = model.predict(tensor)

    
model = load_model()
image = read_and_preprocess_image(slice_image(cv2_to_pil(crop_img()),224, tmp)[0])
classes = ['Amarant', 'Cabbage', 'Watercress']
   
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    prediction = model.predict(image)
    return classes[np.argmax(prediction)]