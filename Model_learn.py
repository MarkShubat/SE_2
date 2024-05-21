import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import pandas as pd
# from google.colab import drive
import glob
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


for dirname, _, filenames in os.walk('/content'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


classes = ['Amarant', 'Cabbage', 'Watercress']
directory_to_extract_to = '/content/drive/My Drive/extracted_images'
image_df = pd.DataFrame(columns=['X', 'y'])
images_set = []

for e in classes:
    dir = directory_to_extract_to + f'/{e}'
    # Получите список всех файлов изображений в директории
    images_path = glob.glob(dir + '/*.jpg') # Или другой формат изображения
    # Считайте изображения
    images = [Image.open(img_path) for img_path in images_path]

    images_set.append(images)

  for dirname, _, filenames in os.walk('/content'):
      for filename in filenames:
          print(os.path.join(dirname, filename))
          # Проходим по всем файлам в архиве


def crop_img(image_CV2, size_x, size_y):
    image = image_CV2

    # Получение размеров изображения
    height, width = image.shape[:2]

    # Определение размера центральной части
    center_x, center_y = width // 2, height // 2

    crop_size_x = (size_x // 224 - 1) * 224
    crop_size_y = (size_y // 224 - 2) * 224

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
            # region.save(f'chunk_{x}_{y}.png')
            tmp_set.append(region)

chunk_size = 224

images_set_2 = []
for img_set in images_set:
  tmp_set = []
  for img in img_set:
    tmp = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, chans = tmp.shape
    image = crop_img(tmp, height, width)
    tmp_set.append(image)
  images_set_2.append(tmp_set)
images_set = []

images_set_3 = []
for img_set in images_set_2:
  tmp_set = []
  for img in img_set:
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pillow = Image.fromarray(tmp)
    slice_image(image_pillow, 224, tmp_set)
  print(len(tmp_set))
  while(len(tmp_set) < 3270):
    tmp_set = tmp_set + tmp_set
  images_set_3.append(tmp_set[0: 512])
images_set_2 = []

for akd in images_set_3:
  print(len(akd))

images_set_4 = []
for img_set in images_set_3:
  tmp_set = []
  for img in img_set:
    tmp = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    tmp_set.append(tmp)
  images_set_4.append(tmp_set)
images_set_3 = []

list_x = []
list_y = []
for i in range(3):
   list_x = list_x + images_set_4[i]
   list_y = list_y + [i for j in range(512)]
print(list_y)

final_df = pd.DataFrame({'X': list_x, 'y': list_y})

imgs_dir = '/content/images'
os.makedirs(imgs_dir, exist_ok=True)
for dir in classes:
  os.makedirs(f'{imgs_dir}/{dir}', exist_ok=True)

for i in range(3):
  for j in range(len(images_set_4[i])):
    cv2.imwrite(f'/content/images/{classes[i]}/{j}.jpg', images_set_4[i][j])

# Каталог с набором данных
data_dir = '/content/images'

early_stop = EarlyStopping(
    monitor='val_loss',  # Метрика для отслеживания
    patience=5,  # Количество эпох для ожидания улучшения
    verbose=1,  # Вывод информации в консоль
    mode='min',  # Режим (min для минимизации метрики)
    restore_best_weights=True
)

classes = [ 'Amarant', 'Cabbage', 'Watercress']

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 )

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Режим классификации
    classes = classes,
    subset='training')

test_data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Размер изображений
    batch_size=32,  # Размер пакета
    class_mode='categorical',  # Режим классификации
    classes = classes,
    subset='validation'  # Определение тестовой выборки
)

class_indices = train_generator.class_indices
print(class_indices)

class_indices = test_data_generator.class_indices
print(class_indices)

file_paths = train_generator.filepaths
for path in file_paths:
    if '.ipynb_checkpoints' in path:
       print(path)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax')) #sigmoid
model.summary()
model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_data_generator,
    callbacks=[early_stop]
    )

model.evaluate(test_data_generator, verbose = 1)

plt.plot(history.history['accuracy'],
         label='Доля правильных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля правильных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля правильных ответов')
plt.legend()
plt.show()

model.save('proj.h5')
