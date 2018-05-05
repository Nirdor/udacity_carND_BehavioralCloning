import csv
import os
import numpy as np
import cv2

from keras.models import load_model

import model_nvd as mod

print('Import Data...')
lines = []
with open('traindata/0/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0))

with open('traindata/1/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0.3))

with open('traindata/-1/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, -0.3))
    
with open('traindata/recover/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0))
    
with open('traindata/additional/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0))

images = []
measurements = []
for line, correction in lines:
  image_name_center = '/'.join(line[0].split('\\')[-4:])
  #image_name_left = '/'.join(line[1].split('\\')[-4:])
  #image_name_right = '/'.join(line[2].split('\\')[-4:])
  image_center = cv2.imread(image_name_center)
  image_center_flipped = np.fliplr(image_center)
  #image_left = cv2.imread(image_name_left)
  #image_left_flipped = np.fliplr(image_left)
  #image_right = cv2.imread(image_name_right)
  #image_right_flipped = np.fliplr(image_right)
  images.append(image_center)
  #images.append(image_left)
  #images.append(image_right)
  images.append(image_center_flipped)
  #images.append(image_left_flipped)
  #images.append(image_right_flipped)
  measurement_center = float(line[3]) + correction
  #measurement_left = float(line[3]) + correction + 0.1
  #measurement_right = float(line[3]) + correction - 0.1
  measurement_center_flipped = -measurement_center
  #measurement_left_flipped = -measurement_left
  #measurement_right_flipped = -measurement_right
  measurements.append(measurement_center)
  #measurements.append(measurement_left)
  #measurements.append(measurement_right)
  measurements.append(measurement_center_flipped)
  #measurements.append(measurement_left_flipped)
  #measurements.append(measurement_right_flipped)
    
X_train = np.array(images)
y_train = np.array(measurements)

print('Data ready. Create Model...')

model_name = 'model_nvd02.h5'

if os.path.exists(model_name):
  model = load_model(model_name, custom_objects = {'LRN':mod.LRN})
else:
  model = mod.createModel()


print('Model created. Start Learning...')
model.fit(X_train, y_train, validation_split = 0.1, shuffle = True, nb_epoch = 6)
print('Learning done!. Saving model...')
model.save(model_name)
print('Finish!')
