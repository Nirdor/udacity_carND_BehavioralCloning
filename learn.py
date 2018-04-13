import csv
import os
import numpy as np
import cv2

from keras.models import load_model

import model as mod

print('Import Data...')
lines = []
with open('traindata/0/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0))

with open('traindata/1/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, 0.4))

with open('traindata/-1/driving_log.csv') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append((line, -0.4))

images = []
measurements = []
for line, correction in lines:
  #image_name = line[0].split('\\')[-1]
  #image = cv2.imread('traindata/IMG/' + image_name)
  image_center = cv2.imread(line[0])
  image_left = cv2.imread(line[1])
  image_right = cv2.imread(line[2])
  images.append(image_center)
  images.append(image_left)
  images.append(image_right)
  measurement_center = float(line[3]) + correction
  measurement_left = float(line[3]) + correction + 0.05
  measurement_right = float(line[3]) + correction - 0.05
  measurements.append(measurement_center)
  measurements.append(measurement_left)
  measurements.append(measurement_right)
    
X_train = np.array(images)
y_train = np.array(measurements)

print('Data ready. Create Model...')

model_name = 'model.h5'

if os.path.exists(model_name):
  model = load_model(model_name, custom_objects = {'LRN':mod.LRN})
else:
  model = mod.createModel()


print('Model created. Start Learning...')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 1)
print('Learning done!. Saving model...')
model.save(model_name)
print('Finish!')
