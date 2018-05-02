import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, merge

def createModel():
  inputs = Input(shape = (160, 320, 3))

  pre = Lambda(lambda x: x / 255 - 0.5)(inputs)

  #shape = (160, 320, 3)
  layer1 = Convolution2D(64, 7, 7, border_mode = 'valid', activation = 'relu')(pre)
  layer1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'valid')(layer1)

  #shape = (76, 156, 64)
  layer2 = Convolution2D(128, 3, 3, border_mode = 'valid', activation = 'relu')(layer1)
  layer3 = Convolution2D(128, 3, 3, border_mode = 'valid', activation = 'relu')(layer2)
  layer3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'valid')(layer3)
  
  #shape = (35, 75, 128)
  layer4 = Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu')(layer3)
  layer5 = Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu')(layer4)
  layer5 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'valid')(layer5)
  
  #shape = (15, 35, 128)
  layer6 = Convolution2D(512, 3, 3, border_mode = 'valid', activation = 'relu')(layer5)
  layer7 = Convolution2D(512, 3, 3, border_mode = 'valid', activation = 'relu')(layer6)
  layer7 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'valid')(layer7)
  
  #shape = (5, 15, 512)
  layer8 = Convolution2D(512, 1, 1, border_mode = 'valid', activation = 'relu')(layer7)
  layer8 = AveragePooling2D(pool_size = (5, 5), strides = (1, 5), border_mode = 'valid')(layer8)
  
  #shape = (1, 3, 512)
  flat = Flatten()(layer8)
  drop = Dropout(0.5)(flat)

  layer9 = Dense(512, activation = 'relu')(drop)
  drop2 = Dropout(0.4)(layer9)
  
  logits = Dense(1)(drop2)

  model = Model(input = inputs, output = logits)
  model.compile(loss = 'mse', optimizer = 'adam')

  return model
