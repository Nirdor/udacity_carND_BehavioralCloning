import tensorflow as tf

from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense

class LRN(Layer):
  def __init__(self, depth_radius = 2, bias = 1.0, alpha = 2e-05, beta = 0.75, **kwargs):
    self.depth_radius = depth_radius
    self.bias = bias
    self.alpha = alpha
    self.beta = beta
    super(LRN, self).__init__(**kwargs)

  def build(self, input_shape):
    super(LRN, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x, mask=None):
    return tf.nn.local_response_normalization(x, depth_radius = self.depth_radius, bias = self.bias,
                                              alpha = self.alpha, beta = self.beta)

  def get_output_shape_for(self, input_shape):
    return input_shape


def createModel():
  inputs = Input(shape = (160, 320, 3))

  pre = Lambda(lambda x: x / 255 - 0.5)(inputs)

  #shape = (160, 320, 3)
  
  layer1 = Convolution2D(24, 5, 5, border_mode = 'same', activation = 'relu')(pre)
  layer1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer1)
  layer1 = LRN()(layer1)

  #shape = (80, 160, 24)
  layer2 = Convolution2D(36, 5, 5, border_mode = 'same', activation = 'relu')(layer1)
  layer2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer2)

  #shape = (40, 80, 48)
  layer3 = Convolution2D(48, 5, 5, border_mode = 'same', activation = 'relu')(layer2)
  layer3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer3)

  #shape = (20, 40, 64)
  layer4 = Convolution2D(64, 3, 3, border_mode = 'same', activation = 'relu')(layer3)
  layer4 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer4)

  #shape = (10, 20, 64)
  layer5 = Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu')(layer4)
  layer5 = AveragePooling2D(pool_size = (10, 2), strides = (1, 2), border_mode = 'valid')(layer5)

  #shape = (1, 20, 64)
  flat = Flatten()(layer5)
  drop = Dropout(0.3)(flat)

  #shape = (1280)
  layer6 = Dense(100, activation = 'relu')(drop)
  drop2 = Dropout(0.3)(layer6)

  layer7 = Dense(50, activation = 'relu')(drop2)
  drop3 = Dropout(0.2)(layer7)

  layer8 = Dense(25, activation = 'relu')(drop3)
  
  logits = Dense(1)(layer8)

  model = Model(input = inputs, output = logits)
  model.compile(loss = 'mse', optimizer = 'adam')

  return model
