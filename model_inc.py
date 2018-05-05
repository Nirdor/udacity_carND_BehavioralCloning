import tensorflow as tf

from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, merge

def inception(x, out_depths, reduce_depths):

  x1 = Convolution2D(out_depths[0], 1, 1, border_mode = 'same', activation = 'relu')(x)
  
  x3 = Convolution2D(reduce_depths[0], 1, 1, border_mode = 'same', activation = 'relu')(x)
  x3 = Convolution2D(out_depths[1], 3, 3, border_mode = 'same', activation = 'relu')(x3)
  
  x5 = Convolution2D(reduce_depths[1], 1, 1, border_mode = 'same', activation = 'relu')(x)
  x5 = Convolution2D(out_depths[2], 5, 5, border_mode = 'same', activation = 'relu')(x5)
  
  p = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), border_mode = 'same')(x)
  p = Convolution2D(out_depths[3], 1, 1, border_mode = 'same', activation = 'relu')(p)
  
  return merge([x1, x3, x5, p], mode = 'concat', concat_axis = 3)

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
  layer1 = Convolution2D(64, 7, 7, border_mode = 'same', activation = 'relu')(pre)
  layer1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer1)
  layer1 = LRN()(layer1)

  #shape = (80, 160, 64)
  layer2 = Convolution2D(64, 1, 1, border_mode = 'same', activation = 'relu')(layer1)
  layer2 = Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu')(layer2)
  layer2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer2)
  layer2 = LRN()(layer2)

  #shape = (40, 80, 128)
  layer3 = inception(layer2, [64, 128, 32, 32], [96, 16])
  layer3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer3)

  #shape = (20, 40, 256)
  layer4 = inception(layer3, [128, 192, 96, 64], [128, 32])
  layer4 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer4)

  #shape = (10, 20, 480)
  layer5 = inception(layer4, [192, 208, 48, 64], [96, 16])

  #shape = (10, 20, 512)
  layer6 = inception(layer5, [128, 256, 64, 64], [128, 24])

  #shape = (10, 20, 512)
  layer7 = inception(layer6, [256, 320, 128, 128], [160, 32])
  layer7 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), border_mode = 'same')(layer7)

  #shape = (5, 10, 832)
  layer8 = inception(layer7, [384, 384, 128, 128], [192, 48])
  layer8 = AveragePooling2D(pool_size = (2, 5), strides = (1, 1), border_mode = 'valid')(layer8)

  #shape = (1, 1, 1024)
  flat = Flatten()(layer8)
  drop = Dropout(0.5)(flat)
  
  layer9 = Dense(512, activation = 'relu')(drop)
  drop2 = Dropout(0.5)(layer9)

  layer10 = Dense(256, activation = 'relu')(drop2)
  logits = Dense(1)(layer9)

  model = Model(input = inputs, output = logits)
  model.compile(loss = 'mse', optimizer = 'adam')

  return model
