# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/center_2018_04_12_18_31_31_600.jpg "Center Lane Driving"
[image2]: ./writeup/center_2018_04_12_18_49_43_075.jpg "Left Lane Driving"
[image3]: ./writeup/center_2018_04_12_19_03_37_564.jpg "Right Lane Driving"
[image4]: ./writeup/center_2018_05_06_14_43_05_441.jpg "Recovery Image 1"
[image5]: ./writeup/center_2018_05_06_14_43_06_903.jpg "Recovery Image 2"
[image6]: ./writeup/center_2018_05_06_14_43_07_391.jpg "Recovery Image 3"
[image7]: ./writeup/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [nvidia self-driving car model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) with some modifications. Because the Keras version used for this project does not support strides with convolutional layers, I used maxpooling after each convolution to reduce dimensionality. I also introduce Local Response Normalisation after the first layer.(model.py lines 30-75) 

The model includes RELU layers to introduce nonlinearity after each layer except the last one, and the data is normalized in the model using a Keras lambda layer (code line 33). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 59, 63, 66). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving at the edge of the track using a correction factor and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a known model architecture and finetune it.

I used the [nvidia model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). Because the Keras version used for this project does not support strides with concolutional layers I used maxpooling layers to reduce the dimensionality. After the last convolutional layer I use averagepooling because this seems to learn faster. 

To combat the overfitting, I added dropout in front of the first 3 fully connected layers at the end of the model.

I also introduced local response normalisation after the first layer because i had good experience with that in other projects.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I generated additional training data recovering from these points.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 30-75) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Operation (stride)| Shape    |
|-------|-------------------|----------|
|Input  |Normalisation      |160x320x2 |
|1      |Convolution 5x5    |160x320x24|
|2      |MaxPooling 3x3(2x2)|80x160x24 |
|3      |LRN                |80x160x24 |
|4      |Convolution 5x5    |80x160x36 |
|5      |MaxPooling 3x3(2x2)|40x80x36  |
|6      |Convolution 5x5    |40x80x48  |
|7      |MaxPooling 3x3(2x2)|20x40x48  |
|8      |Convolution 3x3    |20x40x64  |
|9      |MaxPooling 3x3(2x2)|10x20x64  |
|10     |Convolution 3x3    |10x20x128 |
|11     |AvgPooling10x2(1x2)|1x10x128  |
|12     |Fully Connected    |100       |
|13     |Fully Connected    |50        |
|14     |Fully Connected    |25        |
|Output |Fully Connected    |1         |

Each convolutional and fully connected layer uses Relu as activation function. In front of layers 12, 13 and 14 I use small dropout to prevent overfitting.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap in each direction on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded one lap in each direction driving on the right edge of the track and one lap in each direction driving on the left edge of the track. I used these images with a correction factor of 8.75° in direction to the center. 

![alt text][image2]
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it gets to close to one edge. These images show what a recovery looks like starting from the right edge of the track:

![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data set, I also flipped images thinking that this would balance the amount of data steering left and right For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image7]

After the collection process, I had 15408 number of data points. I did no further preprocessing with this data.


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation set loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
