# Behavioral Cloning

---

Behavioral Cloning Project
---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Files Submitted & Code Quality
---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- [<b>model.py</b> - The script used to create and train the model](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/model.py)
- [<b>drive.py</b> - The script to drive the car](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/drive.py)
- [<b>model.h5</b> - The saved model](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/model.h5)
- [<b>writeup_report.md</b> - A summary of the project](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/writeup_report.md)
- [<b>videoTrack1.mp4</b> - A video recording of driving autonomously around track1](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/videoTrack1.mp4)
<center>
![track1](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack1-r10.gif?raw=true)
</center>

- [<b>videoTrack2.mp4</b> - A video recording of driving autonomously around track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/videoTrack2.mp4)
<center>
![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack2-r10.gif?raw=true)
</center>
  
---

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



Model Architecture and Training Strategy
---

#### 1. An appropriate model architecture has been employed

My model (model.py lines 321-359) is almost identical to the [Nvidia convolution neural network for self driving cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The only changes I made are:

- The input layer is for grayscale images (1 channel instead of 3).
- After the input layer, the images are cropped.
- After the cropping layer, the data is normalized.
- In between the last convolutional layer and flatten layer, I inserted of a dropout layer.

Details on why I made these changes are given below.


#### 2. How I got it to work - grayscale, clahe, dropout, lots-of-data

I had a lot of spectacular virtual crashes as I developed this model. I will explain how I got it to work by discussing this stubborn failure:

<center>
<u>Crash on a difficult section of Track2:</u> 
(https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/videoTrack2.mp4)
![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack2-crash.gif?raw=true)
</center>

This section is particularly difficult, because:

- it contains strong bright section (sun) next to a very dark section (shadow)
- the change occurs in the middle of a sharp turn

The issue with the contrast is addressed by applying grayscale and clahe transforms  during image generation (model.py - generator - line 198), followed by a cropping layer in the keras model (model.py - line 336):

<center>
![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/track2-crash-image-processing.gif?raw=true)
</center>

Up to this point in my model development, I had created training data sets for track 1 and track 2, driving the car in the middle of the road, in both directions. This was a lot of data already, but still the car crashed. My observation was that the "driver" lost track of the right line and then the middle line and did not know what to do. Reviewing the training data, I realized that this type of situation was not yet covered and the model was not trained how to respond.

I created 4 additional training data sets. For track 1 & track 2, I drove while 'hugging the line'. I did this for the right line and the left line.
Here I am showing the training data created for track2, hugging the left line:

<center>
![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/track2-hug-left-line.gif?raw=true)

<b>Training data, using 'line hugging' drive style</b>
</center>

I re-trained the model, and tried it out on both tracks. It perfectly handled track 1, but track 2 still crashed in the same location. I scratched my head a few times, and then few times more, trying to figure out how to fix this. I was sure I had sufficient data, convergence was good (see below), the model was a proven CNN, so what could it be ? I then decided to use a similar trick as what was done for the left and right images. 

For the images (center, left and right) created during 'line hugging' drive style, I applied a small correction of 0.2 to the steering angle to nudge the car to move towards the center of the road. (model.py - HUG_CORRECTION).

I again re-trained the model, and now it successfully navigated both tracks, as shown in the videos above.


 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 352). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

Model Architecture and Training Strategy
---

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
