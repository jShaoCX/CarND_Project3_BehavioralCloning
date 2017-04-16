#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Line 20 of the model.py file is the data augmentation function used by the generator. Lines 47 to 73 contain the code to split the original data into training and validation. Line 76 was used to experiment with the model before using a data generator. Lines 90 to 107 was code to determine the distribution of the data and how to make the data less unbalanced. Line 96 is the generator that uses the data augmentation function and attempts to balance the data at the same time. Finally, lines 156 to 176 is the actual model. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I followed the nvidia self driving car model explained in the lesson videos with only small modifications (model.py lines 156-176). The model first does some preprocessing by cropping out 60 pixels from the top and 20 pixels from the bottom of the image. Then, I use an average pooling layer to downsample the image followed by the lambda layer to normalize the image before sending in into a convolutional layer. The model contains filters of 5x5 and 3x3 and depts varying from 24 to 64. The model uses ReLU layers to introduce nonlinearity as seen in all the convolutional layers' activation functions and following all of the fully connected layers later in the network. 

####2. Attempts to reduce overfitting in the model

The model contains one dropout layer to reduce overfitting (model.py line 171). I did not add more because I assumed that the amount of data augmentation being used would allow for the model to not encounter many occurances of the exact same image and hence overfit to that particular image. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. Line 55 shows the split between data and line 165 shoes the fit_generator function using the separate validation data to determine a validation loss. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 180).

####4. Appropriate training data

I used the training data provided by the project and it seemed to be enough to complete the first track. I scrolled through the data and I believe within the 8000+ images, it shows about 4 laps going clockwise around the track and 4 laps going counter-clockwise around the dirt track. There are recovery sections in the data set so that the car would know how to recover from steering into a lane line. I did not use the left and right camera data. I will go into detail in the sections to come.

###Model Architecture and Training Strategy

####1. Solution Design Approach

From the progression in the lesson videos (from single layer convolution, to LeNet, to the nvidia architecture), it seems that the larger the network, the better and smoother the performance. I split my image and steering angle data into training and validation sets to check the initial performance of the plain nvidia model with only the preprocessing that was shown in the lesson. The resulting loss graph is show below:

RAW NVIDIA MODEL LOSS GRAPH

Though the performance on the dirt track of the original nvidia model was quite good, it did not perform very well at all on the mountain track. This indicates that it overfit to the dirt track's 8000+ data points. The graph shows that the validation loss was actually lower than the training loss and by the 5th epoch, the training loss and the validation loss converge. This seems to indicate no overfitting to the data because by the 5th epoch, the training loss had flattened out. However, it could still mean that the model is fitting the training data ideally and the training data does not represent enough driving scenarios. The low validation loss could be due to the fact that the validation set has a set of images that closely represent the training set (since most of the original data set looks very similar). 

In order to improve performance on the challenge and training times, I added a downsampling layer where I used average pooling after the cropping layer. My assumption is that the features of interest like the lane lines would not be obscured by one layer of downsampling but any noise within the road like small bits of shadow or the texture of the road would be smoothed out. It also decreased the training time by ~75% and the loss graph is shown below. At this point I removed the drop out layer because I assumed that the data augmentation generator would provide diverse enough data to prevent overfitting:

MODIFIED NVIDIA MODEL LOSS GRAPH

Note that the validation loss is still below the training loss the entire time. This could be explained by the fact that a generator was used on the training set. Most of the samples that go through the generator were augmented images so it is unlikely that given the epoch range of 4 to 6 that the model could overfit this dynamic training set. More details on the data augmentation generator can be found in section 3 following this section. This version of the nvidia model performed just as well on the training track (dirt track) and was able to get through at least a quarter of the challenge track.


####2. Final Model Architecture

The final model architecture (model.py lines 156-176) consisted of a cropping layer that removed the top 60 and bottom 20 rows of pixels. The average pooling layer downsamples the image and the lambda layer normalizes the image to values between -1 and 1. The rest of the model closely follows the nvidia model. The first convolution layer is a 5x5x24 with same padding and a ReLU activation followed by a max pooling. The second convolution layer is a 5x5x36 with same padding and a ReLU activation followed by max pooling. The third convolution is a 5x5x48 with same padding and a ReLU activation followed by max pooling. The next two convolution layers are 3x3x64 with same padding and ReLU activation but no max pooling afterwards. The model then goes on to flatten the data and go through a 100 node fully connected layer with ReLU activation, a 50 node fully connected layer with ReLU activation and a 10 node layer with ReLU activation. The final node outputs a regression so there is no nonlinear activation:  

VISUALIZE ARCHITECTURE

![alt text][image1]

####3. Creation of the Training Set & Training Process

I used the original data provided for the project and found that to be enough. Any new data I tried to append did not improve the performance of the training track (the model performs very well without any additional data). Furthermore, the trials that I added made the driving jitter possibly due to the way I drove manually with the keyboard. The pre-collected data seemed to exhibit very smooth behavior on some of the straight sections:

![alt text][image2]

There are various sections where the training data purposely steers into the lane lines and recovers. Furthermore, the sharp turns section after the bridge is where a portion of the data collection started. I assume this was to gather more data on turning instead of gathering more 0 degree steering angle data which is plentiful. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

I did not end up using the left and right camera because I did not have enough time to determine the correction factor and also extra lines would have had to been added to the data augmentation generator. I believe that the shift augmentation applied to the images (along with the steering correction) would create enough data such that using other cameras was not necessary. It also bloated the amount of images I needed to store and transfer between hard drives but given more time it would be interesting to explore the other cameras as well. 

I also added about half the track of images from the second track (the mountain track). I added it hoping that it would improve the performance of the car on the challenge track and it did slightly though was not able to make it all the way around.

I followed the instructions to decrease the preference for a left turn in the dirt track and flipped all non-zero steering angle images and added that to the data set on the fly:

![alt text][image6]
![alt text][image7]

After the collection process, I had 9214 data points and with flipping all non-zero steering angle images and appending that to the data set, the total points increased to 13658. Unforunately, most of them were 0 steering angle data. I used a counter (model.py line 92) to find how many 0 steering angle data points there were. There were 4770 of the 0 steering angle images. I integrated a skip block for 0 steering angle data points using modulus on the index of the data point and decreased the number of 0 steering angle data points to 1114 and reducing the total data points to 10002. 

HISTOGRAM OF DATA DISTRIBUTION

I shuffled the list of strings for the data points from the csv file before splitting them into a validation and train set. The validation set was 20% of the total set of 9214, which resulted in 1841 validation points. The numbers for the 0 steering angle data points and the distribution of the validation set varies because of the shuffle the flipping of the images and filtering out of 0 steering angle images. But that did not have an adverse affect on the training. 

The augmentation of each data point was limited to flattening out of the brightness, rotation, and shift. I figured flattening out the brightness by setting all of the first channel of every image to 128 (the average brightness of the training set) would allow the model to ignore shadows in the road. Rotation may not be necessary but I added small rotations (-6 to 6 degrees) to vary the types of augmentation available. On a flat road, it doesn't make sense but I felt that it would be relevant to the mountain track because there are points where the road tilts going downhill and uphill. The shift was calculated using the image below and some assumptions I made about the road width:

IMAGE OF CALCULATIONS

After some basic runs of 3 epochs to see if the loss was decreasing, I found that the ideal number of epochs was 5-6 as evidenced by the train and validation loss graphs produced in the previous sections. At around 5-6 epochs, the validation and training losses converged, the training loss stopped decreasing as well by then. I used an adam optimizer so that manually training the learning rate wasn't necessary.
