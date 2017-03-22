#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[img_ex]: ./img/signs_ex.png "Examples from training set"
[img_freq]: ./img/signs_freq.png "Frequency of signs in each class"
[img_bilateral]: ./img/signs_bilateral.png "Bilateral filtered images"
[img_grey]: ./img/signs_grey2.png "Grey scaled images"
[image_internet]: ./img/internet_signs.png "Signs from internet"
[image_internet_ans]: ./img/internet_signs_ans.png "Signs from internet ans"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pyau/CarND-Project2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples are in each category.

![alt text][img_freq]

This is a random sampling of some of the training data.

![alt text][img_ex]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I apply a bilateral filter to image. I find bilateral filter being better at preserving edges and keep the image sharp.

![alt text][img_bilateral]

Then I greyscaled the image. I tried running both grayscaled and colored images through my neural network, and grayscale performs slightly better.

![alt text][img_grey]

As a last step, I normalized the image data to avoid very large values or very small values during calculation of lost function.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data was provided as pickle files, and are already separated for training, validation and testing, so I only need to load in the data in the first code cell.

In the fifth code cell, I shuffled the training data, and preprocess all the training, test and validation data.

I did not get a chance to generate additional data for training, but would certainly do so for future data sets that I encounter.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook, called LeNet2.  I had another version of LeNet implemented in the sixth cell, but decide to play with the different layers to get better validation accuracy.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 32x32x16    |
| RELU                  |                                               |
| Max pool 2x2 kernel   | 2x2 stride, valid padding, output 16x16x16    |
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x32    |
| RELU                  |                                               |
| Max pool 2x2 kernel   | 2x2 stride, valid padding, output 8x8x32      |
| Fully connected		| output 2048           						|
| Fully connected       | output 512                                    |
| Fully connected       | output 128                                    |
| Fully connected       | output 43                                     |
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an adam optimizer, a batch size of 128, train for 20 epochs, with a learning rate of 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.941 
* test set accuracy of 0.925

I first tried to implement LeNet for MNIST and it achieve an accuracy of about 89% for this data set. Based on LeNet, I believe that parameters for each layer should be increased, since we are classifying for 43 classes, so I also slightly expanded the number of depths for each convolution layer.

Then I inserted an additional convolution layer, in hope of the layer capturing more high level details for each class. I also added in another fully connected layer, just to see if it also improves the accuracy.

I have experimented using dropout but it will decrease the accuracy by a little. Since the data set is quite small, I feel that I do not need the speed gain by applying dropout.

I took out the first maxpool function for layer 1, and observe a significant improvement in accuracy, so I stick with this change.  But it also increases the model training time.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image_internet] 

These images were originally very large. To fit into the neural network, I have to resize them to 32x32 before preprocessing.  That makes some of the signs quite blurry. The code to load in the images are in the ninth cell.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

![alt text][image_internet_ans] 


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing | Wild animals crossing 						| 
| Pedestrain   			| Road narrows on the right						|
| Double Curve			| Double Curve									|
| 20 km/h	      		| 30 km/h					     				|
| Slippery Road			| Dangerous curve to the left					|
| General caution       | General caution                               |
| Bicycles crossing     | Bicycles crossing                             |
| Traffic signals       | Traffic signals                               |


The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. This compares unfavorably to the accuracy on the test set of 92.2%.  I believe data augmentation would be the key to train for a better accuracy percentage.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is wild animals crossing and is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 31 Wild animals crossing  					| 
| .00     				| 2 50km/h 										|
| .00					| 29 Bicycle crossing							|
| .00	      			| 10 No passing for vehicles over 3.5 metric tons|
| .00				    | 23 Slippery Road      						|


For the second image, the model predicts road narrows on the right, while the correct answer is pedestrians (second highest probability).

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .88                   | 24 Road narrows on the right                  | 
| .08                   | 27 Pedestrians                                |
| .03                   | 28 Children crossing                          |
| .00                   | 11 Right-of-way at the next intersection      |
| .00                   | 30 Beware of ice/snow                         |

For the third image, the model predicts double curve correctly. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .53                   | 21 Double curve                               | 
| .47                   | 25 Road work                                  |
| .00                   | 31 Wild animals crossing                      |
| .00                   | 30 Beware of ice/snow                         |
| .00                   | 23 Slippery Road                              |

For the fourth image, the model predicts the signs as 30km/h, while the correct result is 20km/h.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .78                   | 1 30km/h                                      | 
| .17                   | 18 General caution                            |
| .02                   | 0 20km/h                                      |
| .00                   | 4 70km/h                                      |
| .00                   | 39 Keep right                                 |

For the fifth image the model predicts dangerous curve to the left with not much confidence, while the correct answer is slippery road, which has a close probability to the answer.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .33                   | 19 Dangerous curve to the left                | 
| .30                   | 30 Beware of ice/snow                         |
| .27                   | 23 Slippery                                   |
| .06                   | 24 Road narrows on the right                  |
| .01                   | 28 Children crossing                          |

For the sixth image, the model predicts general caution correctly. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | 18 General caution                            | 
| .00                   | 27 Pedestrians                                |
| .00                   | 26 Traffic signals                            |
| .00                   | 39 Keep left                                  |
| .00                   | 37 Go straight or left                        |

For the seventh image, the model predicts bicycles crossing correctly. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .77                   | 29 Bicycles crossing                          | 
| .23                   | 25 Road work                                  |
| .00                   | 19 Dangerous curve to the left                |
| .00                   | 22 Bumpy Road                                 |
| .00                   | 24 Road narrows on the right                  |

For the eighth image, the model predicts traffic signal correctly.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .99                   | 26 Traffic signal                             | 
| .00                   | 18 General caution                            |
| .00                   | 22 Bumpy road                                 |
| .00                   | 24 Road narrows on the right                  |
| .00                   | 39 Keep left                                  |
