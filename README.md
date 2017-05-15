#**German Traffic Sign Classifier using Convolutional Deep Neural Network ** 

##Pablo Vela

###This was the second project that I did for my Udacity Self-Driving Car Nanodegree. 

---

**Building a Traffic Sign Recognition Network**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/BarChart.PNG "Bar Chart"
[image2]: ./Images/Top5.PNG "Top five"
[image3]: ./Images/Bottom5.PNG "Bottom Five"
[image4]: ./Images/before-preprocessing.PNG "before preprocessing"
[image5]: ./Images/after-preprocessing.PNG "after preprocessing"
[image6]: ./New_Traffic_Signs/3.jpg "new traffic signs"
[image7]: ./New_Traffic_Signs/4.jpg "before preprocessing"
[image8]: ./New_Traffic_Signs/14.jpg "before preprocessing"
[image9]: ./New_Traffic_Signs/23.jpg "before preprocessing"
[image10]: ./New_Traffic_Signs/27.jpg "before preprocessing"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pablovela5620/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Below are the top/bottom five most numerous types of images ...

![alt text][image2]
![alt text][image3]

As well as a bar graph with the number of occurances against the class id of the images

![alt text][image1]
###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Two steps were taken during image preprocessing, first the histogram equalization was applied to the images 
to remove the effect of brightness on the images, and secondly the images were normalized by diving the images by 255 and 
subtracting 0.5

Below you can see examples of images before preprocessing

![alt text][image4]

and after the images have gone through preprocessing

![alt text][image5]
 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation Function							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU          		| Activation Function        					|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Input 5x5x16, flattens to 400					|
| Fully Connected		| Input 400 outputs 120							|
| RELU					| Activation Function							|
| Drop out				| 50 percent keep probability					|
| Fully Connected		| Input 120 outputs 84							|
| RELU					| Activation Function							|
| Drop out				| 50 percent keep probability					|
| Fully connected		| Input 84 outputs 43							|
| Cross entropy/softmax	|												|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a learning rate of 0.001. 
A total of 20 epochs were chosen as a higher number resulted in no significant improvements

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.947 
* test set accuracy of 0.926

If a well known architecture was chosen:
* What architecture was chosen?

    The architecture of the Le Net model was chosen
* Why did you believe it would be relevant to the traffic sign application? 

    The Le Net model was previously used with
    great success on the MNIST data set for hand written number classification, this led me to believe that it would also preform
    very well on classifying other images such as german traffic signs
* How does the final model's accuracy on the validation and test set provide evidence that the model is working well?

    The model had a 95.3% validation set and 93.6% test set accuracy I believe this provides solid evidence that the model performs well
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

Some difficulties that may arise from trying to classify these images is the fact that some do not have an aspect ratio of 1:1. This will cause issues
when the images are resized to 32,32 because it will cause some distrortions to the images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Slippery Road  		| Slippery Road									|
| Pedestrians			| General caution								|
| 60 km/h	      		| 60 km/h					 			    	|
| 70 km/h			    | 70 km/h      					       	    	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 80%. 
Compared to the accuracy results of the test set, the are much relatively accurate.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the new images, the model top softmax probability can be seen in the table below

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Stop sign   									| 
| .99     				| Slipper Road									|
| .73					| General caution								|
| .49	      			| 60 km/h					 				    |
| .45				    | 70 km/h    							        |


A bar graph for each individual image and their softmax probabilities can be found in code cell 24 in the Traffic Sign Classifer IPython Notebook
