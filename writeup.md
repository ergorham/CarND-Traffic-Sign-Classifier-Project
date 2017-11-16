#**Traffic Sign Recognition** 

##Project Writeup

###Summary: Using LeNet classifier as a starting point and adding additional features to the convolutional layers yielded 92% accuracy on the test set.

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

[image1]: ./examples/hist_all_images.png "Histogram of training set pixel values - non-normalized"
[image2]: ./examples/hist_mu_128_sigma_128.png "Histogram of training set pixel values - normalized"
[image3]: ./examples/hist_y_train_classes.png "Historgram of training set class assignments"
[image4]: ./dataset/animals_32.png "Traffic Sign 1"
[image5]: ./dataset/pedestrian_32.png "Traffic Sign 2"
[image6]: ./dataset/slippery_32.png "Traffic Sign 3"
[image7]: ./dataset/speed_32.png "Traffic Sign 4"
[image8]: ./dataset/traffic_32.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This writeup serves to document the methodology and process for creating a deep neural network to identify various traffic signs based on the German Traffic Data set. The link to my [project code](https://github.com/ergorham/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram of the classes labeled within the training set.

![alt text][image3]

Based on the fact there are 34799 images in the train set, there are some traffic signs that are over represented (Speed limit 30 & 50, Keep Left) with more than twice the mean in number of examples. This may lead to a preference, or a bias, within the trained system.

![alt text][image1]

Another feature of the data set is the distribution of intensity of pixel values, as seen in the histogram above. Using the non-normalized values, the plot shows that images tend toward the low intensity and suggest low contrast images. This may be more reflective of real-world scenarios, however it may make classification more challenging. 

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Based on the distribution above, I had attempted to use histogram normalization tools offered by different libraries, however I ran into challenges formatting the output from said tools to conform to the data format expected of the LeNet architecture.

In order to achieve results, I used a simplistic normalization method (subtracting the mean pixel intensity value and dividing by that mean value to achieve unity distribution). This method allowed me to focus on the model rather than data formatting.

![alt text][image2]

The histogram above shows the distribution of normalized pixel value, now a float between -1 and 1, with mean 0. This is desirable for the optimizer when minimizing loss.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Flatten		      	| 5x5x32,  outputs 300 element vector 			|
| Fully connected		| input 800, outputs 120						|
| RELU					|												|
| Fully connected		| input 120, outputs 84 						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, outputs 43 classes					|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 25 epochs and a batch size of 128, with a learning rate of 0.0007. I had chosen these parameters to achieve convergence without overfitting the model, or leading to divergence. I had experimented with different learning rates (between 0.0001 and 0.003). Although I could have used a decaying learning rate, I had achieved good results with my empirically determined values.

The batch size was increased from the default LeNet based on the fact I was using a GPU to train and had more memory available. Although I had more memory at my disposal, tests with higher batch sizes yielded in hunting around a local optimum (at batch 256) and divergence (at batch 512). As a point of further study, I would investigate why this is the case.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.942 
* test set accuracy of 0.919

Testing with the initial LeNet model provided in the exercises proved to be a good starting point, with an accuracy of 0.86. Given that the types of features we are trying to detect were more complex than the handwritten digits, with quadruple the possible classes as the basic LeNet, I opted to increase the number of features in the convolutional layers in order to give the model more flexibility to learn distinguishing marks within the signs. The additional features helped reduce underfitting, however I wanted to avoid possibly overfitting the model by adding too many features.

I extended the standard LeNet to use all three colour channels. An alternative would be to grayscale the images, however in reading the literature, depending on the grayscale algorithm chosen, some data is lost in the process. Retrospectively, working in grayscale would have afforded other benefits such as simply histogram equalization, which could have benefited the classifier's performance.

I opted to leave the filtering and the stride untouched as there was no obvious motivation to adjust these. In hindsight, I could have added a convolutional layer with a large window to look for traffic signs (28x28 px) within the images and start filtering from there. This would be a future improvement.

Based on the lessons and with the additional features, I decided that some redundancy was needed in the model, so I added a dropout layer just before the final connected layer, training the model with a dropout of 0.5.

Although the model achieved the required benchmark, the performance on the test set suggests that there is still some overfitting. This could be remedied by amplifying the training set. One idea that I attempted was putting the traffic signs from the training set on different backgrounds from other images within the training set. However, it was time consuming to do it manually and found a possible algorithm to do this late in the project.



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

4 out of the 5 images are triangular in shape, which may make them difficult to distinguish from one another. Also, because these images are taken from a website as opposed to photos of actual street signs, the colours are saturated. Initially, the classifier performed poorly (0.00 accuracy) with the signs on a white background. I opted to paste the signs onto one of the images from the test set, scaling the signs to 28x28 px, similar to the signs within the test set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild Animal Crossing	| Speed Limit (80 km/h)							| 
| Pedestrian Crossing	| Speed Limit (80 km/h)							|
| Slippery Road			| Speed Limit (80 km/h)							|
| Speed Limit (60 km/h)	| Speed Limit (80 km/h)			 				|
| Traffic Sign			| Traffic Sign	     							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This differs to the accuracy on the test set of 91.9%. It is possible that the scaling of my traffic signs is off, that they are larger than those within the test set and therefore the features would not match anything that have been trained.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image, the model is unsure of its conclusion and could almost easily classify it between the top four choices. Two of the possible signs are triangular signs, suggesting the model had difficulty finding the outline and is inline with my suspicion that the scale of the new images is off as compared to those of the test set. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .13         			| Speed Limit (80 km/h)   						| 
| .10     				| Priority Road									|
| .08					| Traffic Signals								|
| .07	      			| No passing veh > 3.5 tons		 				|
| .05				    | Right-of-way at next intersection				|


For the second image the results were even more depressing. The signs selected bore little resemblance to one another and the only one it was truly convicted of (compared to the rest) did not have the same overall shape as the test image.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .13         			| Speed Limit (80 km/h)   						| 
| .08     				| Priority Road									|
| .06					| Dangerous curve to the right					|
| .05	      			| Speed Limit (60 km/h)			 				|
| .05				    | Right-of-way at next intersection				|

The third image...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .11         			| Speed Limit (80 km/h)   						| 
| .08     				| No passing veh > 3.5 tons						|
| .07					| Speed Limit (60 km/h)							|
| .07	      			| Dangerous curve to the right	 				|
| .06				    | Priority Road									|

The fourth image...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .15         			| Speed Limit (80 km/h)   						| 
| .11     				| Speed Limit (120 km/h)						|
| .08					| Speed Limit (60 km/h)							|
| .07	      			| Speed Limit (50 km/h)	 						|
| .03				    | Wild animals crossing							|

The fifth image data shows that although the classifier was determined to call this an 80 km/h speed limit as well, the correct answer eked out its favourite sign by a mere 7 percentage points. Some of the other usual suspects are there in mix, with similar probabilities, indicating a clear preference for those signs over others.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .19         			| Traffic Signals		   						| 
| .12     				| Speed Limit (80 km/h)							|
| .09					| Right-of-way at next intersection				|
| .07	      			| Speed Limit (100 km/h) 						|
| .06				    | Wild animals crossing							|


