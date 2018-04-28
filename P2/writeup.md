
# **Traffic Sign Recognition** 

## Writeup


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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data spread out.

![Category%20distribution.png](attachment:Category%20distribution.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I think in the traffic sign classify project the color of image didn't influence much. on the other hand, the gray image can make the compute more simple. 
Here is an example of a traffic sign image before and after grayscaling.
![30km.png](attachment:30km.png)
![30km:h.png](attachment:30km:h.png)

Then,I normalized the image data because normalization ensured that every feature of samples will be treated equally in the supervised learning.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x128    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128 				    |
| Flatten		        | outputs 512       							|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| Dropout			    | keep_prop=0.5									|
| Fully connected		| outputs 43        							|
|      				    |         									    |
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer of adam, and I set the hyperparameters as following:batch=128, epochs=50,learning rate = 0.0005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.951 
* test set accuracy of 0.936

I chosed the LeNet model as my base model. the model was first bring out by Yann Lecun in 1998.It has a good performace on digital recognition.
At first, I only changed the output of the model from 10 to 43. the model get a validation accuracy of 0.932 after the 43th epoch. It seem that the model converged too slow and a liitle overfitting, becouse when the training accuracy get 1,the validation accuracy is only 0.932 and didn't seem to increase.
Then I add one more convolutin layer to the model and adjust the output of each layer. I changed the activation of the second full-connect layer to dropout with a keep-propablity of 0.5.
At last, the final model got a training accuracy of 0.998, validation_accuracy of 0.951,test_accuracy of 0.936 within 23 epochs.The result prove out that the model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![4.jpg](attachment:4.jpg) ![5.jpg](attachment:5.jpg) ![6.jpg](attachment:6.jpg) ![3.jpg](attachment:3.jpg)  ![1.jpg](attachment:1.jpg)

The first image might be difficult to classify because there are servral diffrent speed limit sign in the data, and they are almost the same except the first number of speed.

The second image have the same reason as the first image, and the Speed limit (20km/h) sign have less training samples in the training data.

The last three image might be difficult to classify because of their complicated background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(30 km/h)  | Speed limit (60km/h)   						| 
| Speed limit(20 km/h)  | Children crossing 							|
| Priority road			| Priority road									|
| Turn left ahead		| Turn left ahead					 			|
| Yield   			    | Yield      			  				        |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 0.932.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Speed limit (60km/h) sign (probability of 0.998), but the image is a Speed limit (30km/h) sign(probability of 3.09270630e-11). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Speed limit (60km/h)   						| 
| .00189     			| Speed limit (50km/h 							|
| 3.09270630e-11		| Speed limit (30km/h)							|
| 6.38600804e-12	    | Yield					 				        |
| 1.71672204e-17		| End of no passing      						|


For the second image the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 0.574), but the image is a Speed limit (20km/h) sign (probability of 1.61837161e-05). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .574         			| Speed limit (60km/h)   						| 
| .425     			    | Children crossing 							|
| 7.37960916e-04		| No entry							            |
| 2.25005700e-04	    | No passing				 				    |
| 1.61837161e-05		| Speed limit (20km/h)      					|

For the third image the model is very sure that this is a Priority road sign (probability of 0.999), and it's turn out that the image is exactly a Priority road sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Priority road   						        | 
| 3.04646824e-06     	| End of no passing 							|
| 1.22076528e-06		| End of no passing by vehicles over 3.5 metric	|
| 4.03572017e-07	    | Speed limit (60km/h)				 			|
| 1.84953048e-08		| No passing      					            |

For the fourth image the model is quite sure that this is a Turn left ahead sign (probability of 1.00000000e+00 ), and it's turn out that the image is exactly a Turn left ahead sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Turn left ahead   						    | 
| 6.43410969e-09     	| Slippery road 							    |
| 3.26226535e-09		| Ahead only							        |
| 1.74176645e-10	    | End of no passing				 				|
| 3.55553345e-15		| Beware of ice/snow      					    |

For the fifth image the model is quite sure that this is a Yield sign (probability of 1.00000000e+00 ), and it's turn out that the image is exactly a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Yield   						                | 
| 7.85403201e-16     	| End of no passing by vehicles over 3.5 metric |
| 4.15174853e-17		| Priority road							        |
| 4.86654632e-18	    | End of no passing				 				|
| 6.70884548e-23		| No entry      					            |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




