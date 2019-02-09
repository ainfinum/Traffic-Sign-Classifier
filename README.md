# **Traffic Sign Recognition Project**

## Project

In this project, I'll build convolutional neural network to classify traffic signs. 
The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) will be used to train a model and test the accuracy.
I'll test a model on traffic sign images found on the web. In the last part of this project, I visualize several convolution layers.


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



[//]: # (Image References)

[image1]: ./examples/img_ex3.png "Random images from the training set "
[image2]: ./examples/German-Traffic-Sign-Dataset.png "German Traffic Sign Dataset"
[image3]: ./examples/Updated-German-Traffic-Sign-Dataset.png "Updated-German-Traffic-Sign-Dataset"
[image4]: ./examples/transformation.jpg "Image transformation"
[image5]: ./examples/gray-image.png "Gray image"
[image6]: ./examples/normalized.png "Normalized image"
[image7]: ./examples/9-web-images.png "Traffic Signs from the web"
[image8]: ./examples/9-predictions-visual.png "Visualizing predictions"
[image9]: ./examples/conv1.png "Convolution layer 1"
[image10]: ./examples/conv2.png "Convolution layer 2"
[image11]: ./examples/conv3.png "Convolution layer 3"
 
 
## Data Set Summary & Exploration

### 1. A basic summary of the data set. 
After loading the data I got the summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

### 2. Visualization of the dataset.

This grid of images is representing random choosen images from the training set.


![alt text][image1]

The bar below is showing the data distribution in the training set.

![alt text][image2]

## Design and Test a Model Architecture

### 1. Preprocessing the image data

As clearly visible from the data distribution bar chart, there is a huge variability of the distribution between class instances within the dataset, some classes have less than 250 images and some have 2000 images. 
I decided to generate additional images using CV library image transformation methods to equalize them.
After applying rotation (random angle from -15 to +15 degrees) and image shifting (+-3 pixels) I got the following data set:

* Number of training examples = 60503
* Average images per class = 1407
* Classes: 43


![alt text][image3]

Examples of generated images:

![alt text][image4]

I preprocess images for the neural network in 3 steps:
1. Grayscale image. Colors are not important for traffic signs classification task so removing color channels will reduce the data we need to process.

Grayscale image

![alt text][image5]


2. Zero centering - means that I process my data so that the mean (average) of the data lies on the zero


3. Normalization - the process of scaling individual samples to have their values in the range (-1, 1)

Normalized image

![alt text][image6]


### 2. Model architecture.

1. I've tried several architectures. With the classic LeNet5 architecture I was able to get validation accuracy = 0.954 and test accuracy = 0.921
2. I tried to add dropout layers after convolution or fully connected layers. I got the maximum validation accuracy = 0.957 and test accuracy = 0.935
3. After several experiments I found that my model consisted of 3 convolutions and 3 fully connected layers give the best performance: validation accuracy = 0.985 and test accuracy = 0.974


### 3. Model training. 

I've trained models with different hyperparameters in order to find the model with the best accuracy.
The first architecture I tried was LeNet5 and I got the following validation and test set accuracy:

Best accuracy from LeNet5:

| EPOCHS | BATCH_SIZE | Learning rate | Validation Accuracy | Test Accuracy  |
|--------|:----------:|--------------:|--------------------:|---------------:|
| 50     | 128        | 0.05          | 0.950               | 0.921          |

I was able to get for this model is 0.921 trying different epochs, batch size and learning rate the best test accuracy 


Best accuracy from LeNet5 with dropouts:

| EPOCHS | BATCH_SIZE | Learning rate | Dropout rate | Validation Accuracy | Test Accuracy  |
|--------|:----------:|--------------:|-------------:|--------------------:|---------------:|
| 70     | 128        | 0.005         | 0.9          |    0.954            | 0.935          |

This model is slightly better but I continued to experiment with model architecture and finally got better result.

### Final model architecture
#### Hyperparameters
* LEARNING RATE = 0.001
* EPOCHS = 10
* BATCH SIZE = 64
* Dropout keep probability rate : 0.5

#### Optimizer
Adam optimizer


The final model architecture with 6 layers ( 3 convolutions and 3 fully connected layers) achieved test set accuracy = 0.974. I found that the test accuracy can be different after every new training but it stays between 0.965 and 0.975
With the dropout rate 0.6 or 0.8 the results were not so good, so I think the dropout layers generalize model and help to avoid overfitting.


Final model architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x24 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				    |
| Flatten	            | 5x5x48 outputs into one dimension             |
| Fully connected		| Inputs 1200, outputs 256	                	|
| Fully connected		| Inputs 256,  outputs 128		                |
| Fully connected		| Inputs 128,  outputs 43, logits   			|
|						|												|
|						|												|
 

I was surprised that the model was able to get the test accuracy = 0.974 just in 10 epochs and it takes less than 2 minutes for training it in GPU mode.
I think adding 1 or 2 convolution layers (to make the model similar to VGG16) can produce an even greater result. 
 
Final model results:

| EPOCHS | BATCH_SIZE | Learning rate | Dropout rate | Training Set Accuracy | Validation Accuracy | Test Accuracy  |
|--------|:----------:|--------------:|-------------:|----------------------:|--------------------:|---------------:|
| 10     | 64         | 0.001         | 0.5          | 0.99                  |  0.985              | 0.974          | 
| 10     | 64         | 0.001         | 0.5          | 0.99                  |  0.988              | 0.970          | 


## Test a Model on New Images

### 1. German traffic signs found on the web.

In order to test my model on new images I found 9 German traffic signs on the web:

![alt text][image7]

I used 7 good quality images and 2 images with a different perspective. I think one of the "Stop" and "Road work" signs might be difficult to classify correctly. 
 
 
### 2. The model's predictions on new traffic signs

As I expected the model classified correctly 7 good quality images. The "Stop" sign with a transformed perspective was also classified correctly but I thought that the model will fail on this image.
The model was not able to classify "Road work" image with a changed perspective.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road   		    					| 
| Stop             		| Stop   		              					| 
| No entry    		    | No entry	    	        					|
| General caution  		| General caution   							| 
| Speed limit (50km/h)  | Speed limit (50km/h)      	 				|
| Road work	            | Priority road    						        |
| Children crossing    	| Children crossing             				|
| Stop            		| Stop     					 				    |
| Speed limit (60km/h)	| Speed limit (60km/h)     		 				|


The model correctly predicted 7 of the 8 traffic signs which gives an accuracy of 88%. It less than the accuracy on the test set of 97% but my new test set is very small to compare these results.
I think the model will be able to classify the last image correctly if I will generate images with perspective distortion and add them into the training set.


### 3. The softmax probabilities for each prediction



Image 1: The model is 100% confident that this is a "Priority road" sign (probability of 1.0)

Image 2: The model is 100% confident that this is a "Stop" sign (probability of 1.0)

Image 3: "No entry" sign was classified correctly

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| No entry				                		| 
| .000000173			| Go straight or right                 			|


Image 4: The model classified correctly the "General caution" sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999        			| General caution				           		| 
| .0002578      		| Traffic signals                   			|
 

Image 5: The model is 100% confident that this is a "Speed limit (50km/h)" sign (probability of 1.0)

Image 6: The model completely failed to predict "Road work" sign and the correct prediction is not present in the top 5 softmax probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9646        			| Priority road         						| 
| .0177   				| Beware of ice/snow                    		|
| .0106					| Right-of-way at the next intersection 		|
| .0022	      			| Bicycles crossing    							|
| .0018				    | Double curve      						    |

 
Image 7: The model is 100% confident that this is a "Children crossing" sign (probability of 1.0)

Image 8: The model classified correctly the "Stop" sign despite the fact that image is poor quality with a transformed perspective.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .6618        			| Stop         					            	| 
| .1163   				| Roundabout mandatory                   		|
| .0977					| Yield 		                                |
| .0365	      			| Keep right   							        |
| .0255				    | Speed limit (50km/h)     					    |


Image 9: The model classified correctly the "Speed limit (60km/h)" sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999        			| Speed limit (60km/h)			           		| 
| .00037          		| Speed limit (80km/h)                 			|


Here is a visualization of predictions:
![alt text][image8]


 
## Visualizing the Neural Network

It's quite interesting to visualize convolution layers to see what characteristics the neural network use to make classifications.

The feature map activations of the first convolution layer clearly show "Road work" sign. And the shape of the "Road work" sign is still visible in the second convolution layer.
The feature maps activations of the third convolution layer show that this layer searches some specific features of the image. 

Convolution layer 1

![alt text][image9]


Convolution layer 2

![alt text][image10]


Convolution layer 3

![alt text][image11]

 