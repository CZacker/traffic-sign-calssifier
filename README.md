# **Traffic Sign Recognition** 

## Writeup
---
### This project is about [Udacity Self-Driving Car Nanodegree](https://cn.udacity.com/course/self-driving-car-engineer--nd013), the main task is to **Build a Traffic Sign Recognition Project**  
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./ReadmeImg/data-bar-chart.png "Visualization"
[image2]: ./ReadmeImg/percent.png "percent"
[image3]: ./ReadmeImg/1.png "Traffic Sign 0"
[image4]: ./ReadmeImg/2.png "Traffic Sign 1"
[image5]: ./ReadmeImg/3.png "Traffic Sign 2"
[image6]: ./ReadmeImg/4.png "Traffic Sign 3"
[image7]: ./ReadmeImg/5.png "Traffic Sign 4"
[image8]: ./ReadmeImg/incepetion.png "Googlelenet Incepetion"
[image9]: ./ReadmeImg/dropout.png "Dropout"
[image10]: ./ReadmeImg/top_5.png "top_5"
[image11]: ./ReadmeImg/prediction.png "prediction"

---
### README  
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.  

You're reading it! and here is a link to my [project code](https://github.com/CZacker/trafic-sign-calssifier-LeNet/blob/master/Traffic_Sign_Classifier_1.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data look like.

![alt text][image1]

Min number of images per class = 180  
Max number of images per class = 2010  

![percent][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Processed image part:  
* Soften images by `cv2.GaussianBlur()` function to add Gaussian Blur filter.
* Cutting images with `crop(img,mar=3)`, increase valid range of values for images as possible `img.shape (32 , 32 , 3) to (26 , 26 , 3)`.
* Change images contrast by `contr_img()`, processed into clear imagess.

Augment image part:
* Change images contrast by `contr_img()`, processed into clear imagess.
* Rotate images a random degree by `rotate_img()`.
* Magnify randomly in 1.0-1.4 time by `interpolation = cv2.INTER_CUBIC`.
* Rezise to (26, 26, 3), then augment images by function `cv2.equalizeHist()`.

Here is an example of an original image and an augmented image:

![data preprocess][images3]

Final image size is (26, 26, 3).

The difference between the original data set and the augmented data set is the following:  
a.Effective area increase.
b.Feature is more obvious.
c.Generating random more data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 26x26x3 RGB image                             | 
| Convolution 1x1       | 1x1 stride, SAME padding, outputs 26x26x3.    |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, SAME padding, outputs 26x26x64.   |
| RELU                  |                                               |
| GoogLeNet Incepetion  | 1x1 stride,  outputs 26x26x256.               |
| Max pooling           | 2x2 stride, VALID padding, outputs 13x13x256. |
| GoogLeNet Incepetion  | 1x1 stride,  outputs 13x13x512.               |
| Max pooling           | 3x2 stride, VALID padding, outputs 6x6x512.   |
| Convolution 1x1       | 1x1 stride, SAME padding, outputs 6x6x256     |
| Flatten               | 6x6x256                                       |
| Fully connected       | Input = 6x6x256. Output = 512.                |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | Input = 512. Output = 43.                     |

GoogLeNet Incepetion V2 Module
![incepetion][image8]. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Type of optimizer = AdamOptimizer
The batch size = 128
Number of epochs = 15
Learning rate = 0.001 (gradually reduce 0.8 times every 5 epochs) 
Keep probability = 0。5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.979
* validation set accuracy of 0.979885
* test set accuracy of 0.940

* What was the first architecture that was tried and why was it chosen?
    - The first architecture is LeNet, because it's the easiest way to get a model. 
* What were some problems with the initial architecture?
    - Increase neraul network depth and width is the only way to improve performance.
    - Plenty of parameters cause ooverfitting.
    - Depends on larger dataset and excellent hardware.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    - [GoogLeNet Incepetion V2](https://arxiv.org/abs/1512.00567) 
        + Avoid representational bottlenecks, especially early in the network.
        + Higher dimensional representations are easier to process locally within a network.
        + Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
        + Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    - It more difficult to make changes to the network. If the architecture is scaled up naively, large parts of the computational gains can be immediately lost.
    - full connect layer -> dropout layer(drop 50% )
    - ![alt text][image9]
    
A well known architecture was chosen:
* What architecture was chosen?
    - [GoogLeNet Incepetion V2](https://arxiv.org/abs/1512.00567)
* Why did you believe it would be relevant to the traffic sign application?
    - [GoogLeNet Incepetion V2](https://arxiv.org/abs/1512.00567)
        + Avoid representational bottlenecks, especially early in the network.
        + Higher dimensional representations are easier to process locally within a network.
        + Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
        + Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - well done
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image3]

more [test img](https://github.com/CZacker/trafic-sign-calssifier-LeNet/blob/master/ReadmeImg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Road work             | Road work                                     | 
| U-turn                | U-turn                                        |
| Yield                 | Yield                                         |
| 30 km/h               | 30 km/h                                       |
| etc.                  | etc.                                         |

![prediction][image11]
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

For example(the first image), the model is relatively sure that this is a yield sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Yield                                         | 
| 3.02092736e-38        | Speed limit(30km/h)                           |
| 0.00000000e+00        | Speed limit(20km/h)                           |
| 0.00000000e+00        | Speed limit(50km/h)                           |
| 0.00000000e+00        | Speed limit(60km/h)                           |

The following is the new images:
![alt text][image10]


