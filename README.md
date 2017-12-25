**Vehicle Detection Project**

---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/CarVsNonCar.png
[image2]: ./examples/HogFeatures.png
[image3]: ./examples/find_cars.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/heatmap.png
[image6]: ./examples/label.png
[image7]: ./examples/last.png
[video1]: ./project_video_output.mp4 

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
For selection HOG parameters I searched for all possible combination of colorspaces with different channels. I also varied orientation and pixel_per_cell ad checked which combination is giving good accuracy with SVM. Below code give a glimpse of how parameters are searched. 
```
color_space_vector=['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
hog_channel_vector=[0,1,2,3]
orientation_vector=[9,11]
pix_per_cell_vector=[8,16]
for color_space in color_space_vector:
    for hog_channel in hog_channel_vector:
        for orient in orientation_vector:
            for pix_per_cell in pix_per_cell_vector:
```
for parameter selection I put the fallowing parameters fixed
```
cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
```
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I  used hog features and color features for training a linear SVM. The code is in the cell below the title 'Best SVM'  
```
color_space = 'HSV'  orient = 11  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = 3 
spatial_size = (16, 16) 
hist_bins = 16   

```

Feature vector length has a lenght 7284. It took 37.98 Seconds for training with a Test Accuracy of  0.987.
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

for this purpose #find_cars# function is used. We do not need to search the whole image for cars so ystart = 400
ystop = 656 are selected. different values of scale are tried and scale=1.5 showed good results.  The #find_cars# function calculates the hog future for the whole image at once and then depending on the window it extracts the features for that window. We also cocatenated the spatial vector and histogram bin vector to create a test_feature_vector for the window. This test_feature_vector is inputted to the learned SVM for deciding if it contains the car or not.
The below image is resulted after running the #find_function# on test image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier? 

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

