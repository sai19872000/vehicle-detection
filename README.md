# Vehicle Detection 

## Prject Goals

The primary goals of this project are to:
1) Extract HOG features from test images for input into a classifier.

2) develop a classifier to detect presence of vehicle

3) Use a sliding window technique to scan through the image for the purpose of detecting a car

3) Combine the sliding window and classifier for decting and locating vehicles

4) Implement this in a video steam and use the previous frames reduce false positives


[//]: # (Image References)

[im01]: figures/fig1.png "*"
[im02]: figures/fig2.png "*"
[im03]: figures/fig3.png "*"
[im04]: figures/fig4.png "*"
[im05]: figures/fig5.png "*"
[im06]: figures/fig6.png "*"
[im07]: figures/fig7.png "*"
[im08]: figures/fig8.png "*"
[im09]: figures/fig9.png "*"
[im10]: figures/fig10.png "*"
[im11]: figures/fig11.png "*"
[im12]: figures/fig12.png "*"
[im13]: figures/fig13.png "*"
[im14]: figures/fig14.png "*"




## Data exploration

cell-4 in jupyter notebook: Firstly, I used the data provided far, left, middle_close, right point of view of cars additionally
I have also used the images from KITTI. Here are the sample images randomly chooses from these sources

![alt text][im01]

I used the get hog features cell-5 function using skimage library
```python 
from skimage.feature import hog
```
after extracting the hog features the image looks as follows

![alt text][im02]

I also visualized the non car images that are used to train the classifier; after extracting the hog features we have
the following

![alt text][im03]

### Exploring the color space and other parameters

I then started exploring the color space and other parameters such as: number of orientatons, pixels per cell, number fo cells perblock, hog channels and used a SVM classifier to make predictions. The case where there is resonable test accuracy is the classifier that I have used for making vehicle detection in video. 

Here are some sample resutls cells(9 - 28):


| Case | Colorspace | Number of orientations | Pixels Per Cell | Cells Per Block | HOG Channels | SVM Classifier Test Accuracy (%) | 
| :-------: | :--------: | :----------: | :-------------: | :-------------: | :---------: | :----------: |
| 1                   | RGB        | 9            | 8               | 2               | ALL| 97.18 |
| 2                   | HSV        | 9            | 8               | 2               | ALL  | 98.11 |
| 3                   | LUV        | 9            | 8               | 2               | ALL  | 97.55 |
| 4                   | HLS        | 9            | 8               | 2               | ALL  | 98.76 |
| 5                   | YCrCb        | 9            | 8               | 2               | ALL  | 98.25 |
| 6                   | YUV        | 9            | 8               | 2               | ALL  | 98.54 |
| 7                   | YUV        | 10            | 8               | 2               | ALL  | 98.48 |
| 8                   | YUV      | 11            | 8               | 2               | ALL  | 98.37 |
| 9                  | YUV      | 12            | 10               | 2               | ALL  | 98.23 |
| 10                  | YUV      | 11            | 16               | 2               | ALL  | 98.03 |


I have then used the 

```Python 
find_cars
```
function on cell 30 and the function "draw_boxes" function to draw boxes on the cars in the image. The figure after applying these function is as below

![alt text][im04]

## Sliding Window implementation

I have then used sliding window implementation used during the course. First I have used windows with several sizes different window sizes as shown below

![alt text][im05]
![alt text][im06]
![alt text][im07]
![alt text][im08]

after combining we have (cell 56):

![alt text][im09]

Then I apply the "add_heat", apply_threshold, and "draw_labeled_bboxes" adapted from course to make heat maps of the vehicles.

![alt text][im10]

Applying threshold we have

![alt text][im11]

Changing the cmap to "gray"

![alt text][im12]

Finally I then make the box that fits the vehicle

![alt text][im13]

I then combined all the functions and a pipeline and then tested it on all the test images:

![alt text][im14]

I used this pipeline and applied to both the test video and the project video and successfully identified the vehicles
the link for the project video is ![here][https://github.com/sai19872000/vehicle-detection/blob/master/project_video_out.mp4]
