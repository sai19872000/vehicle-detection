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
