# Stereo-Vision

## Project Description

In this project, I have implemented the concept of Stereo Vision on three data sets, each of them containing 2 images of the same scenario but taken from two different camera angles. By comparing the information about the scene from 2 vantage points, we can obtain the 3D information by examining the relative position of the objects. I have also estimated the camera pose using the concept of the Essential matrix (E), Fundamental matrix (F) and, triangulation. 

## Pipeline
### 1. Calibration 
- Compare the two images and select the set of matching feature. Tune the Lowe's ration to reject the outliers
- Estimate the Fundamental matrix using the obtained matching feature. Use the RANSAC to make your estimation more robust. Enforce the rank 2 condition for the fundamental matrix.
- Estimate the Essential matrix(E) from the Fundamental matrix(F) and instrinsic camera parameter.
- Decompose the E into a translation T and rotation R
- Disambugiate the T and R using triangulation.

### 2. Rectification
- Apply perspective transfomation to make sure that the epipolar lines are horizontal for both the images. This will limit the search space to horizontal line during the corrospondace matching process in the later stage.

### 3. Corrospondence
- For each epipolar line, apply the sliding window with SSD to find the corrospondence and calulate disparity.
- Rescale the disparity to be from 0-255 for visualization

### 4. Compute Depth Image
- Using the disparity calculated above, compute the depth map. The resultant image has a depth image instead of disparity.

## Dataset

[MiddleBury Stereo Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/#description)

## Result

- Matching features in left and right image
![Feature_matching](https://user-images.githubusercontent.com/90370308/215355870-3752aa49-60f7-4364-9f0d-456060480ce0.png)

- Epipolar line corrosponding to the obtained matching feature
![Matching_Points_and_Epipolar_line](https://user-images.githubusercontent.com/90370308/215355914-a54e76fe-34c0-45aa-bed6-9196f453f382.png)

- Rectified epipolar lines
![Rectified_Epipolar](https://user-images.githubusercontent.com/90370308/215359693-9471be64-2e45-4125-9cd5-1ae7914ae24a.png)

- Disparity and Depth map
![final_map](https://user-images.githubusercontent.com/90370308/215364599-e8ef55ee-bfe5-4b03-b2bf-8e7d0549e738.png)

- Disparity and Depth heat map
![final_heatmap](https://user-images.githubusercontent.com/90370308/215364607-21834a6d-c15c-4f24-969f-235d2b5caef3.png)

## Requirement
1. Python 2.0 or above
2. OpenCV

## License

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) Jan 2023 Pradip Kathiriya

