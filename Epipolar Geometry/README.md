# Epipolar Geometry

This project computes and visualizes the epipolar geometry between two images. The fundamental matrix is estimated from feature correspondences using SIFT, feature matching, and RANSAC. The project then finds the epipoles and visualizes epipolar lines for selected corresponding points.

## Method

SIFT is used to detect and describe keypoints in both images. The descriptors are matched using a FLANN-based matcher with a ratio test to remove ambiguous matches. The fundamental matrix is then estimated from the matched points using RANSAC, which separates the inlier and outlier correspondences.

The epipoles are obtained from the null spaces of \(F\) and \(F^T\) using SVD. For selected inlier correspondences, the epipolar lines are computed using

$$
l_1 = F^T x_2,\qquad l_2 = F x_1
$$

where $x_1$ and $x_2$ are corresponding points in homogeneous coordinates.

## Results

<table>
  <tr>
    <td align="center">Original Image 1</td>
    <td align="center">Original Image 2</td>
  </tr>
  <tr>
    <td><img src="01.JPG" width="300"></td>
    <td><img src="02.JPG" width="300"></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">Feature matching</td>
  </tr>
  <tr>
    <td><img src="res05.jpg" width="600"></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">Epipole of the first image</td>
    <td align="center">Epipole of the second image</td>
  </tr>
  <tr>
    <td><img src="res06.jpg" width="300"></td>
    <td><img src="res07.jpg" width="300"></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">Epipolar lines</td>
  </tr>
  <tr>
    <td><img src="res08.jpg" width="600"></td>
  </tr>
</table>
