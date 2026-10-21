# Perspective Warping for Logo on Football Field

This project involves creating a distorted version of a logo on the football field such that when filmed by a camera positioned at the side of the field, the logo appears undistorted due to perspective. The goal is to create this distorted version of the logo at the center of the field so that when captured by a central camera placed along the field's length axis, the logo appears as the original. The side camera is positioned 40 meters from the center of the field, at a height of 25 meters, with a focal length of 500 pixels.

To solve this, a homography transformation is computed to map the top-down view of the logo (taken by an imaginary camera) to the side camera's perspective. The process involves calculating the camera projection matrices for both the top-down and side cameras, determining the  homography matrix , and applying this transformation to the logo image.

## Method

1. **Compute Camera Projection Matrices:**

   Define the intrinsic matrix $K$ for both cameras:

   $$
   K = \begin{bmatrix}
   f & 0 & c_x \\
   0 & f & c_y \\
   0 & 0 & 1
   \end{bmatrix}
   $$

   For the side camera, compute $P_1 = K [R_1 \mid t_1]$ and for the top-down camera, compute $P_2 = K [R_2 \mid t_2]$.
2. **Compute the Homography Matrix:**

   Compute the homography that maps the top-down view to the side camera view using the projection matrices of both cameras, where $P_1$ is the side camera and $P_2$ is the top-down camera.

   $$
   H = P_1 \cdot P_2^{-1}
   $$
3. **Apply the Homography to the Logo Image:**

   Apply the homography to map each point of the logo from the top-down view to the side-camera view, then warp the full image accordingly.

   $$
   x_{\text{side}} = H \cdot x_{\text{top}}
   $$

## Result

<table>
  <tr>
    <td align="center">Original Image</td>
    <td align="center">Final Result</td>
  </tr>
  <tr>
    <td><img src="logo.png" width="300"></td>
    <td><img src="res12.jpg" width="300"></td>
  </tr>
</table>
