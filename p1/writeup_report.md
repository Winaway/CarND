
# **Finding Lane Lines on the Road** 

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

First, I converted the images to grayscale, then I use a Gaussian_blur on the grayed image with a kernal_size of 3 .

![image_blur.png](attachment:image_blur.png)

Second, I set up a Canny edge detector to detect the adge of the gray_blur image.

![edges.png](attachment:edges.png)

Third, I build a mask to mask everything else out.
![masked_edges.png](attachment:masked_edges.png)

Fourth, by doing a Hough transform operating on the masked image, we get an array containing the endpoints (x1, y1, x2, y2) of all line segments detected by the transform operation. And I draw this line out with red color.
![line_image.png](attachment:line_image.png)

Fifth, I put the line_image and origin image together to get a weighted image.
![w_images.png](attachment:w_images.png)

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by deviding the lines into two classes according to the slope. Then I append all the endpoints of lines in each class in an array. Then I used np.ployfit() to do a curve fitting on the points we get above. So,we can get the slope and intercept of two lines,one on the left and one on the right.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that the line will shake when there is other object with obvious edges near in the unmasked region.

Another shortcoming could be that the line will shake when the image has a low contrast ratio. It will be hard to detect the lane line.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve the contrast ratio of the images.



