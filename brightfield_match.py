import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read Image In
filepath = 'data/f0113/F0113_10012021_initial_BF.tif'
imp_c = cv2.imread(filepath)

# Convert to grayscale and normalize
imp_bw = cv2.cvtColor(imp_c, cv2.COLOR_BGR2GRAY)

plt.imshow(imp_bw)
plt.show()

#mask = cv2.inRange(imp, 50, 140)
#imp = cv2.bitwise_and(imp, mask)


cv2.normalize(imp_bw, imp_bw, 0, 255, cv2.NORM_MINMAX)
imp = 255 - imp_bw

# Blur 
imp = cv2.GaussianBlur(imp, (7,7), 0)
imp = 255 - imp

# Adaptive Thresholding for due to different lighting regions and biofilm
imp = cv2.adaptiveThreshold(imp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
imp = 255 - imp

# Morphological Filtering
k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
imp = cv2.morphologyEx(imp, cv2.MORPH_OPEN, k)
imp = cv2.morphologyEx(imp, cv2.MORPH_CLOSE, k)

# Erode to get thinner lines
k1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
imp = cv2.morphologyEx(imp, cv2.MORPH_ERODE, k1)

plt.imshow(imp)
plt.show()

# Find Contours in order to bound rectangles
contours, hierarchy = cv2.findContours(imp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Sort contours by their size
num_contours = len(contours)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Take top 1-5 contours and use the rest of the contours to mask out stuff we do not care about
new_contours = sorted_contours[1]

# Another thing we can try TODO: 
# Determine bounding box for contour
cv2.drawContours(imp_c, sorted_contours[1], -1, (255,0,0), 3)
rot_rect = cv2.minAreaRect(sorted_contours[1])
box = cv2.boxPoints(rot_rect) 
box = np.int0(box)
cv2.drawContours(imp_c, [box], -1, (0,255,0), 3)

# From our box points, try and mask out everything that we do not care about
# From my observations, it seems that 630 to 1825 is chip height, 1200 height)
# 1575 to 1825 is side stuff that we dont even want (200 height?)
# 1825 to 1925 on width side of things (100 width)
# chip width total is like: 120 to 1920 or 1800 width

# So reduce width by 1/18 on each side 
# reduce height by 1/6 on each side
# Gives us bottom left, top left, top right, bottom right

# Determine the size of the box
box_height = box[0][1] - box[1][1]
box_width = box[2][0] - box[1][0]

# Width Shrink size
width_shrink = int(box_width * (1/20000))
height_shrink = int(box_height * (1/9))

# Adjust all widths
box[0][0] = box[0][0] + width_shrink
box[1][0] = box[1][0] + width_shrink
box[2][0] = box[2][0] - width_shrink
box[3][0] = box[3][0] - width_shrink

# Adjust all heights
box[0][1] = box[0][1] - 1.7 * height_shrink
box[1][1] = box[1][1] + height_shrink
box[2][1] = box[2][1] + height_shrink
box[3][1] = box[3][1] - 1.7 * height_shrink

mask = np.zeros((len(imp), len(imp)))
cv2.fillPoly(mask, [box], 255)
mask = np.array(mask, dtype='uint8')

imp = cv2.bitwise_not(cv2.bitwise_and(imp, mask))

k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
imp = cv2.morphologyEx(imp, cv2.MORPH_OPEN, k2)
imp = cv2.morphologyEx(imp, cv2.MORPH_CLOSE, k2)

imp_reverse_mask = cv2.bitwise_not(imp)
new_bw = cv2.bitwise_and(imp_bw, imp_reverse_mask)
plt.imshow(new_bw)
plt.show()

new_bw = cv2.morphologyEx(new_bw, cv2.MORPH_OPEN, k2)
new_bw = cv2.morphologyEx(new_bw, cv2.MORPH_CLOSE, k2)

dst = cv2.cornerHarris(new_bw, 5, 5, 0.08)

# marking dilated corners 
dst = cv2.dilate(dst, None) 
  
# reverting back to the original image
imp_c[dst > 0.01 * dst.max()]=[255, 0, 0] 
plt.imshow(imp_c)
plt.show()

contours, hierarchy = cv2.findContours(imp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
sort = sorted(contours, key=cv2.contourArea, reverse=True)

cv2.drawContours(imp_c, sort[1], -1, (0,0,255), 3)

rot_rect = cv2.minAreaRect(sort[1])
box = cv2.boxPoints(rot_rect) 
box = np.int0(box)

cv2.drawContours(imp_c, [box], -1, (0,255,0), 3)



# fin_mask = np.zeros((len(imp), len(imp)))
# print("Number of Contours: " + str(len(contours)))
#
# # FIXME: THIS TAKES FOREVER TO ACTUALLY RUN SO GET RID OF IF DON'T NEED
# for i in range(5,len(sorted_contours)):
#     # Determine bounding box for contour
#     rot_rect = cv2.minAreaRect(sorted_contours[i])
#     box = cv2.boxPoints(rot_rect) 
#     box = np.int0(box)

#     mask = np.zeros((len(imp), len(imp)))
#     cv2.fillPoly(mask, [box], 1)

#     fin_mask = np.logical_or(fin_mask, mask)

# # Invert all values to create a mask where we just have everything else but small contour boxes
# fin_mask = fin_mask * 255
# fin_mask = np.array(fin_mask, dtype='uint8')

# plt.imshow(fin_mask)
# plt.show()

# # Now, mask it with our image
# imp = cv2.bitwise_or(imp, fin_mask)

# plt.imshow(imp)
# plt.show()

#cv2.drawContours(imp_c, new_contours, -1, (0,255,0), 3)

# Show IMage
plt.imshow(imp_c)
plt.show()
