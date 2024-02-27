import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('curves.jpg')  # Replace with the path to your image file

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
blue_curve = cv2.bitwise_and(image, image, mask=mask)

# Convert the blue curve image to grayscale
gray_curve = cv2.cvtColor(blue_curve, cv2.COLOR_BGR2GRAY)

# Find contours in the grayscale image
contours, _ = cv2.findContours(gray_curve, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print (len(contours))
# Select specific contours to combine (e.g., second and fourth contours)
selected_contours = [contours[3]]
# Combine selected contours into a single contour
combined_contour = np.concatenate(selected_contours)
# Find the contour with the maximum area (assuming it's the curve)
#max_contour = max(contours, key=cv2.contourArea)

# Extract x and y values from the contour
curve_coordinates = combined_contour[:, 0, :]

# Get 100 equidistant points on the curve
#num_points = min(100, len(combined_contour))  # Ensure not to exceed the number of points in the contour
#indices = np.linspace(0, len(combined_contour) - 1, num_points, dtype=int)
#curve_coordinates = combined_contour[indices,0,:]

# Sort by x values
sorted_coordinates = sorted(curve_coordinates, key=lambda point: point[0])

# Calculate mean y values for the same x
result_coordinates = []
current_x = sorted_coordinates[0][0]
sum_y = 0
count_y = 0

for x, y in sorted_coordinates:
    if x == current_x:
        sum_y += y
        count_y += 1
    else:
        result_coordinates.append((current_x, sum_y / count_y))
        current_x = x
        sum_y = y
        count_y = 1

# Add the last set of coordinates
result_coordinates.append((current_x, sum_y / count_y))

# Print sorted and averaged coordinates
x_list=[]
y_list=[]
for x, y in result_coordinates:
    print(f"X: {x}, Y: {y}")
    x_list.append(x)
    y_list.append(y)


# Optionally, you can visualize the blue curve
#cv2.drawContours(image, [combined_contour], -1, (0, 255, 0), 2)
#cv2.imshow('Blue Curve', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Provided mappings
x_min, x_max = 114, 1351
X_min, X_max = 0, 10

y_min, y_max = 242, 43
Y_min, Y_max = -2.5, 2.5


# Perform the conversion
X = ((np.array(x_list) - x_min) / (x_max - x_min)) * (X_max - X_min) + X_min
Y = ((np.array(y_list) - y_min) / (y_max - y_min)) * (Y_max - Y_min) + Y_min

# Print the converted values
X = np.insert(X, 0, 0)
Y = np.insert(Y, 0, 0)
print(f"x: {x_list}")
print(f"y: {y_list}")
print(f"X: {X.tolist()}")
print(f"Y: {Y.tolist()}")
np.save('data.npy', {'X': X, 'Y': Y})
plt.plot(X,Y)
plt.show()

