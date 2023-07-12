import cv2
import numpy as np

# Intrinsic parameters and distortion coefficients
# Left camera
left_intrinsic = np.array(
    [
        [457.587, 0, 379.999],
        [0, 456.134, 255.238],
        [0, 0, 1],
    ]
)

left_distortion = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05])

# Right camera
right_intrinsic = np.array(
    [
        [458.654, 0, 369.215],
        [0, 457.296, 248.375],
        [0, 0, 1],
    ]
)

right_distortion = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

# Load left and right images
left_gray = cv2.imread("left.png", cv2.COLOR_BGR2GRAY)
right_gray = cv2.imread("right.png", cv2.COLOR_BGR2GRAY)

# Extract and match keypoints using ORB
orb = cv2.ORB_create()
keypoints_right, descriptors_right = orb.detectAndCompute(left_gray, None)
keypoints_left, descriptors_left = orb.detectAndCompute(right_gray, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors_right, descriptors_left)

# num_keypoints = 20
# selected_matches = sorted(matches, key=lambda x: x.distance)[:num_keypoints]

# Undistort feature points
right_undistorted_points = cv2.undistortPoints(
    np.float32([keypoint.pt for keypoint in keypoints_right]).reshape(-1, 1, 2),
    right_intrinsic,
    right_distortion,
    None,
    right_intrinsic,
)

left_undistorted_points = cv2.undistortPoints(
    np.float32([keypoint.pt for keypoint in keypoints_left]).reshape(-1, 1, 2),
    left_intrinsic,
    left_distortion,
    None,
    left_intrinsic,
)

# Estimate fundamental matrix and essential matrix using undistorted keypoints
F, mask = cv2.findFundamentalMat(
    right_undistorted_points, left_undistorted_points, cv2.FM_RANSAC, 0.1, 0.99
)
E = np.dot(right_intrinsic.T, np.dot(F, left_intrinsic))

# Step 5: Obtain R and t from the essential matrix
_, R, t, _ = cv2.recoverPose(
    E, right_undistorted_points, left_undistorted_points, right_intrinsic
)

# Perform rectification
R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
    right_intrinsic,
    right_distortion,
    left_intrinsic,
    left_distortion,
    right_gray.shape[::-1],
    R,
    t,
)

# Convert undistorted keypoints to homogeneous coordinates
right_undistorted_points_homogeneous = (
    cv2.convertPointsToHomogeneous(right_undistorted_points).squeeze(1).T
)
left_undistorted_points_homogeneous = (
    cv2.convertPointsToHomogeneous(left_undistorted_points).squeeze(1).T
)

# Rectify undistorted keypoints
rectified_right_points_homogeneous = np.dot(
    R1, right_undistorted_points_homogeneous
) + P1[:, -1].reshape(3, 1)
rectified_left_points_homogeneous = np.dot(
    R2, left_undistorted_points_homogeneous
) + P2[:, -1].reshape(3, 1)

# Convert rectified keypoints back to 2D coordinates
rectified_right_points = cv2.convertPointsFromHomogeneous(
    rectified_right_points_homogeneous.T.reshape(-1, 1, 3)
)
rectified_left_points = cv2.convertPointsFromHomogeneous(
    rectified_left_points_homogeneous.T.reshape(-1, 1, 3)
)
print(rectified_left_points, rectified_right_points)
# Visualize undistorted and rectified keypoints
right_image_keypoints = cv2.drawKeypoints(
    right_gray, keypoints_right, None, color=(0, 0, 255), flags=0
)
left_image_keypoints = cv2.drawKeypoints(
    left_gray, keypoints_left, None, color=(0, 0, 255), flags=0
)

rectified_image = np.concatenate((right_gray, left_gray), axis=1)
rectified_image = cv2.cvtColor(rectified_image, cv2.COLOR_GRAY2BGR)
for point in rectified_right_points:
    x, y = point[0]
    cv2.circle(rectified_image, (int(x), int(y)), 5, (0, 0, 255), -1)

for point in rectified_left_points:
    x, y = point[0]
    x += right_gray.shape[1]  # Shift x-coordinate for left keypoints
    cv2.circle(rectified_image, (int(x), int(y)), 5, (0, 0, 255), -1)

# Display the images
cv2.imshow("Right Keypoints", right_image_keypoints)
cv2.imshow("Left Keypoints", left_image_keypoints)
cv2.imshow("Rectified Keypoints", rectified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
