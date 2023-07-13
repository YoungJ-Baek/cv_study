import cv2
import config as cf
import numpy as np
from stereo_utils import *
from skimage.transform import warp, ProjectiveTransform


# Load camera parameter
def loadCameraParameter():
    return cf.K_left, cf.K_right, cf.D_left, cf.D_right, cf.P_left, cf.P_right


# Load left and right images
def loadStereoImages(show=False):
    left_gray = cv2.imread("left.png", cv2.COLOR_BGR2GRAY)
    right_gray = cv2.imread("right.png", cv2.COLOR_BGR2GRAY)

    if show == True:
        dst = np.hstack((left_gray, right_gray))
        cv2.imshow("stereo images", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return left_gray, right_gray


# Equalize histogram of stereo images, 0 is conventional, and 1 is CLAHE
def equalizeStereoHist(image1, image2, method=0, show=False):
    if method == 0:
        image_left = cv2.equalizeHist(image1)
        image_right = cv2.equalizeHist(image2)
    elif method == 1:
        clahe_left = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_right = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_left = clahe_left.apply(image1)
        image_right = clahe_right.apply(image2)

    if show == True:
        compare_result = np.hstack((image1, image_left))
        cv2.imshow("left image", compare_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_left, image_right


# Feature matching stereo images via ORB to obtain R and t
def obtainCorrespondingPoints(image_left, image_right, num_points=20, show=False):
    orb = cv2.ORB_create()
    matched_left, matched_right = [], []

    kp_left, des_left = orb.detectAndCompute(image_left, None)
    kp_right, des_right = orb.detectAndCompute(image_right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_left, des_right), key=lambda x: x.distance)[
        :num_points
    ]

    points_left = np.float32(
        [kp_left[m.queryIdx].pt for m in matches]
    )  # .reshape(-1, 1, 2)
    points_right = np.float32(
        [kp_right[m.trainIdx].pt for m in matches]
    )  # .reshape(-1, 1, 2)

    matched_left = np.array(points_left)
    matched_right = np.array(points_right)

    if show == True:
        matched_image = cv2.drawMatches(
            image_left,
            kp_left,
            image_right,
            kp_right,
            matches,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        )

        # Display the matched image
        cv2.imshow("Matched Features", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return matched_left, matched_right


# Undistort whole images
def undistortStereoImages(
    image_left, image_right, K_left, K_right, D_left, D_right, show=False
):
    undistort_left = cv2.undistort(image_left, K_left, D_left)
    undistort_right = cv2.undistort(image_right, K_right, D_right)

    if show == True:
        dst = np.hstack((image_left, undistort_left))
        cv2.imshow("undistortion", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return undistort_left, undistort_right


# Calculate R (rotation matrix) and t (translation vector) via F and E using matched points
def calculateRotationTranslation(
    matched_left, matched_right, K_left, K_right, D_left, D_right
):
    matched_left = matched_left.reshape(-1, 2)
    matched_right = matched_right.reshape(-1, 2)

    # H: homogeneous
    H_left = cv2.convertPointsToHomogeneous(matched_left).reshape(-1, 3)
    H_right = cv2.convertPointsToHomogeneous(matched_right).reshape(-1, 3)
    F, _ = cv2.findFundamentalMat(H_left, H_right, cv2.FM_8POINT)
    E = np.matmul(np.matmul(K_right.T, F), K_left)
    _, R, t, _ = cv2.recoverPose(
        E=E,
        points1=matched_left,
        points2=matched_right,
        cameraMatrix=K_left,
    )

    return R, t


# Rectify stereo images
def rectifyStereoImages(
    image_left, image_right, K_left, D_left, K_right, D_right, R, T, show=False
):
    # Get the image size
    img_size = (image_left.shape[1], image_left.shape[0])

    # Compute the rectification transforms
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        img_size,
        R,
        T,
        alpha=0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, img_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, img_size, cv2.CV_32FC1
    )

    # Remap the images using the rectification maps
    img_left_rectified = cv2.remap(image_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(
        image_right, map_right_x, map_right_y, cv2.INTER_LINEAR
    )

    if show == True:
        dst = np.hstack((img_left_rectified, img_right_rectified))
        cv2.imshow("rectification", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_left_rectified, img_right_rectified


def main():
    K_left, K_right, D_left, D_right, P_left, P_right = loadCameraParameter()
    image_left, image_right = loadStereoImages(show=False)
    image_left, image_right = equalizeStereoHist(
        image_left, image_right, method=1, show=False
    )
    image_left, image_right = undistortStereoImages(
        image_left, image_right, K_left, K_right, D_left, D_right, show=False
    )

    matched_left, matched_right = obtainCorrespondingPoints(
        image_left.astype(np.uint8), image_right.astype(np.uint8), 50, show=False
    )
    matched_left = cv2.convertPointsToHomogeneous(matched_left).reshape(-1, 3)
    matched_right = cv2.convertPointsToHomogeneous(matched_right).reshape(-1, 3)

    # show_matching_result(image_left, image_right, matched_left, matched_right)
    F = compute_fundamental_matrix_normalized(matched_left, matched_right)
    # plot_epipolar_lines(
    #     image_left, image_right, matched_left, matched_right, show_epipole=False
    # )
    e1 = compute_epipole(F)
    e2 = compute_epipole(F.T)
    # print(np.round(e2.T @ F @ e1))
    # plot_epipolar_lines(
    #     image_left, image_right, matched_left, matched_right, show_epipole=True
    # )
    H1, H2 = compute_matching_homographies(
        e2, F, image_right, matched_left, matched_right
    )
    # Transform points based on the homography matrix
    new_points1 = H1 @ matched_left.T
    new_points2 = H2 @ matched_right.T
    new_points1 /= new_points1[2, :]
    new_points2 /= new_points2[2, :]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    im1_warped = warp(image_left, ProjectiveTransform(matrix=np.linalg.inv(H1)))
    im2_warped = warp(image_right, ProjectiveTransform(matrix=np.linalg.inv(H2)))
    # warp images based on the homography matrix
    # im1_warped = cv2.warpPerspective(
    #     image_left,
    #     np.linalg.inv(H1),
    #     (image_left.shape[1], image_left.shape[0]),
    #     # flags=cv2.INTER_LINEAR,
    #     # borderMode=cv2.BORDER_CONSTANT,
    #     # borderValue=0,
    # )
    # im2_warped = cv2.warpPerspective(
    #     image_right,
    #     np.linalg.inv(H2),
    #     (image_right.shape[1], image_right.shape[0]),
    #     # flags=cv2.INTER_LINEAR,
    #     # borderMode=cv2.BORDER_CONSTANT,
    #     # borderValue=0,
    # )
    image = np.hstack((im1_warped, im2_warped))
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    h, w = image_left.shape

    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))

    # plot image 1
    ax1 = axes[0]
    ax1.set_title("Image 1 warped")
    ax1.imshow(im1_warped, cmap="gray")

    # plot image 2
    ax2 = axes[1]
    ax2.set_title("Image 2 warped")
    ax2.imshow(im2_warped, cmap="gray")

    # plot the epipolar lines and points
    n = new_points1.shape[0]
    for i in range(n):
        p1 = new_points1[i]
        p2 = new_points2[i]

        ax1.hlines(p2[1], 0, w, color="orange")
        ax1.scatter(*p1[:2], color="blue")

        ax2.hlines(p1[1], 0, w, color="orange")
        ax2.scatter(*p2[:2], color="blue")
    plt.show()

    # R, t = calculateRotationTranslation(
    #     matched_left, matched_right, K_left, K_right, D_left, D_right
    # )
    # rectified_left, rectified_right = rectifyStereoImages(
    #     image_left.astype(np.uint8),
    #     image_right.astype(np.uint8),
    #     K_left,
    #     D_left,
    #     K_right,
    #     D_right,
    #     R,
    #     t,
    #     show=True,
    # )


if __name__ == "__main__":
    main()
