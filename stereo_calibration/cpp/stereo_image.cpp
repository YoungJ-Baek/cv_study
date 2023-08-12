#include "stereo_calibration.h"
#include <ctime>
#include <iostream>

int main() {
    clock_t start, finish;
    start = clock();

    cv::Mat K_left = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);

    cv::Mat K_right = (cv::Mat_<double>(3, 3) << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0, 1);
    cv::Mat C_left =
        (cv::Mat_<double>(3, 4) << 0.0148655429818, -0.999880929698, 0.00414029679422,
         -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
         -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949);
    cv::Mat C_right =
        (cv::Mat_<double>(3, 4) << 0.0125552670891, -0.999755099723, 0.0182237714554,
         -0.0198435579556, 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
         -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038);
    cv::Mat P_left = K_left * C_left;
    cv::Mat P_right = K_right * C_left;
    cv::Mat D_left =
        (cv::Mat_<double>(5, 1) << -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);
    cv::Mat D_right =
        (cv::Mat_<double>(5, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    cv::Mat img1 = cv::imread("../../original_images/left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../../original_images/right.png", cv::IMREAD_GRAYSCALE);

    equalizeStereoHist(img1, img2, 1, false);
    cv::Mat undistort_left, undistort_right;
    undistortStereoImages(img1, img2, undistort_left, undistort_right, K_left, K_right, D_left,
                          D_right, false);

    std::vector<cv::Point2f> matched_left, matched_right;
    obtainCorrespondingPoints(undistort_left, undistort_right, matched_left, matched_right, 50,
                              false);

    std::vector<cv::Point3f> matched_left_homogeneous, matched_right_homogeneous;
    cv::convertPointsToHomogeneous(matched_left, matched_left_homogeneous);
    cv::convertPointsToHomogeneous(matched_right, matched_right_homogeneous);

    Eigen::MatrixXd matched_left_eigen = convertToEigenMatrix(matched_left_homogeneous);
    Eigen::MatrixXd matched_right_eigen = convertToEigenMatrix(matched_right_homogeneous);

    Eigen::MatrixXd F = computeFundamentalmatrixNormalized(matched_left_eigen, matched_right_eigen);
    Eigen::Vector3d p1 = matched_left_eigen.row(0);
    Eigen::Vector3d p2 = matched_right_eigen.row(0);

    Eigen::Vector3d e1 = compute_epipole(F);
    Eigen::Vector3d e2 = compute_epipole(F.transpose());

    std::pair<Eigen::Matrix3d, Eigen::Matrix3d> homographies =
        compute_matching_homographies(e2, F, img2, matched_left_eigen, matched_right_eigen);

    Eigen::MatrixXd new_points1 =
        divideByZ(homographies.first * matched_left_eigen.transpose()).transpose();
    Eigen::MatrixXd new_points2 =
        divideByZ(homographies.second * matched_right_eigen.transpose()).transpose();

    finish = clock();
    std::cout << static_cast<double>(finish - start) / CLOCKS_PER_SEC << std::endl;

    cv::Mat im1_warped, im2_warped;
    cv::warpPerspective(img1, im1_warped, eigenToMat(homographies.first.inverse()), img1.size(),
                        cv::INTER_LINEAR);
    cv::warpPerspective(img2, im2_warped, eigenToMat(homographies.second.inverse()), img2.size(),
                        cv::INTER_LINEAR);

    cv::Mat result;
    cv::hconcat(im1_warped, im2_warped, result);
    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
