#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat K_left = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);

cv::Mat K_right = (cv::Mat_<double>(3, 3) << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0, 1);
cv::Mat C_left =
    (cv::Mat_<double>(3, 4) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
     0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768, -0.0257744366974,
     0.00375618835797, 0.999660727178, 0.00981073058949);
cv::Mat C_right =
    (cv::Mat_<double>(3, 4) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
     0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918,
     0.0179005838253, 0.999517347078, 0.00786212447038);
cv::Mat P_left = K_left * C_left;
cv::Mat P_right = K_right * C_left;
cv::Mat D_left = (cv::Mat_<double>(5, 1) << -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);
cv::Mat D_right = (cv::Mat_<double>(5, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

Eigen::MatrixXd compute_fundamental_matrix(const Eigen::MatrixXd &points1,
                                           const Eigen::MatrixXd &points2);

Eigen::MatrixXd computeFundamentalmatrixNormalized(const Eigen::MatrixXd &points1,
                                                   const Eigen::MatrixXd &points2);

Eigen::Vector3d compute_epipole(const Eigen::MatrixXd &F);

std::pair<Eigen::Matrix3d, Eigen::Matrix3d>
compute_matching_homographies(const Eigen::Vector3d &e2, const Eigen::MatrixXd &F,
                              const cv::Mat &im2, const Eigen::MatrixXd &points1,
                              const Eigen::MatrixXd &points2);

void obtainCorrespondingPoints(cv::Mat &image_left, cv::Mat &image_right,
                               std::vector<cv::Point2f> &matched_left,
                               std::vector<cv::Point2f> &matched_right, int num_points = 20,
                               bool show = false);

void equalizeStereoHist(cv::Mat &image1, cv::Mat &image2, int method = 0, bool show = false);

void undistortStereoImages(cv::Mat &image_left, cv::Mat &image_right, cv::Mat &undistort_left,
                           cv::Mat &undistort_right, const cv::Mat &K_left, const cv::Mat &K_right,
                           const cv::Mat &D_left, const cv::Mat &D_right, bool show = false);

Eigen::MatrixXd convertToEigenMatrix(const std::vector<cv::Point3f> &points);

Eigen::VectorXd leastSquares(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

Eigen::MatrixXd divideByZ(const Eigen::MatrixXd &points);

cv::Mat eigenToMat(const Eigen::MatrixXd &eigenMatrix);

int main() {
    cv::Mat img1 = cv::imread(
        "/Users/youngjin/Desktop/Dev_python/cv_study/stereo_calibration/original_images/left.png",
        cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(
        "/Users/youngjin/Desktop/Dev_python/cv_study/stereo_calibration/original_images/right.png",
        cv::IMREAD_GRAYSCALE);

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
    // std::cout << "(" << matched_left_eigen.rows() << ", " << matched_left_eigen.cols() << ")";

    Eigen::MatrixXd F = computeFundamentalmatrixNormalized(matched_left_eigen, matched_right_eigen);
    // std::cout << F << std::endl;
    // std::cout << matched_left_eigen.row(0) << std::endl;
    Eigen::Vector3d p1 = matched_left_eigen.row(0);
    Eigen::Vector3d p2 = matched_right_eigen.row(0);
    // std::cout << p1.transpose() << std::endl;
    // std::cout << round(p1.transpose() * F * p2) << std::endl;

    Eigen::Vector3d e1 = compute_epipole(F);
    Eigen::Vector3d e2 = compute_epipole(F.transpose());
    // std::cout << e1.transpose() << std::endl << e2.transpose() << std::endl;
    // std::cout << round(e2.transpose() * F * e1) << std::endl;

    std::pair<Eigen::Matrix3d, Eigen::Matrix3d> homographies =
        compute_matching_homographies(e2, F, img2, matched_left_eigen, matched_right_eigen);
    // std::cout << homographies.first << std::endl << homographies.second << std::endl;

    Eigen::MatrixXd new_points1 =
        divideByZ(homographies.first * matched_left_eigen.transpose()).transpose();
    // std::cout << new_points1 << std::endl;
    Eigen::MatrixXd new_points2 =
        divideByZ(homographies.second * matched_right_eigen.transpose()).transpose();

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
}

void equalizeStereoHist(cv::Mat &image1, cv::Mat &image2, int method, bool show) {
    if (method == 0) {
        cv::equalizeHist(image1, image1);
        cv::equalizeHist(image2, image2);
    } else if (method == 1) {
        cv::Ptr<cv::CLAHE> clahe_left = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Ptr<cv::CLAHE> clahe_right = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe_left->apply(image1, image1);
        clahe_right->apply(image2, image2);
    }

    if (show) {
        cv::Mat compare_result;
        cv::hconcat(image1, image2, compare_result);
        cv::imshow("histogram equalization", compare_result);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

// Obtain corresponding points in stereo images via ORB to obtain R and t
void obtainCorrespondingPoints(cv::Mat &image_left, cv::Mat &image_right,
                               std::vector<cv::Point2f> &matched_left,
                               std::vector<cv::Point2f> &matched_right, int num_points, bool show) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp_left, kp_right;
    cv::Mat des_left, des_right;

    orb->detectAndCompute(image_left, cv::noArray(), kp_left, des_left);
    orb->detectAndCompute(image_right, cv::noArray(), kp_right, des_right);

    cv::BFMatcher bf(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    bf.match(des_left, des_right, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    matches.resize(std::min(num_points, static_cast<int>(matches.size())));

    matched_left.clear();
    matched_right.clear();
    for (const auto &match : matches) {
        matched_left.push_back(kp_left[match.queryIdx].pt);
        matched_right.push_back(kp_right[match.trainIdx].pt);
    }

    if (show) {
        cv::Mat matched_image;
        cv::drawMatches(image_left, kp_left, image_right, kp_right, matches, matched_image,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Display the matched image
        cv::imshow("Matched Features", matched_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void undistortStereoImages(cv::Mat &image_left, cv::Mat &image_right, cv::Mat &undistort_left,
                           cv::Mat &undistort_right, const cv::Mat &K_left, const cv::Mat &K_right,
                           const cv::Mat &D_left, const cv::Mat &D_right, bool show) {
    cv::undistort(image_left, undistort_left, K_left, D_left);
    cv::undistort(image_right, undistort_right, K_right, D_right);

    if (show) {
        cv::Mat undistort_result;
        cv::hconcat(undistort_left, undistort_right, undistort_result);
        cv::imshow("undistorted image", undistort_result);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

Eigen::MatrixXd convertToEigenMatrix(const std::vector<cv::Point3f> &points) {
    int num_points = static_cast<int>(points.size());
    Eigen::MatrixXd matrix(num_points, 3);

    for (int i = 0; i < num_points; ++i) {
        const cv::Point3f &point = points[i];
        matrix.row(i) << point.x, point.y, point.z;
    }

    return matrix;
}

Eigen::MatrixXd computeFundamentalmatrixNormalized(const Eigen::MatrixXd &points1,
                                                   const Eigen::MatrixXd &points2) {
    // Validate points
    assert(points1.rows() == points2.rows() && points1.cols() == 3 && points2.cols() == 3);

    int n = points1.rows();

    // Compute centroid of points
    Eigen::Vector3d c1 = points1.colwise().mean();
    Eigen::Vector3d c2 = points2.colwise().mean();

    // Compute the scaling factor
    double s1 =
        std::sqrt(2 / ((points1.rowwise() - c1.transpose()).rowwise().squaredNorm().mean()));
    double s2 =
        std::sqrt(2 / ((points2.rowwise() - c2.transpose()).rowwise().squaredNorm().mean()));

    // Compute the normalization matrices for both points
    Eigen::Matrix3d T1, T2;
    T1 << s1, 0, -s1 * c1(0), 0, s1, -s1 * c1(1), 0, 0, 1;

    T2 << s2, 0, -s2 * c2(0), 0, s2, -s2 * c2(1), 0, 0, 1;

    // Normalize the points
    Eigen::MatrixXd points1_n = (T1 * points1.transpose()).transpose();
    Eigen::MatrixXd points2_n = (T2 * points2.transpose()).transpose();

    // Compute the normalized fundamental matrix
    Eigen::MatrixXd F_n = compute_fundamental_matrix(points1_n, points2_n);

    // De-normalize the fundamental matrix
    return T2.transpose() * F_n * T1;
}

Eigen::MatrixXd compute_fundamental_matrix(const Eigen::MatrixXd &points1,
                                           const Eigen::MatrixXd &points2) {
    // Validate points
    assert(points1.rows() == points2.rows() && points1.cols() == 3 && points2.cols() == 3);

    int n = points1.rows();

    // Extract homogeneous coordinates
    Eigen::VectorXd u1 = points1.col(0);
    Eigen::VectorXd v1 = points1.col(1);
    Eigen::VectorXd u2 = points2.col(0);
    Eigen::VectorXd v2 = points2.col(1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

    // Construct the matrix A
    Eigen::MatrixXd A(n, 9);
    A << u2.cwiseProduct(u1), u2.cwiseProduct(v1), u2, v2.cwiseProduct(u1), v2.cwiseProduct(v1), v2,
        u1, v1, one;

    // Perform SVD on A and find the minimum value of |Af|
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().rightCols<1>();
    Eigen::MatrixXd F = Eigen::MatrixXd::Map(f.data(), 3, 3); // Reshape f as a matrix

    // Constrain F by making rank 2
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_F(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd S = svd_F.singularValues();
    S(2) = 0; // Zero out the last singular value
    F = svd_F.matrixU() * S.asDiagonal() * svd_F.matrixV().transpose(); // Recombine again

    return F;
}

Eigen::Vector3d compute_epipole(const Eigen::MatrixXd &F) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullV);
    Eigen::VectorXd e = svd.matrixV().col(2);
    return e / e(2);
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d>
compute_matching_homographies(const Eigen::Vector3d &e2, const Eigen::MatrixXd &F,
                              const cv::Mat &im2, const Eigen::MatrixXd &points1,
                              const Eigen::MatrixXd &points2) {
    int h = im2.rows;
    int w = im2.cols;

    // Create the homography matrix H2 that moves the epipole to infinity
    Eigen::Matrix3d T;
    T << 1, 0, -w / 2.0, 0, 1, -h / 2.0, 0, 0, 1;

    Eigen::Vector3d e2_p = T * e2;
    e2_p /= e2_p(2);
    double e2x = e2_p(0);
    double e2y = e2_p(1);
    // std::cout << e2x << ' ' << e2y << std::endl;

    // Create the rotation matrix to rotate the epipole back to X axis
    double a = (e2x >= 0) ? 1.0 : -1.0;
    double R1 = a * e2x / std::sqrt(e2x * e2x + e2y * e2y);
    double R2 = a * e2y / std::sqrt(e2x * e2x + e2y * e2y);
    Eigen::Matrix3d R;
    R << R1, R2, 0, -R2, R1, 0, 0, 0, 1;
    e2_p = R * e2_p;
    double x = e2_p(0);

    // Create matrix to move the epipole to infinity
    Eigen::Matrix3d G;
    G << 1, 0, 0, 0, 1, 0, -1 / x, 0, 1;

    // Create the overall transformation matrix H2
    Eigen::Matrix3d H2 = T.inverse() * G * R * T;
    // std::cout << H2 << std::endl;
    // std::cout << '(' << points1.rows() << ',' << points1.cols() << std::endl;

    // Create the corresponding homography matrix for the other image
    Eigen::Matrix3d e_x;
    e_x << 0, -e2(2), e2(1), e2(2), 0, -e2(0), -e2(1), e2(0), 0;
    Eigen::Matrix3d M = e_x * F + e2 * Eigen::RowVector3d::Ones();
    // std::cout << '(' << M.rows() << ',' << M.cols() << ')' << std::endl;
    Eigen::MatrixXd points1_t = divideByZ((H2 * M * points1.transpose()));
    Eigen::MatrixXd points2_t = divideByZ((H2 * points2.transpose()));
    // Eigen::MatrixXd points1_t = divideByZ(points1_);

    Eigen::VectorXd b = points2_t.row(0);
    // std::cout << b.rows() << std::endl;
    // std::cout << b.cols() << std::endl;
    // std::cout << points1_t.rows() << std::endl;
    // std::cout << points1_t.cols() << std::endl;

    Eigen::VectorXd coeff = leastSquares(points1_t.transpose(), b);
    Eigen::Matrix3d H_A;
    H_A << coeff(0), coeff(1), coeff(2), 0, 1, 0, 0, 0, 1;

    Eigen::Matrix3d H1 = H_A * H2 * M;
    return std::make_pair(H1, H2);
}

Eigen::VectorXd leastSquares(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
    // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    // Eigen::VectorXd x = qr.solve(b);
    return A.colPivHouseholderQr().solve(b);
    // return x;
}

Eigen::MatrixXd divideByZ(const Eigen::MatrixXd &points) {
    Eigen::MatrixXd divided_points(points.rows(), points.cols());
    for (int i = 0; i < points.cols(); ++i) {
        double z_value = points(2, i);
        divided_points.col(i) = points.col(i) / z_value;
    }
    return divided_points;
}

cv::Mat eigenToMat(const Eigen::MatrixXd &eigenMatrix) {
    // Create a cv::Mat with the same size as the Eigen matrix
    cv::Mat cvMat(eigenMatrix.rows(), eigenMatrix.cols(), CV_64FC1);

    // Copy the data from Eigen matrix to cv::Mat
    for (int i = 0; i < eigenMatrix.rows(); ++i) {
        for (int j = 0; j < eigenMatrix.cols(); ++j) {
            cvMat.at<double>(i, j) = eigenMatrix(i, j);
        }
    }

    return cvMat;
}