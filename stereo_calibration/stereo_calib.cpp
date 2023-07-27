#include <Eigen/Dense>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat K_left = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296,
                  248.375, 0, 0, 1);

cv::Mat K_right = (cv::Mat_<double>(3, 3) << 457.587, 0, 379.999, 0, 456.134,
                   255.238, 0, 0, 1);
cv::Mat C_left =
    (cv::Mat_<double>(3, 4) << 0.0148655429818, -0.999880929698,
     0.00414029679422, -0.0216401454975, 0.999557249008, 0.0149672133247,
     0.025715529948, -0.064676986768, -0.0257744366974, 0.00375618835797,
     0.999660727178, 0.00981073058949);
cv::Mat C_right =
    (cv::Mat_<double>(3, 4) << 0.0125552670891, -0.999755099723,
     0.0182237714554, -0.0198435579556, 0.999598781151, 0.0130119051815,
     0.0251588363115, 0.0453689425024, -0.0253898008918, 0.0179005838253,
     0.999517347078, 0.00786212447038);
cv::Mat P_left = K_left * C_left;
cv::Mat P_right = K_right * C_left;
cv::Mat D_left = (cv::Mat_<double>(5, 1) << -0.28368365, 0.07451284,
                  -0.00010473, -3.55590700e-05);
cv::Mat D_right = (cv::Mat_<double>(5, 1) << -0.28340811, 0.07395907,
                   0.00019359, 1.76187114e-05);

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> compute_matching_homographies(
    const Eigen::Vector3d& e2, const Eigen::Matrix3d& F, const cv::Mat& img2,
    const Eigen::MatrixXd& points1, const Eigen::MatrixXd& points2);
void obtainCorrespondingPoints(cv::Mat& image_left, cv::Mat& image_right,
                               std::vector<cv::Point2f>& matched_left,
                               std::vector<cv::Point2f>& matched_right,
                               int num_points = 20, bool show = false);
void equalizeStereoHist(cv::Mat& image1, cv::Mat& image2, int method = 0,
                        bool show = false);
cv::Mat compute_fundamental_matrix_normalized(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2);
cv::Mat compute_fundamental_matrix(const std::vector<cv::Point2f>& points1,
                                   const std::vector<cv::Point2f>& points2);

cv::Point3d compute_epipole(const cv::Mat& F);
double compute_scaling_factor(const std::vector<cv::Point2f>& points,
                              const cv::Point2f& centroid);

int main() {
  cv::Mat img1 = cv::imread(
      "/Users/youngjin/Desktop/Dev_python/cv_study/stereo_calibration/"
      "original_images/left.png",
      cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(
      "/Users/youngjin/Desktop/Dev_python/cv_study/stereo_calibration/"
      "original_images/right.png",
      cv::IMREAD_GRAYSCALE);

  //   cv::Mat combined_img(img1.rows, img1.cols + img2.cols, CV_8UC1);
  //   cv::hconcat(img1, img2, combined_img);

  //   cv::imshow("Image loaded", combined_img);
  //   cv::waitKey(0);
  //   cv::destroyAllWindows();

  std::vector<cv::Point2f> matched_left, matched_right;
  equalizeStereoHist(img1, img2, 1);
  obtainCorrespondingPoints(img1, img2, matched_left, matched_right, 50, true);

  // Undistort the matched points
  std::vector<cv::Point2f> undistorted_left, undistorted_right;
  cv::undistortPoints(matched_left, undistorted_left, K_left, D_left);
  cv::undistortPoints(matched_right, undistorted_right, K_right, D_right);

  // Convert undistorted points to homogeneous coordinates
  std::vector<cv::Point3f> matched_left_homogeneous, matched_right_homogeneous;
  cv::convertPointsToHomogeneous(matched_left, matched_left_homogeneous);
  cv::convertPointsToHomogeneous(matched_right, matched_right_homogeneous);
  std::cout << matched_left_homogeneous << std::endl;
  // Compute the normalized fundamental matrix
  cv::Mat F =
      compute_fundamental_matrix_normalized(matched_left, matched_right);

  //   std::cout << F << std::endl;

  // Compute the epipoles
  cv::Point3d e1 = compute_epipole(F);
  cv::Point3d e2 = compute_epipole(F.t());

  // Validate the fundamental matrix equation for epipoles
  double result =
      e2.x * F.at<double>(0, 0) * e1.x + e2.y * F.at<double>(1, 0) * e1.x +
      e2.x * F.at<double>(0, 1) * e1.y + e2.y * F.at<double>(1, 1) * e1.y +
      e2.x * F.at<double>(0, 2) * e1.z + e2.y * F.at<double>(1, 2) * e1.z;

  //   // Print the result
  //   std::cout << "Result: " << result << std::endl;

  //   // Print the epipoles
  //   std::cout << "Epipole 1: " << e1 << std::endl;
  //   std::cout << "Epipole 2: " << e2 << std::endl;

  return 0;
}

// Obtain corresponding points in stereo images via ORB to obtain R and t
void obtainCorrespondingPoints(cv::Mat& image_left, cv::Mat& image_right,
                               std::vector<cv::Point2f>& matched_left,
                               std::vector<cv::Point2f>& matched_right,
                               int num_points, bool show) {
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  std::vector<cv::KeyPoint> kp_left, kp_right;
  cv::Mat des_left, des_right;

  orb->detectAndCompute(image_left, cv::noArray(), kp_left, des_left);
  orb->detectAndCompute(image_right, cv::noArray(), kp_right, des_right);

  cv::BFMatcher bf(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  bf.match(des_left, des_right, matches);

  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) {
              return a.distance < b.distance;
            });
  matches.resize(std::min(num_points, static_cast<int>(matches.size())));

  matched_left.clear();
  matched_right.clear();
  for (const auto& match : matches) {
    matched_left.push_back(kp_left[match.queryIdx].pt);
    matched_right.push_back(kp_right[match.trainIdx].pt);
  }

  if (show) {
    cv::Mat matched_image;
    cv::drawMatches(image_left, kp_left, image_right, kp_right, matches,
                    matched_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the matched image
    cv::imshow("Matched Features", matched_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}

void equalizeStereoHist(cv::Mat& image1, cv::Mat& image2, int method,
                        bool show) {
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
    cv::imshow("left image", compare_result);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}

// Compute the fundamental matrix given the point correspondences
cv::Mat compute_fundamental_matrix(const std::vector<cv::Point2f>& points1,
                                   const std::vector<cv::Point2f>& points2) {
  // Validate points
  assert(points1.size() == points2.size() && !points1.empty());

  cv::Mat A(points1.size(), 9, CV_64F);
  for (size_t i = 0; i < points1.size(); ++i) {
    double u1 = points1[i].x;
    double v1 = points1[i].y;
    double u2 = points2[i].x;
    double v2 = points2[i].y;

    A.at<double>(i, 0) = u2 * u1;
    A.at<double>(i, 1) = u2 * v1;
    A.at<double>(i, 2) = u2;
    A.at<double>(i, 3) = v2 * u1;
    A.at<double>(i, 4) = v2 * v1;
    A.at<double>(i, 5) = v2;
    A.at<double>(i, 6) = u1;
    A.at<double>(i, 7) = v1;
    A.at<double>(i, 8) = 1;
  }

  // Perform SVD on A and find the minimum value of |Af|
  cv::Mat U, S, Vt;
  cv::SVDecomp(A, S, U, Vt, cv::SVD::FULL_UV);
  cv::Mat f = Vt.row(Vt.rows - 1);
  f = f.reshape(0, 3);  // reshape f as a matrix

  // Constrain F: make rank 2 by zeroing out last singular value
  cv::Mat F;
  cv::SVD::compute(f, S, U, Vt);
  S.at<double>(2) = 0;            // zero out the last singular value
  F = U * cv::Mat::diag(S) * Vt;  // recombine again
  return F;
}

// Compute the normalized fundamental matrix
cv::Mat compute_fundamental_matrix_normalized(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2) {
  // Validate points
  assert(points1.size() == points2.size() && !points1.empty());

  // Compute centroid of points
  cv::Point2f c1(0, 0);
  cv::Point2f c2(0, 0);
  for (size_t i = 0; i < points1.size(); ++i) {
    c1 += points1[i];
    c2 += points2[i];
  }
  c1 *= (1.0 / points1.size());
  c2 *= (1.0 / points2.size());

  // Compute the scaling factor
  double s1 = compute_scaling_factor(points1, c1);
  double s2 = compute_scaling_factor(points2, c2);

  // Compute the normalization matrix for both the points
  cv::Mat T1 =
      (cv::Mat_<double>(3, 3) << s1, 0, -s1 * c1.x, 0, s1, -s1 * c1.y, 0, 0, 1);
  cv::Mat T2 =
      (cv::Mat_<double>(3, 3) << s2, 0, -s2 * c2.x, 0, s2, -s2 * c2.y, 0, 0, 1);

  // Normalize the points
  std::vector<cv::Point2f> points1_n, points2_n;
  cv::perspectiveTransform(points1, points1_n, T1.inv());
  cv::perspectiveTransform(points2, points2_n, T2.inv());

  // Compute the normalized fundamental matrix
  cv::Mat F_n = compute_fundamental_matrix(points1_n, points2_n);

  // De-normalize the fundamental matrix
  return T2.t() * F_n * T1;
}

// Compute epipole using the fundamental matrix
cv::Point3d compute_epipole(const cv::Mat& F) {
  cv::SVD svd(F, cv::SVD::FULL_UV);
  cv::Mat V = svd.vt.t();
  cv::Mat e = V.row(V.rows - 1);
  e /= e.at<double>(2);
  return cv::Point3d(e.at<double>(0), e.at<double>(1), e.at<double>(2));
}

// Compute the scaling factor for points
double compute_scaling_factor(const std::vector<cv::Point2f>& points,
                              const cv::Point2f& centroid) {
  double sum_squared_distance = 0.0;
  for (const cv::Point2f& p : points) {
    double squared_distance = (p.x - centroid.x) * (p.x - centroid.x) +
                              (p.y - centroid.y) * (p.y - centroid.y);
    sum_squared_distance += squared_distance;
  }
  double mean_squared_distance = sum_squared_distance / points.size();
  return std::sqrt(2.0 / mean_squared_distance);
}

// void compute_matching_homographies(const cv::Point3d& e2, const cv::Mat& F,
//                                    const cv::Mat& im2,
//                                    const std::vector<cv::Point2f>& points1,
//                                    const std::vector<cv::Point2f>& points2,
//                                    cv::Mat& H1, cv::Mat& H2) {
//   int h = im2.rows;
//   int w = im2.cols;

//   // ... (rest of the code as before)

//   // Create the translation matrix to shift to the image center
//   cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -w / 2, 0, 1, -h / 2, 0, 0,
//   1); cv::Mat e2_p = T * cv::Mat(e2); e2_p /= e2_p.at<double>(2); double e2x
//   = e2_p.at<double>(0); double e2y = e2_p.at<double>(1);

//   // Create the rotation matrix to rotate the epipole back to X axis
//   double a = (e2x >= 0) ? 1 : -1;
//   double R1 = a * e2x / std::sqrt(e2x * e2x + e2y * e2y);
//   double R2 = a * e2y / std::sqrt(e2x * e2x + e2y * e2y);
//   cv::Mat R = (cv::Mat_<double>(3, 3) << R1, R2, 0, -R2, R1, 0, 0, 0, 1);
//   e2_p = R * e2_p;
//   double x = e2_p.at<double>(0);

//   // Create matrix to move the epipole to infinity
//   cv::Mat G = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, -1 / x, 0, 1);

//   // Create the overall transformation matrix
//   H2 = T.inv() * G * R * T;

//   // Create the corresponding homography matrix for the other image
//   cv::Mat e_x = (cv::Mat_<double>(3, 3) << 0, -e2.z, e2.y, e2.z, 0, -e2.x,
//                  -e2.y, e2.x, 0);
//   cv::Mat M = e_x * F +
//               e2.cross(cv::Mat_<double>(F.row(0)),
//               cv::Mat_<double>(F.row(1)));

//   cv::Mat points1_t = H2 * M * cv::Mat(points1).t();
//   cv::Mat points2_t = H2 * cv::Mat(points2).t();
//   points1_t /= points1_t.row(2);
//   points2_t /= points2_t.row(2);
//   cv::Mat b = points2_t.row(0);
//   cv::Mat a = cv::Mat_<double>(1, 3);
//   cv::solve(points1_t.t(), b.t(), a, cv::DECOMP_SVD);
//   cv::Mat H_A = (cv::Mat_<double>(3, 3) << a.at<double>(0), a.at<double>(1),
//                  a.at<double>(2), 0, 1, 0, 0, 0, 1);
//   // Update the homography matrices H1 and H2 by reference
//   H1 = H_A * H2 * M;
// }