#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Eigen::Matrix3d compute_fundamental_matrix_normalized(
    const Eigen::MatrixXd& points1, const Eigen::MatrixXd& points2);
Eigen::Matrix3d compute_fundamental_matrix(const Eigen::MatrixXd& points1,
                                           const Eigen::MatrixXd& points2);
Eigen::Vector3d compute_epipole(const Eigen::Matrix3d& F);
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> compute_matching_homographies(
    const Eigen::Vector3d& e2, const Eigen::Matrix3d& F, const cv::Mat& img2,
    const Eigen::MatrixXd& points1, const Eigen::MatrixXd& points2);

int main() {
  cv::Mat img1 = cv::imread(
      "/home/youngjin/cv_study/stereo_calibration/original_images/left.png",
      cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(
      "/home/youngjin/cv_study/stereo_calibration/original_images/right.png",
      cv::IMREAD_GRAYSCALE);

  cv::Mat combined_img(img1.rows, img1.cols + img2.cols, CV_8UC1);
  cv::hconcat(img1, img2, combined_img);

  cv::imshow("Image loaded", combined_img);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}