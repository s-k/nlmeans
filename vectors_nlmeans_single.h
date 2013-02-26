#include <opencv2/core/core.hpp>

cv::Mat vectors_nlmeans_single(cv::Mat img,
                               cv::Mat V,
                               int neighborhoodRadius,
                               int searchWindowRadius,
                               float h,
                               int nThreads);
