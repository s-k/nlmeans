#include <opencv2/core/core.hpp>

/**
Removes noise from an image using the non-local means algorithm.

@param img The image to be denoised. Has to have the type 'CV_32FC1' (float).

@param h The filter strenght. Corresponds to the parameter h of non-local means.

@param neighborhoodRadius The radius of the neighborhood to be compared. The
neighborhood will have a size of
(2 * neighborhoodRadius + 1) x (2 * neighborhoodRadius + 1).

@param searchWindowRadius The radius of the search window. The search window
will have a size of
(2 * searchWindowRadius + 1) x (2 * searchWindowRadius + 1).

@param blockSize The image is processed in blocks to utilize CPU caching. This
parameter specifies the block size.

@param nThreads The implementation uses multiple threads. This parameter
specifies the number of threads.

@return the denoised image of type 'CV_32FC1'.
*/
cv::Mat nlmeans(cv::Mat img,
                float h,
                int neighborhoodRadius = 3,
                int searchWindowRadius = 10,
                int blockSize = 150,
                int nThreads = 4);
