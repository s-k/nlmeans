#include "nlmeans.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "image2vectors_single.h"
#include "vectors_nlmeans_single.h"

cv::Mat nlmeans(cv::Mat img,
                float h,
                int neighborhoodRadius,
                int searchWindowRadius,
                int blockSize,
                int nThreads)
{
    assert(CV_32FC1 == img.type());

    int borderSize = neighborhoodRadius + searchWindowRadius;
    cv::Mat paddedImg;
    cv::copyMakeBorder(img,
                       paddedImg,
                       borderSize,
                       borderSize,
                       borderSize,
                       borderSize,
                       cv::BORDER_REFLECT);

    cv::Mat outputImg(img.size(), img.type());

    int realBlockSize
        = blockSize - 2 * (neighborhoodRadius + searchWindowRadius);
    for (int y1 = 0; y1 < paddedImg.size().height; y1 += realBlockSize) {
        for (int x1 = 0; x1 < paddedImg.size().width; x1 += realBlockSize) {
            int x2 = x1 + blockSize;
            int y2 = y1 + blockSize;
            x2 = cv::max(cv::min(x2, paddedImg.size().width), 1);
            y2 = cv::max(cv::min(y2, paddedImg.size().height), 1);
            int x3 = x1;
            int y3 = y1;
            int x4 = cv::min(x1 + realBlockSize, img.size().width);
            int y4 = cv::min(y1 + realBlockSize, img.size().height);

            if ((x4 > x3) && (y4 > y3)) {
                cv::Mat imgBlock = paddedImg(cv::Range(y1, y2),
                                             cv::Range(x1, x2)).clone();

                cv::Mat V = image2vectors_single(imgBlock,
                                                 neighborhoodRadius,
                                                 nThreads);

                cv::Mat filteredImgBlock
                    = vectors_nlmeans_single(imgBlock,
                                             V,
                                             neighborhoodRadius,
                                             searchWindowRadius,
                                             h,
                                             nThreads);

                assert(y4 - y3 == filteredImgBlock.size().height);
                assert(x4 - x3 == filteredImgBlock.size().width);
                filteredImgBlock.copyTo(outputImg(cv::Range(y3, y4),
                                                  cv::Range(x3, x4)));
            }
        }
    }

    return outputImg;
}
