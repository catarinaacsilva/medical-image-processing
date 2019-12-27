#ifndef OD_H
#define OD_H

#include <iostream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"

class Object {
    private:
    cv::Rect boundRect;
    std::vector<cv::Point> contours;
    std::array<double, 8> hist;
    friend std::ostream& operator<<(std::ostream&, const Object&);

    public:
    Object(cv::Rect&, std::vector<cv::Point>&);
};

void chain(const std::vector<cv::Point> &contour, std::vector<uchar> &_chain);

#endif
