#include "lib_od.h"

uchar encode(const cv::Point &a, const cv::Point &b) {
    uchar up    = (a.y > b.y);
    uchar left  = (a.x > b.x);
    uchar down  = (a.y < b.y);
    uchar right = (a.x < b.x);
    uchar equx  = (a.y == b.y);
    uchar equy  = (a.x == b.x);

    return (up    && equy)  ? 0 : // N
           (up    && right) ? 1 : // NE
           (right && equx)  ? 2 : // E
           (down  && right) ? 3 : // SE
           (down  && equy)  ? 4 : // S
           (left  && down)  ? 5 : // SW
           (left  && equx)  ? 6 : // W
                              7 ; // NW
}

void chain(const std::vector<cv::Point> &contour, std::vector<uchar> &_chain) {
    unsigned int i = 0;
    for (; i<contour.size()-1; i++) {
        _chain.push_back(encode(contour[i],contour[i+1]));
    }
    _chain.push_back(encode(contour[i],contour[0]));
}

Object::Object(cv::Rect &_boundRect, std::vector<cv::Point> &_contours) {
    boundRect = _boundRect;
    contours = _contours;

    std::vector<uchar> chaincode;
    chain(contours, chaincode);
    unsigned long total = 0;
        
    for (auto vec : chaincode) {
        hist[vec] ++;
        total ++;
    }

    for(int i = 0; i < 8; i++) {
        hist[i] /= total;
    }
}

std::ostream& operator<<(std::ostream &strm, const Object &o) {
    strm << "Histogram: [";
    for(unsigned int i = 0; i < o.hist.size(); ++i) {
        strm << std::fixed << std:: setprecision(2) << o.hist[i];
        if (i != o.hist.size() - 1) {
            strm << ", ";
        }
    }
    strm << "]";
    return strm;
}