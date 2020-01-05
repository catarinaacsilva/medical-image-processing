/**
 * @file lib_od.h
 * @brief Object Detection library
 *
 * Functions used to detect objects in medical images.
 * The code several filters and edge extraction methods to detect objects.
 *
 * @author $Author: Catarina Silva $
 * @version $Revision: 1.0 $
 * @date $Date: 2020/01/05 $
 */

#ifndef OD_H
#define OD_H

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <tuple>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"

namespace fs = std::filesystem;

class Object {
  private:
    cv::Rect boundRect;
    std::vector<cv::Point> contour;
    double area;
    friend std::ostream& operator<<(std::ostream&, const Object&);

  public:
    Object(std::vector<cv::Point>&);
    bool operator<(const Object&) const;
    std::vector<cv::Point> get_contour() const;
    cv::Rect get_boundRect() const;
};

void show_image(const cv::Mat&, const std::string&);

std::vector<unsigned char> chain(const std::vector<cv::Point>&);

std::tuple<std::vector<Object>, cv::Mat, cv::Mat>
get_objects(const unsigned int, const fs::path&, const bool verbose=false);

#endif
