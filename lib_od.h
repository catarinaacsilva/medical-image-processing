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

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"

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

void show_images(const cv::Mat&, const cv::Mat&, const std::string&);

std::vector<unsigned char> chain(const std::vector<cv::Point>&);

std::pair<std::vector<Object>, cv::Mat> get_objects(const unsigned int, const std::string&, const bool verbose=false);

/**
 * A simple implementation of the imfill image of Matlab.
 * According to the documentation, the function fills holes in the binary image src.
 * A hole is a set of background pixels that cannot be reached by filling in the background from the edge of the image.
 *
 * @param src source image
 * @param dst destination image
 * @param seed seed point to the backgound (by default [0,0])
 */
void imfill(cv::Mat& src, cv::Mat& dst, cv::Point& seed);

/**
 * A simple implementation of the imfill image of Matlab.
 * According to the documentation, the function fills holes in the binary image src.
 * A hole is a set of background pixels that cannot be reached by filling in the background from the edge of the image.
 * This function does not need a seed point, it will find the first black point on the background.
 *
 * @param src source image
 * @param dst destination image
 * @param seed seed point to the backgound (by default [0,0])
 */
void imfill(cv::Mat& src, cv::Mat& dst);

/**
 * A simple implementation of morphological reconstruction, based on this code:
 * https://stackoverflow.com/questions/29104091/morphological-reconstruction-in-opencv
 *
 * @param in input image
 * @param mask mask that identifies the seeds for the objects to keep
 * @param kernel kernel used in the morphological operations
 * @param out output image
 */
void morphological_reconstruction(cv::Mat& in, cv::Mat& mask, cv::Mat& kernel, cv::Mat& out);

#endif
