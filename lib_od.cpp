#include "lib_od.h"

static cv::Point SEED_ZERO = cv::Point(0,0);

unsigned char encode(const cv::Point &a, const cv::Point &b) {
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

void imfill(cv::Mat& src, cv::Mat& dst, cv::Point& seed = SEED_ZERO) {
  cv::Mat edges_neg = src.clone();
  cv::floodFill(edges_neg, seed, CV_RGB(255,255,255));
  bitwise_not(edges_neg, edges_neg);
  dst = (edges_neg | src);
}

cv::Mat watershed(cv::Mat image_input){
  cv::Mat imageResult = cv::Mat::zeros(cv::Size(image_input.size().width, image_input.size().height), CV_8UC1);
  // Perform the distance transform algorithm
  cv::Mat dist;
  distanceTransform(image_input, dist, cv::DIST_L2, 3);
  normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
  // Threshold to obtain the peaks; This will be the markers for the foreground objects
  threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
  // Dilate a bit the dist image
  cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
  dilate(dist, dist, kernel1);
  // Create the CV_8U version of the distance image; It is needed for findContours()
  cv::Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U);
  // Find total markers
  std::vector<std::vector<cv::Point>> contours;
  findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  // Create the marker image for the watershed algorithm
  cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
  // Perform the watershed algorithm
  watershed(imageResult, markers);
  return imageResult;
}  

cv::Mat morphological_reconstruction(cv::Mat image, cv::Mat mask, cv::Mat kernel){
  cv::Mat imageRec = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1), 
  imageResult = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1), 
  imageDilate = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1);;
  mask.copyTo(imageRec(cv::Rect(0, 0, mask.size().width, mask.size().height)));
  bool eq = false;
  do{
    imageRec.copyTo(imageResult(cv::Rect(0, 0, imageRec.size().width, imageRec.size().height)));
    cv::morphologyEx(imageResult, imageDilate, cv::MORPH_DILATE, kernel);
    cv::min(image, imageDilate, imageRec);
    cv::Mat diff = imageRec != imageResult;
    eq = cv::countNonZero(diff) == 0;
  }while(!eq);
  return imageResult;
}

std::vector<unsigned char> chain(const std::vector<cv::Point> &contour) {
  std::vector<unsigned char> rv;
  size_t i = 0;
  for (; i<contour.size()-1; i++) {
    rv.push_back(encode(contour[i],contour[i+1]));
  }
  rv.push_back(encode(contour[i],contour[0]));
  return rv;
}

void show_images(const cv::Mat& im0, const cv::Mat& im1, const std::string &name) {
  unsigned int width = im0.size().width + im1.size().width,
  height = std::max(im0.size().height, im1.size().height);
  cv::Mat canvas = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
  
  if (im0.channels() == 1) {
    cv::Mat temp = cv::Mat::zeros(im0.size(), CV_8UC3);
    cv::cvtColor(im0, temp, cv::COLOR_GRAY2BGR);
    temp.copyTo(canvas(cv::Rect(0, 0, im0.size().width, im0.size().height)));
  } else {
    im0.copyTo(canvas(cv::Rect(0, 0, im0.size().width, im0.size().height)));
  }

  if (im1.channels() == 1) {
    cv::Mat temp = cv::Mat::zeros(im1.size(), CV_8UC3);
    cv::cvtColor(im1, temp, cv::COLOR_GRAY2BGR);
    temp.copyTo(canvas(cv::Rect(im0.size().width, 0, im1.size().width, im1.size().height)));
  } else {
    im1.copyTo(canvas(cv::Rect(im0.size().width, 0, im1.size().width, im1.size().height)));
  }

  show_image(canvas, name);
}

void show_image(const cv::Mat &image, const std::string &name) {
  // Create window
  cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
  // Display image
  cv::imshow(name, image);
  // Wait for a click
  cv::waitKey(0);
}

std::pair<std::vector<Object>,cv::Mat>
get_objects(const unsigned int pre, const std::string &path, const bool verbose) {
  auto originalImage = cv::imread(path, cv::IMREAD_UNCHANGED);

  if(originalImage.empty())
  {
    // NOT SUCCESSFUL : the data attribute is empty
    std::cerr << "Image "<<path<<" could not be open..." << std::endl;
    exit(EXIT_FAILURE);
  }

  if(originalImage.channels() > 1)
  {
    // Convert to a single-channel, intensity image
    cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2GRAY, 1);
  }

  if(verbose) {
    show_image(originalImage, "Original image");
  }

  cv::Mat smooth_image;
  medianBlur(originalImage, smooth_image, 9);

  if(verbose) {
    show_image(smooth_image, "Averaging Filter 9 x 9 - 1 Iter");
  }

  // Binary image
  cv::Mat binary_image;
  unsigned int high_thresh = (unsigned int)cv::threshold(smooth_image, binary_image, 0, 255, cv::THRESH_OTSU),
  low_thresh = 0;

  cv::bitwise_not(binary_image, binary_image);
  if(verbose) {
    show_image(binary_image, "Threshold Otsu");
  }

  //vários kernel para teste!
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
  auto kernel_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
  auto kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(23, 23));
  auto kernel_close_2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  auto kernel_rec = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(23, 23));
  auto kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(45, 45));
  //cv::morphologyEx(binary_image, smooth_image, cv::MORPH_OPEN, kernel);
  //cv::morphologyEx(smooth_image, smooth_image, cv::MORPH_CLOSE, kernel);
  
  //RECONSTRUÇÃO MORFOLÓGICA
  /*cv::morphologyEx(binary_image, binary_image, cv::MORPH_CLOSE, kernel_close);
  if(verbose) {
    show_image(binary_image, "original");
  }

  cv::morphologyEx(binary_image, binary_image, cv::MORPH_OPEN, kernel_open);
  if(verbose) {
    show_image(binary_image, "open");
  }*/

  // fill some parts of original image
  imfill(binary_image, binary_image); 
  if(verbose){
    show_image(binary_image, "original");
  }


  cv::morphologyEx(binary_image, smooth_image, cv::MORPH_ERODE, kernel_erode);
  if(verbose) {
    show_image(smooth_image, "erosao");
  }

  smooth_image = morphological_reconstruction(binary_image, smooth_image, kernel_rec);
  if(verbose) {
    show_image(smooth_image, "morph rec");
  }

  /*cv::morphologyEx(smooth_image, smooth_image, cv::MORPH_CLOSE, kernel_close_2);
  if(verbose) {
    show_image(smooth_image, "close");
  }*/


  // After binarization is necessary reduce noise, again
  /*medianBlur(smooth_image, smooth_image, 9);

  if(verbose) {
    show_image(smooth_image, "Averaging Filter 9 x 9 - 2 Iter");
  }*/

  cv::Mat edges;
  switch(pre){
    case 0:
      low_thresh = high_thresh / 2;
      cv::Canny(smooth_image, edges, low_thresh, high_thresh);
      if(verbose) {
        show_image(edges, "Canny");
      }
    break;
    case 1:
      cv::morphologyEx(smooth_image, edges, cv::MORPH_GRADIENT, kernel);
      if(verbose) {
        show_image(edges, "Morph GRADIENT");
      }
    break;
    default:
      low_thresh = high_thresh / 2;
      cv::Canny(smooth_image, edges, low_thresh, high_thresh);
      if(verbose) {
        show_image(edges, "Canny");
      }
    break;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::vector<Object> objects;
  for (size_t i = 0; i < contours.size(); i++) {
    objects.push_back(Object(contours[i]));
    //std::cout << objects[objects.size()-1] << std::endl;
  }

  if(verbose) {
    cv::destroyAllWindows();
  }

  return std::pair(objects, originalImage);
}

Object::Object(std::vector<cv::Point> &_contour) {
  contour = _contour;
  boundRect = cv::boundingRect(_contour);
  area = cv::contourArea(_contour);
}

std::ostream& operator<<(std::ostream &strm, const Object &o) {
  strm << "BB: ["<<o.boundRect.width<<", "<<o.boundRect.height<<"] Area: "<<o.area;
  return strm;
}

bool Object::operator<(const Object &other) const {
  return area < other.area;
}

std::vector<cv::Point> Object::get_contour() const {
  return contour;
}

cv::Rect Object::get_boundRect() const {
  return boundRect;
}

