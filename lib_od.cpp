#include "lib_od.h"

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

std::vector<unsigned char> chain(const std::vector<cv::Point> &contour) {
  std::vector<unsigned char> rv;
  size_t i = 0;
  for (; i<contour.size()-1; i++) {
    rv.push_back(encode(contour[i],contour[i+1]));
  }
  rv.push_back(encode(contour[i],contour[0]));
  return rv;
}

//create structuring element
cv::Mat structuring_element(unsigned int size) {
  //return  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4*size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
  return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
}

//reduce noise with morphologic operartions
cv::Mat morph_opening(const cv::Mat &imageInput, const cv::Mat kernel){
  cv::Mat image_dest;
  cv::morphologyEx(imageInput, image_dest, cv::MORPH_OPEN, kernel); // 1 iteração
  return image_dest;
}

cv::Mat morph_grad(const cv::Mat imageInput, const cv::Mat kernel){
  cv::Mat image_dest;
  cv::morphologyEx(imageInput, image_dest, cv::MORPH_GRADIENT, kernel);
  return image_dest;
}

void show_image(const cv::Mat &image, const std::string &name) {
  // Create window
  cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
  // Display image
  cv::imshow(name, image);
  // Wait for a click
  cv::waitKey(0);
}

/// Method used in the paper
cv::Mat preprocessing_0(const cv::Mat &originalImage, const bool verbose) {
  // Smooth image eliminar ruido
  cv::Mat smooth_image;
  medianBlur(originalImage, smooth_image, 9);

  if(verbose) {
    show_image(smooth_image, "Averaging Filter 9 x 9 - 1 Iter");
  }

  // Binary image
  cv::Mat binary_image;
  unsigned int high_thresh = (unsigned int)cv::threshold(smooth_image, binary_image, 0, 255, cv::THRESH_OTSU);

  if(verbose) {
    show_image(binary_image, "THRESH OTSU");
  }

  // After binarization is necessary reduce noise, again
  medianBlur(binary_image, smooth_image, 9);

  if(verbose) {
    show_image(smooth_image, "Averaging Filter 9 x 9 - 2 Iter");
  }

  // Use canny edge detector
  cv::Mat edges;
  unsigned int low_thresh = high_thresh / 2;
  cv::Canny(smooth_image, edges, low_thresh, high_thresh);

  if(verbose) {
    show_image(edges, "Canny");
  }

  return edges;
}

/// Method based on morphological operations
cv::Mat preprocessing_1(const cv::Mat &originalImage, const bool verbose) {
  cv::Mat edges;
  return edges;
}

std::tuple<std::vector<Object>,cv::Mat, cv::Mat>
get_objects(const unsigned int pre, const fs::path& path, const bool verbose) {
  std::vector<Object> objects;

  auto originalImage = cv::imread(path, cv::IMREAD_UNCHANGED);

  if(originalImage.empty())
  {
    // NOT SUCCESSFUL : the data attribute is empty
    std::cerr << "Image "<<path<<" could not be open..." << std::endl;
    exit(EXIT_FAILURE);
  }

  if(originalImage.channels() > 1 )
  {
    // Convert to a single-channel, intensity image
    cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2GRAY, 1);
  }

  if(verbose) {
    show_image(originalImage, "Original image");
  }

  cv::Mat edges;
  switch(pre){
    case 0: edges = preprocessing_0(originalImage, verbose); break;
    case 1: edges = preprocessing_1(originalImage, verbose); break;
    default: edges = preprocessing_0(originalImage, verbose); break;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  for (size_t i = 0; i < contours.size(); i++) {
    objects.push_back(Object(contours[i]));
    //std::cout << objects[objects.size()-1] << std::endl;
  }

  if(verbose) {
    cv::destroyAllWindows();
  }

  return std::tuple(objects, originalImage, edges);
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