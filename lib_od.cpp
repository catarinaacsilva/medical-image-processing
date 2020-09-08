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

void imfill(cv::Mat& src, cv::Mat& dst) {
  cv::Point seed;
  bool done = false;
  // Search for the first black pixel in a BW image
  for (auto i = 0; i < src.size().width && !done; i++) {
    for(auto j = 0; j < src.size().height && !done; j++) {
      if(src.at<uchar>(i,j) == 0) {
        // The first black pixel will be select as the seed for the imfill
        seed = cv::Point(i,j);
        done = true;
      }
    }
  }
  imfill(src, dst, seed);
}

void imfill(cv::Mat& src, cv::Mat& dst, cv::Point& seed) {
  cv::Mat edges_neg = src.clone();
  cv::floodFill(edges_neg, seed, CV_RGB(255,255,255));
  bitwise_not(edges_neg, edges_neg);
  dst = (edges_neg | src);
}

cv::Mat watershed(cv::Mat &src, cv::Mat &smooth){
  // Create a kernel that we will use to sharpen our image
  cv::Mat kernel = (cv::Mat_<float>(3,3) <<
  1,  1, 1,
  1, -8, 1,
  1,  1, 1);
  // an approximation of second derivative, a quite strong kernel

  cv::Mat imgLaplacian;
  cv::filter2D(src, imgLaplacian, CV_32F, kernel);
  cv::Mat sharp;
  src.convertTo(sharp, CV_32F);
  cv::Mat imgResult = sharp - imgLaplacian;
  
  // convert back to 8bits gray scale
  imgResult.convertTo(imgResult, CV_8UC3);
  show_image(imgResult, "Stuff...");
  
  // Perform the distance transform algorithm
  cv::Mat dist;
  cv::distanceTransform(smooth, dist, cv::DIST_L2, 3);
      
  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
  show_image(dist, "Distance Transform Image");
  
  // Threshold to obtain the peaks
  // This will be the markers for the foreground objects
  cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
  
  // Dilate a bit the dist image
  cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
  cv::dilate(dist, dist, kernel1);

  // Create the CV_8U version of the distance image
  // It is needed for findContours()
  cv::Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U);

  // Find total markers
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Create the marker image for the watershed algorithm
  cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);

  // Perform the watershed algorithm
  std::cout<<"SRC = "<<imgResult.type()<<"("<<CV_8UC3<<"/"<<CV_32F<<")"<<" DST = "<<markers.type()<<std::endl;
  cv::watershed(imgResult, markers);

  cv::Mat mark;
  markers.convertTo(mark, CV_8U);
  bitwise_not(mark, mark);
  show_image(mark, "Markers");
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark

  return markers;
}

void morphological_reconstruction(cv::Mat& in, cv::Mat& mask, cv::Mat& kernel, cv::Mat& out) {
  cv::Mat img_rec = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1),
  img_dilate = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1);
  mask.copyTo(img_rec);
  bool eq = false;
  do {
    img_rec.copyTo(out(cv::Rect(0, 0, img_rec.size().width, img_rec.size().height)));
    cv::morphologyEx(out, img_dilate, cv::MORPH_DILATE, kernel);
    cv::min(in, img_dilate, img_rec);
    cv::Mat diff = img_rec != out;
    eq = cv::countNonZero(diff) == 0;
  } while(!eq);
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
  size_t width = im0.size().width + im1.size().width,
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

  if(originalImage.empty()) {
    // NOT SUCCESSFUL : the data attribute is empty
    std::cerr << "Image "<<path<<" could not be open..." << std::endl;
    exit(EXIT_FAILURE);
  }

  // remove alpha channel
  if(originalImage.channels() > 3) {
    cv::cvtColor(originalImage, originalImage, cv::COLOR_RGBA2RGB);
  }

  cv::Mat bw;
  if(originalImage.channels() > 1) {
    // Convert to a single-channel, intensity image
    cv::cvtColor(originalImage, bw, cv::COLOR_BGR2GRAY, 1);
  }

  if(verbose) {
    show_image(bw, "Original image");
  }

  cv::Mat smooth_image;
  medianBlur(bw, smooth_image, 9);

  if(verbose) {
    show_image(smooth_image, "Averaging Filter 9 x 9 - 1 Iter");
  }

  // Binary image
  cv::Mat binary_image;
  unsigned int high_thresh = (unsigned int)cv::threshold(smooth_image, binary_image, 0, 255, cv::THRESH_OTSU),
  low_thresh = 0;

  cv::bitwise_not(binary_image, binary_image);
  if(verbose) {
    show_image(binary_image, "Threshold Image");
  }

  //vários kernel para teste!
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
  auto kernel_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
  auto kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(23, 23));
  auto kernel_close_2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  auto kernel_rec = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
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
    show_image(binary_image, "Fill Holes (imfill)");
  }

  /*binary_image = watershed(binary_image);
  if(verbose){
    show_image(binary_image, "watershed");
  }*/

  cv::morphologyEx(binary_image, smooth_image, cv::MORPH_ERODE, kernel_erode);
  if(verbose) {
    show_image(smooth_image, "Erosion");
  }

  morphological_reconstruction(binary_image, smooth_image, kernel_rec, smooth_image);
  if(verbose) {
    show_image(smooth_image, "Morphological Reconstruction");
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
        show_image(edges, "Morphological Gradient");
      }
    break;
    case 2:{
      cv::Mat markers = watershed(originalImage, smooth_image);
      double min, max;
      std::cout<<"("<<min<<"; "<<max<<")"<<std::endl;
      cv::minMaxLoc(markers, &min, &max);
      std::vector<cv::Vec3b> colors;
      for (size_t i = 0; i < max; i++)
      {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
      }
      // Create the result image
      cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
      // Fill labeled objects with random colors
      for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
          int index = markers.at<int>(i,j);
          if (index > 0 && index <= static_cast<int>(max)) {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
      }
      if(verbose) {
        show_image(edges, "Watershed");
      }
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

  return std::pair(objects, bw);
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

