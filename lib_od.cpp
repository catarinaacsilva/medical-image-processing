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


// PRIMEIRA IMPLEMENTAÇÃO QUE DIFERE DA SEGUNDA POR COLOCAR MARKERMASK= EDGES
//markerMask -> image gray  
//edges -> findContours
cv::Mat watershed_aux(cv::Mat markerMask, cv::Mat edges){
  cv::Mat markers(markerMask.size(), CV_32S);
  cv::Mat original_image;
  cvtColor(markerMask, original_image, cv::COLOR_GRAY2BGR);
  watershed(original_image, markers);
  cv::Mat wshed(markers.size(), CV_8UC3);
  wshed = wshed*0.5+original_image*0.5;
  return wshed;
}
/*

// SEGUNDA IMPLEMENTAÇÃO QUE SUPOSTAMENTE ESTA MAIS CORRETA MAS DA ERROS
/*
WATERSHED:
  - original_image: gray imagem that results of canny but with 3 channels
  - markers: results of markersMask (findContours)
  - markermask: image result of canny (gray) - 1 channel = edges of findContours
*/
/*cv::Mat watershed_aux(cv::Mat markerMask, cv::Mat edges){
  cv::Mat original_image = cv::Mat::zeros(cv::Size(markerMask.size().width, markerMask.size().height), CV_8UC3);
  // Convert the image with 1 channel to 2 channels: result of canny to other image. This image will be the input of watershed function
  cvtColor(original_image, markerMask, cv::COLOR_GRAY2BGR);
  //It's necessary to obtain the result of findContours but it isn't implemented here.
  markerMask = edges;
  cv::Mat markers(markerMask.size(), CV_32S);
  watershed(original_image, markers);
  cv::Mat wshed(markers.size(), CV_8UC3);
  wshed = wshed*0.5 + original_image*0.5;
  return wshed;
}*/

void morphological_reconstruction(cv::Mat& in, cv::Mat& mask, cv::Mat& kernel, cv::Mat& out) {
  cv::Mat img_rec = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1),
  img_dilate = cv::Mat::zeros(cv::Size(mask.size().width, mask.size().height), CV_8UC1);
  mask.copyTo(img_rec);
  bool eq = false;
  do{
    img_rec.copyTo(out(cv::Rect(0, 0, img_rec.size().width, img_rec.size().height)));
    cv::morphologyEx(out, img_dilate, cv::MORPH_DILATE, kernel);
    cv::min(in, img_dilate, img_rec);
    cv::Mat diff = img_rec != out;
    eq = cv::countNonZero(diff) == 0;
  }while(!eq);
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
    show_image(binary_image, "original");
  }

  /*binary_image = watershed(binary_image);
  if(verbose){
    show_image(binary_image, "watershed");
  }*/

  cv::morphologyEx(binary_image, smooth_image, cv::MORPH_ERODE, kernel_erode);
  if(verbose) {
    show_image(smooth_image, "erosao");
  }

  morphological_reconstruction(binary_image, smooth_image, kernel_rec, smooth_image);
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

  cv::Mat image_gray = edges;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  cv::Mat w = watershed_aux(image_gray, edges);
  

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

