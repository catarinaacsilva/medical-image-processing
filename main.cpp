#include <iostream>

#include "argh.h"
#include "lib_od.h"
#include "lib_oc.h"
#include "lib_fs.h"

void print_help() {
  std::cout<<"Program used to identify anomalous blood cells."<<std::endl
    <<"usage: main [-p] [-k] [-i] [-o] [-h]"<<std::endl<<std::endl
    <<"Parameters:"<<std::endl
    <<"  -p, the preprocessig method            [default = 0]"<<std::endl
    <<"  -m, the classification model           [default = './resources/model/model.json']"<<std::endl
    <<"  -i, the folder with images to classify [default = './resources/test/']"<<std::endl
    <<"  -v, verbose"<<std::endl
    <<"  -h, this help message"<<std::endl;
}

int main(const int argc, const char** argv) {
  argh::parser cmdl;
  cmdl.add_params({ "-p", "-m", "-i" }); // batch pre-register multiple params: name + value
  cmdl.parse(argc, argv);

  if (cmdl["-h"]) {
    print_help();
    return EXIT_SUCCESS;
  }

  std::string input = "./resources/test/";
  if (cmdl("-i")) {
    cmdl("-i") >> input;
  }
  std::cout<<"Input: "<<input<<std::endl;

  unsigned int pre = 0;
  if (cmdl("-p")) {
    std::string value;
    cmdl("-p") >> value;
    pre = std::atoi(value.c_str());
  }
  std::cout<<"Preprocessig = "<<pre<<std::endl;

  std::string model_path = "./resources/model/model.json";
  if (cmdl("-m")) {
    cmdl("-m") >> model_path;
  }
  std::cout<<"Model: "<<model_path<<std::endl;
  auto model = KNN::load(model_path);
  std::cout<<model<<std::endl;

  auto files = get_files(input);
  for (auto f: files) {
    std::cout<<"File: "<<f<<std::endl;
    auto tuple = get_objects(pre, f, cmdl["-v"]);
    auto objects = std::get<0>(tuple);
    auto originalImage = std::get<1>(tuple);
    auto edges = std::get<2>(tuple);

    cv::Mat drawing = cv::Mat::zeros(edges.size(), CV_8UC3);
    for(size_t i = 0; i < objects.size(); i++) {
      auto label = model.predict(objects[i]);
      std::cout<<"Label = "<<label<<std::endl;
      auto color = cv::Scalar(0,256,0);
      if(label.compare("good")) {
        color = cv::Scalar(0,0,256);
      }
      auto boundRect = objects[i].get_boundRect();
      std::vector<std::vector<cv::Point>> contours;
      contours.push_back(objects[i].get_contour());
      cv::drawContours(drawing, contours, 0, cv::Scalar(256, 256, 256));
      cv::rectangle(drawing, boundRect.tl(), boundRect.br(), color, 2);
    }

    show_image(drawing, "Detection");
    cv::destroyAllWindows();
  }


  /*// Smooth image --> morphological operations - opening
    Mat smooth_image_op;
    namedWindow("opening", WINDOW_AUTOSIZE);
    imshow( "opening", morph_opening(originalImage));
    waitKey( 0 );

  //grad morph
  Mat element = structuring_element();
  namedWindow("grad", WINDOW_AUTOSIZE);
  imshow("grad", morph_grad(smooth_image_2));
  waitKey(0);*/

  return EXIT_SUCCESS;
}
