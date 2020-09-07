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
  ML& model = ML::load(model_path);
  //std::cout<<model<<std::endl;

  auto files = get_files(input);
  for (auto f: files) {
    std::cout<<"File: "<<f<<std::endl;
    auto pair = get_objects(pre, f, cmdl["-v"]);
    auto objects = pair.first;
    auto originalImage = pair.second;

    cv::Mat drawing = cv::Mat::zeros(originalImage.size(), CV_8UC3);
    double good = 0, bad = 0;
    for(size_t i = 0; i < objects.size(); i++) {
      auto label = model.predict(objects[i]);
      //std::cout<<"Label = "<<label<<std::endl;
      auto color = cv::Scalar(0,256,0);
      if(label.compare("good")) {
        ++good;
        color = cv::Scalar(0,0,256);
      } else {
        ++bad;
      }
      auto boundRect = objects[i].get_boundRect();
      std::vector<std::vector<cv::Point>> contours;
      contours.push_back(objects[i].get_contour());
      cv::drawContours(drawing, contours, 0, cv::Scalar(256, 256, 256));
      cv::rectangle(drawing, boundRect.tl(), boundRect.br(), color, 2);
    }

    std::cout<<"Acanthocytes = "<<bad<<"/"<<(bad+good)<<std::endl;
    show_images(originalImage, drawing, "Detection");
    cv::destroyAllWindows();
  }
  
  return EXIT_SUCCESS;
}
