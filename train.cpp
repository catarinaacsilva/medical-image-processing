#include <iostream>
#include <string>

#include "argh.h"
#include "lib_od.h"
#include "lib_oc.h"
#include "lib_fs.h"

void print_help() {
  std::cout<<"Program used to train a kNN model to identify anomalous blood cells."<<std::endl
    <<"usage: train [-p] [-k] [-i] [-o] [-h]"<<std::endl<<std::endl
    <<"Parameters:"<<std::endl
    <<"  -p, the preprocessig method               [default = 0]"<<std::endl
    <<"  -m, ML model (0 - ARFF; 1 - KNN; 2 - LR)  [default = 0]"<<std::endl
    <<"  -k, the number of nearest neighbors       [default = 1]"<<std::endl
    <<"  -d, Minkowski distance of order p         [default = 2]"<<std::endl
    <<"  -i, the input folder with images to train [default = './resources/train/']"<<std::endl
    <<"  -o, the output model                      [default = './resources/model/model.json']"<<std::endl
    <<"  -v, verbose"<<std::endl
    <<"  -h, this help message"<<std::endl;
}

int main(const int argc, const char** argv) {
  argh::parser cmdl;
  cmdl.add_params({"-p", "-m", "-d", "-k", "-i", "-o"}); // batch pre-register multiple params: name + value
  cmdl.parse(argc, argv);

  if (cmdl["-h"]) {
    print_help();
    return EXIT_SUCCESS;
  }

  std::string input = "./resources/train/";
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

  unsigned int k = 1;
  if (cmdl("-k")) {
    std::string value;
    cmdl("-k") >> value;
    k = std::atoi(value.c_str());
  }

  unsigned int d = 2;
  if (cmdl("-d")) {
    std::string value;
    cmdl("-d") >> value;
    d = std::atoi(value.c_str());
  }

  auto classes = get_directories(input);
  std::vector<std::pair<std::string, Features>> instances;
  for (auto c: classes) {
    const std::string label = c.filename().u8string();
    std::cout << "Loading the following class: " << label << std::endl;
    auto files = get_files(c);
    for (auto f: files) {
      std::cout<<"File: "<<f<<std::endl;
      auto objects = (get_objects(pre, f, cmdl["-v"])).first;
      if (objects.size() > 0) {
        auto object = *std::max_element(std::begin(objects), std::end(objects));
        std::cout<<"Object = "<<object<<std::endl;
        std::vector<cv::Point> contour = object.get_contour();
        if(contour.size() > 0) {
          auto features = Features(contour);
          std::cout<<features<< std::endl;
          std::pair<std::string, Features> p = std::make_pair(label, features);
          instances.push_back(p);
        }
      }
    }
  }

  unsigned int m = 0;
  if (cmdl("-m")) {
    std::string value;
    cmdl("-m") >> value;
    m = std::atoi(value.c_str());
  }

  std::string output = "./resources/model/model.json";
  if (cmdl("-o")) {
    cmdl("-o") >> output;
  }
  std::cout<<"Output: "<<output<<std::endl;
  
  switch(m) {
    case 0:
      // output ARFF file...
      std::cout<<"@RELATION mi"<<std::endl
      <<"@ATTRIBUTE h0           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h1           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h2           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h3           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h4           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h5           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h6           NUMERIC"<<std::endl
      <<"@ATTRIBUTE h7           NUMERIC"<<std::endl
      <<"@ATTRIBUTE circularity  NUMERIC"<<std::endl
      <<"@ATTRIBUTE convex       NUMERIC"<<std::endl
      <<"@ATTRIBUTE aspect_ratio NUMERIC"<<std::endl
      <<"@ATTRIBUTE extent       NUMERIC"<<std::endl
      <<"@ATTRIBUTE class        {good, bad}"<<std::endl
      <<"@DATA"<<std::endl;
      for(auto i: instances) {
        auto fe = i.second.get_features();
        for (size_t j = 1; j < fe.size(); j++){
          std::cout<< fe[j] << ", ";
        }
        std::cout << i.first << std::endl;
      }
      break;
    case 1:{
      auto model = KNN(k, d);
      std::cout << "Model learning..." << std::endl;
      model.learn(instances);
      model.store(output);
      std::cout<<model<<std::endl;
      break;}
    case 2:{
      auto model = LR();
      std::cout << "Model learning..." << std::endl;
      model.learn(instances);
      model.store(output);
      std::cout<<model<<std::endl;
      break;}
  }
  
  return EXIT_SUCCESS;
}
