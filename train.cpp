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
    <<"  -k, the number of nearest neighbors       [default = 1]"<<std::endl
    <<"  -d, Minkowski distance of order p         [default = 2]"<<std::endl
    <<"  -i, the input folder with images to train [default = './resources/train/']"<<std::endl
    <<"  -o, the output model                      [default = './resources/model/model.json']"<<std::endl
    <<"  -v, verbose"<<std::endl
    <<"  -h, this help message"<<std::endl;
}

int main(const int argc, const char** argv) {
  argh::parser cmdl;
  cmdl.add_params({"-p", "-d", "-k", "-i", "-o"}); // batch pre-register multiple params: name + value
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
  std::cout<<"K = "<<k<<std::endl;

  unsigned int d = 2;
  if (cmdl("-d")) {
    std::string value;
    cmdl("-d") >> value;
    d = std::atoi(value.c_str());
  }
  std::cout<<"D = "<<d<<std::endl;

  auto model = KNN(k, d);
  auto classes = get_directories(input);
  for (auto c: classes) {
    std::cout <<"Learn the following class: "<< c.filename().c_str() << std::endl;
    auto files = get_files(c);
    std::vector<Object> instances;
    for (auto f: files) {
      std::cout<<"File: "<<f<<std::endl;
      auto objects = (get_objects(pre, f, cmdl["-v"])).first;
      auto object = *std::max_element(std::begin(objects), std::end(objects));
      std::cout<<"Object = "<<object<<std::endl;
      instances.push_back(object);
    }
    model.learn_class(c.filename().c_str(), instances);
  }

  std::cout<<model<<std::endl;

  std::string output = "./resources/model/model.json";
  if (cmdl("-o")) {
    cmdl("-o") >> output;
  }
  std::cout<<"Output: "<<output<<std::endl;
  KNN::store(model, output);

  //auto model2 = KNN::load(output);
  //std::cout<<model2<<std::endl;
  return EXIT_SUCCESS;
}
