#include <fstream>

#include "lib_oc.h"

ChainHistogram::ChainHistogram(const std::array<double, 8> &_hist) {
  hist = _hist;
}

ChainHistogram::ChainHistogram(const std::vector<cv::Point>& contours) {
  std::vector<unsigned char> chaincode = chain(contours);
  unsigned long total = 0;

  for (auto vec : chaincode) {
    hist[vec] ++;
    total ++;
  }

  if(total > 0) {
    for(size_t i = 0; i < 8; i++) {
      hist[i] /= total;
    }
  }
}

double ChainHistogram::distance(const ChainHistogram &other, const unsigned int p) const {
  double sum = 0.0;
  for(size_t i = 0; i < hist.size(); i++) {
    sum += pow(hist[i] - other.hist[i], p);
  }
  return pow(sum, 1.0/p);
}

std::array<double, 8> ChainHistogram::get_histogram() const{
  return hist;
}

std::ostream& operator<<(std::ostream &strm, const ChainHistogram &o) {
  strm << "[";
  for(size_t i = 0; i < o.hist.size(); ++i) {
    strm << std::fixed << std:: setprecision(2) << o.hist[i];
    if (i != o.hist.size() - 1) {
      strm << ", ";
    }
  }
  strm << "]";
  return strm;
}

KNN::KNN(const unsigned int _k) {
  k = _k;
}

KNN::KNN(const unsigned int _k, const std::vector<std::pair<std::string, ChainHistogram>> &_instances){
  k = _k;
  instances = _instances;
}

void KNN::learn_class(const std::string &label, const std::vector<Object> &objects) {
  for(auto o: objects) {
    instances.push_back(std::pair(label, ChainHistogram(o.get_contour())));
  }
}

std::ostream& operator<<(std::ostream &strm, const KNN &o) {
  strm << "KNN: {'k':"<<o.k<<"', instances:['"<<std::endl;
  for(auto i: o.instances) {
    strm << "{'label':"<<i.first<<", 'histogram':"<<i.second<<"},"<<std::endl;
  }
  strm << "]}";
  return strm;
}

std::string most_frequent(const std::vector<std::string> &votes) { 
  // Insert all elements in hash. 
  std::unordered_map<std::string, unsigned int> hash; 
  for (size_t i = 0; i < votes.size(); i++) {
    hash[votes[i]]++;
  }

  // find the max frequency 
  unsigned int max_count = 0; 
  std::string rv = votes[0]; 
  for (auto i : hash) { 
    if (max_count < i.second) { 
      rv = i.first; 
      max_count = i.second; 
    } 
  } 

  return rv; 
}

std::string KNN::predict(const Object &object) const {
  std::vector<std::string> votes;
  std::vector<std::pair<double, std::string>> distances;

  auto hist = ChainHistogram(object.get_contour());

  for(unsigned int i = 0; i < instances.size(); i++){
    distances.push_back(std::pair(hist.distance(instances[i].second), instances[i].first));
  }

  sort(distances.begin(), distances.end());

  /*std::cout<<"Sorted Distances:"<<std::endl;
  for(auto d: distances) {
    std::cout<<d.first<<", "<<d.second<<std::endl;
  }*/

  for(unsigned int i = 0; i < k; i++) {
    votes.push_back(distances[i].second);
  }

  return most_frequent(votes);
}

void KNN::store(const KNN &model, const fs::path &path){
  json j;
  j["k"] = model.k;
  json instances;
  for(auto i: model.instances) {
    json instance;
    instance["label"] = i.first;
    json histogram;
    for(auto h: i.second.get_histogram()) {
      histogram.push_back(h);
    }
    instance["histogram"] = histogram;
    instances.push_back(instance);
  }
  j["instances"] = instances;

  //std::cout<<j.dump(2)<<std::endl;
  std::ofstream o(path);
  o << std::setw(2) << j << std::endl;
}

KNN KNN::load(const fs::path &path) {
  std::ifstream i(path);
  json j;
  i >> j;

  std::vector<std::pair<std::string, ChainHistogram>> instances;

  for(auto i: j["instances"]) {
    std::array<double, 8> hist;
    for(size_t j = 0; j < i["histogram"].size(); j++) {
      hist[j] = i["histogram"][j];
    }
    auto histogram = ChainHistogram(hist);
    instances.push_back(std::pair(i["label"], histogram));
  }

  return KNN(j["k"], instances);
}
