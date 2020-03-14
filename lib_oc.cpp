#include "lib_oc.h"

#include <fstream>
#include <cmath>

#define _USE_MATH_DEFINES

Features::Features(const std::array<double, 8> &_hist, const double _circularity) {
  hist = _hist;
  circularity = _circularity;
}

Features::Features(const std::vector<cv::Point>& contour) {
  std::vector<unsigned char> chaincode = chain(contour);
  std::fill(std::begin(hist), std::end(hist), 0);

  for (auto vec : chaincode) {
    hist[vec]++;
  }

  if(chaincode.size() > 0) {
    for(size_t i = 0; i < hist.size(); i++) {
      hist[i] /= chaincode.size();
    }
  }

  double area = cv::contourArea(contour),
  perimeter = cv::arcLength(contour, true);

  circularity = (4.0*M_PI*area)/pow(perimeter, 2);
}

double Features::distance(const Features& other, const unsigned int d) const {
  double histogram_distance = 0;
  
  //Chebyshev distance
  if(d == 0) {
    double max = 0;
    for(size_t i = 0; i < hist.size(); i++) {
      double t = fabs(hist[i] - other.hist[i]);
      if(t < max) {
        max = t;
      }
    }
    histogram_distance = max;
  } else {
    double sum = 0.0;
    for(size_t i = 0; i < hist.size(); i++) {
      sum += pow(hist[i] - other.hist[i], d);
    }
    histogram_distance = pow(sum, 1.0/d);
  }

  double circularity_distance = fabs(circularity - other.circularity);

  return (histogram_distance + circularity_distance) / 2.0;
}

std::pair<std::array<double, 8>,double> Features::get_features() const{
  return std::pair(hist, circularity);
}

std::ostream& operator<<(std::ostream &strm, const Features &o) {
  strm << "{'c':"<<o.circularity<<", 'h':[";
  for(size_t i = 0; i < o.hist.size(); ++i) {
    strm << std::fixed << std:: setprecision(2) << o.hist[i];
    if (i != o.hist.size() - 1) {
      strm << ", ";
    }
  }
  strm << "]";
  return strm;
}

KNN::KNN(const unsigned int _k, const unsigned int _d) {
  k = _k;
  d = _d;
}

KNN::KNN(const unsigned int _k, const unsigned int _d,
const std::vector<std::pair<std::string, Features>> &_instances){
  k = _k;
  d = _d;
  instances = _instances;
}

void KNN::learn_class(const std::string &label, const std::vector<Object> &objects) {
  for(auto o: objects) {
    instances.push_back(std::pair(label, Features(o.get_contour())));
  }
}

std::ostream& operator<<(std::ostream &strm, const KNN &o) {
  strm << "KNN: {'k':"<<o.k<<"', 'd':"<<o.d<<", instances:['"<<std::endl;
  for(auto i: o.instances) {
    strm << "{'label':"<<i.first<<", 'features':"<<i.second<<"},"<<std::endl;
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

  auto features = Features(object.get_contour());

  for(unsigned int i = 0; i < instances.size(); i++){
    distances.push_back(std::pair(features.distance(instances[i].second, d), instances[i].first));
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

void KNN::store(const KNN &model, const std::string &path){
  json j;
  j["k"] = model.k;
  j["d"] = model.d;
  json instances;
  for(auto i: model.instances) {
    json instance;
    instance["label"] = i.first;
    auto f = i.second.get_features();
    instance["circularity"] = f.second;
    json histogram;
    for(auto h: f.first) {
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

KNN KNN::load(const std::string &path) {
  std::ifstream i(path);
  json j;
  i >> j;

  std::vector<std::pair<std::string, Features>> instances;

  for(auto i: j["instances"]) {
    std::array<double, 8> hist;
    double circularity = i["circularity"];
    for(size_t j = 0; j < i["histogram"].size(); j++) {
      hist[j] = i["histogram"][j];
    }
    auto features = Features(hist, circularity);
    instances.push_back(std::pair(i["label"], features));
  }

  return KNN(j["k"], j["d"], instances);
}
