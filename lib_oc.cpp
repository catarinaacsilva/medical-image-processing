#include "lib_oc.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <cmath>

#define _USE_MATH_DEFINES

double magnitude_vector(const std::vector<double> vector) {
  double res = 0.0;

  for(auto v: vector) {
    res += pow(v,2);
  }

  return sqrt(res);
}

double sigmoid(const double x) {
  return 1.0/(1.0+exp(-x));
}

double dot_product(const std::vector<double> a0, const std::vector<double> a1) {

}

Features::Features(const std::array<double, 8> &_hist, const double _circularity,
const bool _convex, const double _aspect_ratio, const double _extent) {
  hist = _hist;
  circularity = _circularity;
  convex = _convex;
  aspect_ratio = _aspect_ratio;
  extent = _extent;
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

  auto bb = cv::boundingRect(contour);
  double area = cv::contourArea(contour),
  perimeter = cv::arcLength(contour, true),
  width = bb.width, height = bb.height;

  circularity = (4.0*M_PI*area)/pow(perimeter, 2);
  convex = cv::isContourConvex(contour);
  aspect_ratio = width / height;
  extent = area / (width * height);
}

double Features::distance(const Features& other, const unsigned int d) const {
  auto array0 = get_features(), array1 = other.get_features();
  
  double distance = 0;
  
  //Chebyshev distance
  if(d == 0) {
    double max = 0;
    for(size_t i = 0; i < array0.size(); i++) {
      double t = fabs(array0[i] - array1[i]);
      if(t < max) {
        max = t;
      }
    }
    distance = max;
  } else {
    double sum = 0.0;
    for(size_t i = 0; i < array0.size(); i++) {
      sum += pow(array0[i] - array1[i], d);
    }
    distance = pow(sum, 1.0/d);
  }

  return distance;
}

std::vector<double> Features::get_features() const {
  std::vector<double> res;
  res.push_back(1.0);
  for(auto h: hist) {
    res.push_back(h);
  }
  //std::copy(hist.begin(), hist.end(), res.begin()+1);
  res.push_back(circularity);
  res.push_back(convex);
  res.push_back(aspect_ratio);
  res.push_back(extent); 
  return res;
}

std::array<double, 8> Features::get_histogram() const {
  return hist;
}

double Features::get_circularity() const {
  return circularity;
}

bool Features::get_convex() const {
  return convex;
}

double Features::get_aspect_ratio() const {
  return aspect_ratio;
}

double Features::get_extent() const {
  return extent;
}

std::ostream& operator<<(std::ostream &strm, const Features &o) {
  strm << "{'convex':"<<o.convex<<", 'circularity':"<<o.circularity<<
  ", 'aspect_ratio':"<<o.aspect_ratio<<", 'extent':"<<o.extent<<", 'h':[";
  for(size_t i = 0; i < o.hist.size(); ++i) {
    strm << std::fixed << std:: setprecision(2) << o.hist[i];
    if (i != o.hist.size() - 1) {
      strm << ", ";
    }
  }
  strm << "]}";
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

void KNN::learn(const std::vector<std::pair<std::string, Features>> &inst) {
  for(auto i: inst) {
    instances.push_back(i);
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

  auto feature = Features(object.get_contour());

  for(unsigned int i = 0; i < instances.size(); i++){
    distances.push_back(std::pair(feature.distance(instances[i].second, d), instances[i].first));
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

void KNN::store(const std::string &path) const {
  json j;
  j["k"] = k;
  j["d"] = d;
  json inst;
  for(auto i: instances) {
    json instance;
    instance["label"] = i.first;
    instance["circularity"] = i.second.get_circularity();
    instance["convex"] = i.second.get_convex();
    instance["aspect_ratio"] = i.second.get_aspect_ratio();
    instance["extent"] = i.second.get_extent();
    json histogram;
    for(auto h: i.second.get_histogram()) {
      histogram.push_back(h);
    }
    instance["histogram"] = histogram;
    inst.push_back(instance);
  }
  j["instances"] = inst;

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
    bool convex = i["convex"];
    double aspect_ratio = i["aspect_ratio"]; 
    double extent = i["extent"];
    
    for(size_t j = 0; j < i["histogram"].size(); j++) {
      hist[j] = i["histogram"][j];
    }
    
    auto features = Features(hist, circularity, convex, aspect_ratio, extent);
    instances.push_back(std::pair(i["label"], features));
  }

  return KNN(j["k"], j["d"], instances);
}

LR::LR() {
}

LR::LR(const std::vector<double> _parameters){
  parameters = _parameters;
}

std::ostream& operator<<(std::ostream &strm, const LR &o) {
  strm << "LR: {'weights':"; //<<o.parameters;
  /*<<"', 'd':"<<o.d<<", instances:['"<<std::endl;
  for(auto i: o.instances) {
    strm << "{'label':"<<i.first<<", 'features':"<<i.second<<"},"<<std::endl;
  }*/
  strm << "}";
  return strm;
}

std::vector<double> compute_gradient(const std::vector<double> &parameters,
const std::vector<std::vector<double>> &features,
const std::vector<double> &labels, const size_t m, const double beta) {
  std::vector<double> predictions, errors;

  for(size_t i = 0; i < features.size(); i++) {
    // compute the prediction
    double p = 0.0;
    for(unsigned int j = 0; j < parameters.size(); j++) {
      p += features[i][j] * parameters[j];
    }
    double pred = sigmoid(p);
    predictions.push_back(pred);
    // compute the error
    errors.push_back(pred - labels[i]);
  }

  // init the gradient
  std::vector<double> gradient;
  for(unsigned int i = 0; i < m; i++) {
    gradient.push_back(0);
  }

  // compute the gradient
  for(unsigned int i = 0; i < m; i++) {
    for(unsigned int j = 0; j < errors.size(); j++) {
      gradient[i] += errors[j] * features[j][i];
    }
    gradient[i] = (gradient[i]/m) + (beta/m)*gradient[i];
  }

  return gradient;
}

void LR::learn(const std::vector<std::pair<std::string, Features>> &inst,
const double alpha, const double beta) {
  std::vector<double> labels;
  std::vector<std::vector<double>> features;
  for(size_t i = 0; i < inst.size(); i++) {
    if (inst[i].first.compare("bad") == 0) {
      labels.push_back(0);
    } else {
      labels.push_back(1);
    }
    auto feature = inst[i].second.get_features();
    features.push_back(feature);
  }

  size_t m = features[0].size();
  std::vector<double> m_t, v_t, m_cap, v_cap;
  for(unsigned int i = 0; i < m; i++) {
    parameters.push_back(0);
    m_t.push_back(0);
    v_t.push_back(0);
    m_cap.push_back(0);
    v_cap.push_back(0);
  }

  double magnitude = 1.0, eps=1e-8, beta1=0.9, beta2=0.999;
  size_t it = 0;
  while(magnitude > 0.001) {
    it++;
    std::vector gradient = compute_gradient(parameters, features, labels, m, beta);
    
    for(size_t i = 0; i < gradient.size(); i++) {
      m_t[i] = beta1 * m_t[i] + (1.0 - beta1) * gradient[i];
      v_t[i] = beta2 * v_t[i] + (1.0 - beta2) * pow(gradient[i], 2);
      m_cap[i] = m_t[i] / (1.0 - pow(beta1, it));
      v_cap[i] = v_t[i] / (1.0 - pow(beta2, it));
    }

    for(size_t i = 0; i < parameters.size(); i++) {
      parameters[i] = parameters[i] - ((alpha * m_cap[i])/(sqrt(v_cap[i]) + eps));
      //parameters[i] = parameters[i] - (alpha * gradient[i]);
    }

    magnitude = magnitude_vector(gradient);
    if(it%10 == 0) {
      std::cout << "GM = " << magnitude << std::endl;
    }
  }
}

std::string LR::predict(const Object &object) const {
  auto features = Features(object.get_contour()).get_features();
  double p = 0.0;
  for(unsigned int j = 0; j < parameters.size(); j++) {
    p += features[j] * parameters[j];
  }
  double pred = 1.0/(1.0+exp(-p));
  std::cout<<"Pred = "<<pred<<std::endl;
  if(pred > 0.5) {
    pred = 1.0;
  } else {
    pred = 0.0;
  }
  if(pred == 0.0) {
    return "bad";
  } else {
    return "good";
  }
}

void LR::store(const std::string &path) const {
  json j;
  json jpar;
  for(auto p: parameters) {
    jpar.push_back(p);
  }
  j["parameters"] = jpar;

  //std::cout<<j.dump(2)<<std::endl;
  std::ofstream o(path);
  o << std::setw(2) << j << std::endl;
}

LR LR::load(const std::string &path) {
  std::ifstream i(path);
  json j;
  i >> j;

  std::vector<double> parameters;

  for(auto p: j["parameters"]) {
    parameters.push_back(p);
  }

  return LR(parameters);
}