/**
 * @file lib_oc
 * @brief Object Classification library
 *
 * Functions used to classify objects based on their shape.
 * The code uses chain code histograms and a kNN algorithm to classify objects.
 *
 * @author $Author: Catarina Silva $
 * @version $Revision: 1.0 $
 * @date $Date: 2020/01/05 $
 */

#ifndef OC_H
#define OC_H

#include <filesystem>

#include "json.hpp"
#include "lib_od.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class ChainHistogram {
  private:
    std::array<double, 8> hist;
    friend std::ostream& operator<<(std::ostream&, const ChainHistogram&);

  public:
    ChainHistogram(const std::array<double, 8> &);
    ChainHistogram(const std::vector<cv::Point>&);
    double distance(const ChainHistogram&, const unsigned int p=2) const;
    std::array<double, 8> get_histogram() const;
};

class KNN {
  private:
    unsigned int k;
    std::vector<std::pair<std::string, ChainHistogram>> instances;
    friend std::ostream& operator<<(std::ostream&, const KNN&);

  public:
    KNN(const unsigned int);
    KNN(const unsigned int, const std::vector<std::pair<std::string, ChainHistogram>>&);
    void learn_class(const std::string&, const std::vector<Object>&);
    std::string predict(const Object&) const;
    static void store(const KNN&, const fs::path&);
    static KNN load(const fs::path&);
};

#endif
