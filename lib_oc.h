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

#include "json.hpp"
#include "lib_od.h"

using json = nlohmann::json;

class Features {
  private:
    std::array<double, 8> hist;
    double circularity;
    friend std::ostream& operator<<(std::ostream&, const Features&);

  public:
    Features(const std::array<double, 8> &, const double);
    Features(const std::vector<cv::Point>&);
    double distance(const Features&, const unsigned int p=2) const;
    std::pair<std::array<double, 8>,double> get_features() const;
};

class KNN {
  private:
    unsigned int k, d;
    std::vector<std::pair<std::string, Features>> instances;
    friend std::ostream& operator<<(std::ostream&, const KNN&);

  public:
    KNN(const unsigned int, const unsigned int);
    KNN(const unsigned int, const unsigned int, const std::vector<std::pair<std::string, Features>>&);
    void learn_class(const std::string&, const std::vector<Object>&);
    std::string predict(const Object&) const;
    static void store(const KNN&, const std::string&);
    static KNN load(const std::string&);
};

#endif
