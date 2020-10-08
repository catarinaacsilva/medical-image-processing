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

/**
 * Features extraction class to obtain the following features: circularity, roundness, aspect ratio and solidity
 *
 */
class Features {
  private:
    std::array<double, 8> hist;
    double circularity, roundness, aspect_ratio, solidity;
    friend std::ostream& operator<<(std::ostream&, const Features&);

  public:
    Features(const std::array<double, 8> &, const double, const double, const double, const double);
    Features(const std::vector<cv::Point>&);
    double distance(const Features&, const unsigned int p=2) const;
    std::vector<double> get_features() const;
    std::array<double, 8> get_histogram() const;
    double get_circularity() const;
    double get_roundness() const;
    double get_aspect_ratio() const;
    double get_solidity() const;
};

class ML {
  public:
    virtual void learn(const std::vector<std::pair<std::string, Features>>&) = 0;
    virtual std::string predict(const Object&) const = 0;
    virtual void store(const std::string&) const = 0;
    
    static ML& load(const std::string&);

    friend std::ostream& operator<<(std::ostream&, const ML&);
};

/**
 * A kNN implementation class
 *
 */
class KNN : public ML {
  private:
    unsigned int k, d;
    std::vector<std::pair<std::string, Features>> instances;
    friend std::ostream& operator<<(std::ostream&, const KNN&);

  public:
    KNN(const unsigned int, const unsigned int);
    KNN(const unsigned int, const unsigned int, const std::vector<std::pair<std::string, Features>>&);
    
    void learn(const std::vector<std::pair<std::string, Features>>&);
    std::string predict(const Object&) const;
    void store(const std::string&) const;
    
    static KNN& load(const json&);
};

/**
 * A Logistic Regression implementation class
 *
 */
class LR : public ML {
  private:
    std::vector<double> parameters;
    friend std::ostream& operator<<(std::ostream&, const LR&);
  
  public:
    LR();
    LR(const std::vector<double> parameters);
    
    void learn(const std::vector<std::pair<std::string, Features>>&);
    std::string predict(const Object&) const;
    void store(const std::string&) const;
    
    static LR& load(const json&);
};

#endif
