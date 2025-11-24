#pragma once
#include <string>
#include <vector>

#include "scalarproperties.h"

namespace core::simulation {

class ScalarPropertiesProcessor {
 public:
  ScalarPropertiesProcessor(const float &trimPercent = 0.01)
      : trimPercent(trimPercent) {}
  void init(ScalarProperties &properties, const std::string &key,
            const std::vector<double> &values);

 private:
  void initMinMax(ScalarProperty &property, const std::vector<double> &values);
  // trim values by percent
  float trimPercent;
};
}  // namespace core::simulation
