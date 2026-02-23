#pragma once
#include <string>

#include "scalarproperties.h"
#include "type.h"

namespace core::simulation {

class ScalarPropertiesProcessor {
 public:
  ScalarPropertiesProcessor(const float &trimPercent = 0.01)
      : trimPercent(trimPercent) {}
  void init(ScalarProperties &properties, const std::string &key,
            const ScalarVec &values);

 private:
  void initMinMax(ScalarProperty &property, const ScalarVec &values);
  // trim values by percent
  float trimPercent;
};
}  // namespace core::simulation
