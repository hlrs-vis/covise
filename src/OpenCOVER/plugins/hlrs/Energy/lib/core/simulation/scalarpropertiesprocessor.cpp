#include "scalarpropertiesprocessor.h"

#include <algorithm>
#include <cmath>

namespace {
std::pair<double, double> robustMinMax(const std::vector<double> &values,
                                       double trimPercent = 0.01) {
  if (values.empty()) return {0.0, 0.0};
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  size_t trim = static_cast<size_t>(std::round(n * trimPercent));
  size_t lower = std::min(trim, n - 1);
  size_t upper = std::max(n - trim - 1, lower);
  return {sorted[lower], sorted[upper]};
}
}  // namespace

namespace core::simulation {
void ScalarPropertiesProcessor::initMinMax(ScalarProperty &property,
                                           const std::vector<double> &values) {
  auto [min_elem, max_elem] = robustMinMax(values, trimPercent);
  property.min = min_elem;
  property.max = max_elem;
}

void ScalarPropertiesProcessor::init(ScalarProperties &properties,
                                     const std::string &key,
                                     const std::vector<double> &values) {
  initMinMax(properties.ref()[key], values);
  properties.setUnit(key);
  properties.setPreferredColorMap(key);
  properties.setTimesteps(key, values.size());
  properties.setSpecies(key, key);
}
}  // namespace core::simulation
