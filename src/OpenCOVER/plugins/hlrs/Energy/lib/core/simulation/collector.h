#pragma once

#include "object.h"

namespace core::simulation {
template <typename T>
class Collector {
 public:
  virtual ~Collector() = default;
  virtual T collect() = 0;
};

typedef std::map<std::string, std::vector<double>> ScalarValuesMap;

class ScalarCollector
    : public Collector<ScalarValuesMap> {
 public:
  ScalarCollector(const ObjectMapView &view) : m_view(view) {}
  ScalarValuesMap collect() override;

 private:
  ObjectMapView m_view;
};
}  // namespace core::simulation
