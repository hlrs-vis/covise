#pragma once

#include "object.h"
#include "type.h"

namespace core::simulation {    

template <typename T>
class Collector {
 public:
  virtual ~Collector() = default;
  virtual T collect() = 0;
};

template <typename T>
class ScalarCollector : public Collector<T> {
 public:
  ScalarCollector(const ObjectMapView &view) : m_view(view) {}
  virtual ~ScalarCollector() = default;

 protected:
  ObjectMapView m_view;
};

class ScalarMapCollector : public ScalarCollector<ScalarMap> {
 public:
  ScalarMapCollector(const ObjectMapView &view) : ScalarCollector<ScalarMap>(view) {}
  ScalarMap collect() override;
};


class ScalarByNameCollector : public ScalarCollector<ScalarByNameCollectorResult> {
 public:
  ScalarByNameCollector(const ObjectMapView &view, std::string_view name,
                        std::string_view species)
      : ScalarCollector<ScalarByNameCollectorResult>(view), m_name(name), m_species(species) {}
  ScalarByNameCollectorResult collect() override;

 private:
  std::string_view m_name;
  std::string_view m_species;
};
}  // namespace core::simulation
