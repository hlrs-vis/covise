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

typedef const ScalarVec *const_ScalarVecs;

class ScalarByNameCollector : public ScalarCollector<const_ScalarVecs> {
 public:
  ScalarByNameCollector(const ObjectMapView &view, std::string_view name,
                        std::string_view species)
      : ScalarCollector<const_ScalarVecs>(view), m_name(name), m_species(species) {}
  const_ScalarVecs collect() override;

 private:
  std::string_view m_name;
  std::string_view m_species;
};
}  // namespace core::simulation
