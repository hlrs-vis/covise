#pragma once
#include <map>
#include <string>
#include <vector>
#include <string_view>
#include "type.h"

namespace core::simulation {
using Data = std::map<std::string, ScalarVec, std::less<>>;

class Object {
 public:
//   Object() = default;
  Object(const ::std::string &name, const Data &data = {})
      : m_name(name), m_data(data) {};

  const auto &getName() const { return m_name; }

  auto &getData() { return m_data; }
  const auto &getData() const { return m_data; }

  void addData(std::string key, ScalarVec value) {
    m_data.insert_or_assign(std::move(key), std::move(value));
  }

  void addData(std::string_view key, const double &value) {
    m_data[std::string(key)].push_back(value);
  }

  decltype(auto) operator[]( const std::string &param )       { return m_data[param]; }

  auto begin()       { return m_data.begin(); }
  auto begin() const { return m_data.begin(); }
  auto end()         { return m_data.end(); }
  auto end()   const { return m_data.end(); }

  void emplace_back(const std::string &key, double value) {
    m_data[key].emplace_back(std::move(value));
  }

 private:
  std::string m_name;
  Data m_data;  // timestep data
};

using ObjectMap = std::map<std::string, Object, std::less<>>;
using ObjectMapView = std::vector<const ObjectMap *>;
}  // namespace core::simulation
