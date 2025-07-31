#pragma once
#include <map>
#include <string>
#include <vector>

namespace core::simulation {
typedef std::map<std::string, std::vector<double>> Data;

class Object {
 public:
  Object(const ::std::string &name, const Data &data = {})
      : m_name(name), m_data(data) {};
  virtual ~Object() = default;

  const auto &getName() const { return m_name; }

  auto &getData() { return m_data; }
  const auto &getData() const { return m_data; }

  void addData(const std::string &key, const std::vector<double> &value) {
    m_data[key] = value;
  }

  void addData(const std::string &key, const double &value) {
    m_data[key].push_back(value);
  }

  Data::const_iterator begin() const { return m_data.begin(); }
  Data::const_iterator end() const { return m_data.end(); }
  void emplace_back(const std::string &key, const double &value) {
    m_data[key].emplace_back(value);
  }

 private:
  std::string m_name;
  Data m_data;  // timestep data
};
}  // namespace core::simulation
