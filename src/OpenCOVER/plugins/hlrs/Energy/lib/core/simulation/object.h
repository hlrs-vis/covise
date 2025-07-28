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

template <typename T>
class ObjectContainer {
  static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");

 public:
  void add(const std::string &name, const Data &data = {}) {
    m_elements.emplace(name, T(name, data));
  }

  template <typename... Args>
  void addDataToContainerObject(const std::string &containerName, Args &&...args) {
    if (auto it = m_elements.find(containerName); it != m_elements.end())
      it->second.addData(std::forward<Args>(args)...);
  }

  const auto &get() const { return m_elements; }
  const auto begin() const { return m_elements.begin(); }
  auto begin() { return m_elements.begin(); }
  const auto end() const { return m_elements.end(); }
  auto end() { return m_elements.end(); }
  auto find(const std::string &name) const { return m_elements.find(name); }
  T &operator[](const std::string &name) { return m_elements.at(name); }
  const T &operator[](const std::string &name) const { return m_elements.at(name); }

 private:
  std::map<std::string, T> m_elements;
};
}  // namespace core::simulation
