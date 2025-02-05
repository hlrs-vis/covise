#include "simulation.h"

#include <algorithm>

namespace core::simulation {

void Simulation::computeMinMax(const std::string &key,
                               const std::vector<double> &values) {
  const auto &[min_elem, max_elem] =
      std::minmax_element(values.begin(), values.end());
  if (auto it = m_minMax.find(key); it == m_minMax.end()) {
    m_minMax.insert({key, {*min_elem, *max_elem}});
  } else {
    const auto &[min, max] = it->second;
    if (*min_elem < min) it->second.first = *min_elem;
    if (*max_elem > max) it->second.second = *max_elem;
  }
}

void Simulation::computeMaxTimestep(const std::string &key,
                                    const std::vector<double> &values) {
  m_timesteps[key] = values.size();
}
}  // namespace core::simulation
