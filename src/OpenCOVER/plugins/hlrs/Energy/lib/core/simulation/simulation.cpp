#include "simulation.h"

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

const core::simulation::UnitMap UNIT_MAP = core::simulation::UnitMap(
    {{{"kWh", "leistung", "power"}, "kWh"},
     {{"kW"}, "kW"},
     {{"q_dem_w", "waermestromdichte"}, "W/m2"},
     {{"delta_q", "aenderung_stromdichte"}, "W/m2"},
     {{"mass_flow", "massenstrom"}, "kg/s"},
     {{"celcius", "temp", "inlet_temp", "outlet_temp"}, "°C"},
     {{"electricity_selling_price"}, "Cent/kWh"},
     {{"heating_cost"}, "€"},
     {{"voltage", "volt"}, "V"},
     {{"current", "ampere"}, "A"},
     {{"i_ka"}, "kA"},
     {{"resistance", "ohm"}, "Ω"},
     {{"power_factor", "cos_phi"}, ""},
     {{"efficiency", "eta"}, ""},
     {{"reactive_power", "q"}, "var"},
     {{"active_power", "p"}, "W"},
     {{"apparent_power", "s"}, "VA"},
     {{"vm_pu"}, "pu (voltage per unit)"},
     {{"q_mvar"}, "Mvar"},
     {{"loading_percent", "percent"}, "%"},
     {{"res_mw"}, "MW"}});

const core::simulation::UnitMap COLORMAP_MAP =
    core::simulation::UnitMap({{{"res_mw"}, "Power_Grey"},
                               {{"loading_percent"}, "Utilization"},
                               {{"vm_pu"}, "Voltage"}});
}  // namespace

namespace core::simulation {

void Simulation::computeMinMax(const std::string &key,
                               const std::vector<double> &values,
                               const double &trimPercent) {
  auto [min_elem, max_elem] = robustMinMax(values, trimPercent);
  m_scalarProperties[key].min = min_elem;
  m_scalarProperties[key].max = max_elem;
}

void Simulation::computeMaxTimestep(const std::string &key,
                                    const std::vector<double> &values) {
  m_scalarProperties[key].timesteps = values.size();
}

void Simulation::setUnit(const std::string &key) {
  m_scalarProperties[key].unit = UNIT_MAP[key];
}

void Simulation::setPreferredColorMap(const std::string &key) {
  m_scalarProperties[key].preferredColorMap = COLORMAP_MAP[key];
}

}  // namespace core::simulation
