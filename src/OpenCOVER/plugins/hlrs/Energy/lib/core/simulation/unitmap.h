#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace core::simulation {

constexpr auto INVALID_UNIT = "unknown";
struct UnitPair {
  std::vector<std::string> names;
  std::string unit;
};

class UnitMap {
 public:
  UnitMap(std::vector<UnitPair> &&unitPairs) {
    for (const auto &pair : unitPairs) {
      for (const auto &name : pair.names) {
        unit_map[name] = pair.unit;
      }
    }
  }

  const std::string operator[](const std::string &key) const {
    auto it = unit_map.find(key);
    if (it != unit_map.end()) {
      return it->second;
    }
    return INVALID_UNIT;
  }

  auto begin() { return unit_map.begin(); }
  auto end() { return unit_map.end(); }

 private:
  std::unordered_map<std::string, std::string> unit_map;
};

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
}  // namespace core::simulation
