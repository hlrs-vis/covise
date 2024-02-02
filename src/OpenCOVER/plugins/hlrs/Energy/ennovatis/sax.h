#ifndef _SAX_H
#define _SAX_H

#include <nlohmann/json.hpp>
#include "building.h"

namespace ennovatis {
class sax_channelid_parser: public nlohmann::json::json_sax_t {
public:
    //   sax_location_parser(EnergyPlugin::DeviceList &_map)
    //       : m_strDevList(std::make_unique<EnergyPlugin::DeviceList>(_map)) {}
    sax_channelid_parser() = default;

    bool string(string_t &val) override;
    bool key(string_t &val) override;
    bool null() override;
    bool boolean(bool val) override;
    bool number_integer(number_integer_t val) override;
    bool number_unsigned(number_unsigned_t val) override;
    bool number_float(number_float_t val, const string_t &s) override;
    bool start_object(std::size_t elements) override;
    bool end_object() override;
    bool start_array(std::size_t elements) override;
    bool end_array() override;
    bool binary(nlohmann::json::binary_t &val) override;
    bool parse_error(std::size_t position, const std::string &last_token, const nlohmann::json::exception &ex) override;
    const std::vector<std::string> &getDebugLogs() const { return m_debugLogs; }
    std::vector<Building> &getBuildings() { return m_buildings; }

private:
    bool m_currBuilding = false;
    bool m_channel = false;
    bool m_triggerd_obj = false;
    Channel m_curChannel;
    //   std::unique_ptr<EnergyPlugin::DeviceList> m_strDevList;
    std::vector<std::string> m_debugLogs;
    std::vector<Building> m_buildings;
};
} // namespace ennovatis

#endif