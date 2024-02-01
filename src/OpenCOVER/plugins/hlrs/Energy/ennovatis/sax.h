#ifndef _SAX_H
#define _SAX_H

#include <nlohmann/json.hpp>

namespace ennovatis {
/**
 * @brief JSON SAX handler for parsing JSON channelid data to channelid list.
 *
 * This struct implements the nlohmann::json_sax interface to handle SAX events
 * during JSON parsing and creates a channelid list. Each event corresponds to a specific type of JSON value,
 * such as null, boolean, number, string, object, or array.
 */
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

private:
    bool m_currBuilding = false;
    //   std::unique_ptr<EnergyPlugin::DeviceList> m_strDevList;
    std::vector<std::string> m_debugLogs;
};
} // namespace ennovatis

#endif