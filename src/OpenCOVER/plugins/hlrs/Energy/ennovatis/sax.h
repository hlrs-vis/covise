#ifndef _SAX_H
#define _SAX_H

#include <nlohmann/json.hpp>
#include "building.h"

namespace ennovatis {

typedef std::vector<Building> Buildings;

class sax_channelid_parser: public nlohmann::json::json_sax_t {
public:
    sax_channelid_parser() = default;
    sax_channelid_parser(std::shared_ptr<Buildings> buildings): m_buildings(buildings){};

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
    bool m_isBuilding = false;
    bool m_isChannel = false;
    bool m_isBuildingID = false;
    bool m_isObj = false;
    // current channel in iteration
    Channel m_channel;
    // current attr key in channel iteration
    std::string m_curChannelAttrKey;
    std::vector<std::string> m_debugLogs;
    std::shared_ptr<Buildings> m_buildings;
};
} // namespace ennovatis

#endif