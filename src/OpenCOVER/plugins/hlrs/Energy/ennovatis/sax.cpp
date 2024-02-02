#include <cstdlib>
#include <nlohmann/json.hpp>
#include <sax.h>
#include <building.h>
#include <build_options.h>

using json = nlohmann::json;
namespace {
constexpr bool debug = build_options.debug_ennovatis;
}

namespace ennovatis {
bool sax_channelid_parser::string(string_t &val)
{
    if constexpr (debug)
        m_debugLogs.push_back("string(val=" + val + ")");

    if (m_currBuilding) {
        m_buildings.emplace_back(Building(val.c_str()));
    }
    else if (m_channel) {
        // auto &building = m_buildings.back();
        if (m_curChannel.everythingEmpty())
            m_curChannel.clear();
        // TODO: fill channel data
    }
    return true;
}

bool sax_channelid_parser::key(string_t &val)
{
    if constexpr (debug)
        m_debugLogs.push_back("key(val=" + val + ")");

    if (m_triggerd_obj) {
        m_currBuilding = val == "building";
        m_channel = val == "channel";
    }
    return true;
}

bool sax_channelid_parser::null()
{
    if constexpr (debug)
        m_debugLogs.push_back("null()");
    return true;
}

bool sax_channelid_parser::boolean(bool val)
{
    if constexpr (debug)
        m_debugLogs.push_back("boolean(val=" + std::string(val ? "true" : "false") + ")");
    return true;
}

bool sax_channelid_parser::number_integer(number_integer_t val)
{
    if constexpr (debug)
        m_debugLogs.push_back("number_integer(val=" + std::to_string(val) + ")");
    return true;
}

bool sax_channelid_parser::number_unsigned(number_unsigned_t val)
{
    if constexpr (debug)
        m_debugLogs.push_back("number_unsigned(val=" + std::to_string(val) + ")");
    return true;
}

bool sax_channelid_parser::number_float(number_float_t val, const string_t &s)
{
    if constexpr (debug)
        m_debugLogs.push_back("number_float(val=" + std::to_string(val) + ", s=" + s + ")");
    return true;
}

bool sax_channelid_parser::start_object(std::size_t elements)
{
    if constexpr (debug)
        m_debugLogs.push_back("start_object(elements=" + std::to_string(elements) + ")");
    m_triggerd_obj = true;
    return true;
}

bool sax_channelid_parser::end_object()
{
    if constexpr (debug)
        m_debugLogs.push_back("end_object()");
    
    if (!m_channel)
        m_currBuilding = false;
    m_channel = false;
    m_triggerd_obj = false;
    return true;
}

bool sax_channelid_parser::start_array(std::size_t elements)
{
    if constexpr (debug)
        m_debugLogs.push_back("start_array(elements=" + std::to_string(elements) + ")");
    return true;
}

bool sax_channelid_parser::end_array()
{
    if constexpr (debug)
        m_debugLogs.push_back("end_array()");
    return true;
}

bool sax_channelid_parser::binary(json::binary_t &val)
{
    if constexpr (debug)
        m_debugLogs.push_back("binary(val=[...])");
    return true;
}

bool sax_channelid_parser::parse_error(std::size_t position, const std::string &last_token, const json::exception &ex)
{
    if constexpr (debug)
        m_debugLogs.push_back("parse_error(position=" + std::to_string(position) + ", last_token=" + last_token +
                              ",\n            ex=" + std::string(ex.what()) + ")");
    return false;
}
} // namespace ennovatis