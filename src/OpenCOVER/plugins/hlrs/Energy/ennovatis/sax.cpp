#include <cstdlib>
#include <nlohmann/json.hpp>
#include <sax.h>
#include <building.h>
#include <build_options.h>

using json = nlohmann::json;
namespace {
constexpr bool debug = build_options.debug_ennovatis;

void channelgroup_switch(ennovatis::Channel &channel, const std::string &val)
{
    if (val == "Strom")
        channel.group = ennovatis::ChannelGroup::Strom;
    else if (val == "Wasser")
        channel.group = ennovatis::ChannelGroup::Wasser;
    else if (val == "Waerme")
        channel.group = ennovatis::ChannelGroup::Waerme;
    else if (val == "Kaelte")
        channel.group = ennovatis::ChannelGroup::Kaelte;
}

void add_attr_to_channel(ennovatis::Channel &channel, const std::string &key, const std::string &val)
{
    if (key == "channel")
        channel.name = val;
    else if (key == "channelId")
        channel.id = val;
    else if (key == "description")
        channel.description = val;
    else if (key == "type")
        channel.type = val;
    else if (key == "unit")
        channel.unit = val;
    else if (key == "group")
        channelgroup_switch(channel, val);
}
} // namespace

namespace ennovatis {
bool sax_channelid_parser::string(string_t &val)
{
    if constexpr (debug)
        m_debugLogs.push_back("string(val=" + val + ")");

    if (m_isBuilding) {
        m_buildings->push_back(Building(val.c_str()));
        m_isBuilding = false;
    } else if (m_isBuildingID) {
        m_buildings->back().setId(val);
        m_isBuildingID = false;
    } else if (m_isChannel)
        add_attr_to_channel(m_channel, m_curChannelAttrKey, val);

    return true;
}

bool sax_channelid_parser::key(string_t &val)
{
    if constexpr (debug)
        m_debugLogs.push_back("key(val=" + val + ")");

    if (m_isObj) {
        m_isBuilding = val == "building";
        m_isChannel = val == "channel";
        m_isObj = false;
    }
    m_isBuildingID = val == "buildingID";
    m_curChannelAttrKey = (m_isChannel) ? val : "";
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
    m_isObj = true;
    return true;
}

bool sax_channelid_parser::end_object()
{
    if constexpr (debug)
        m_debugLogs.push_back("end_object()");

    if (m_isChannel)
        m_buildings->back().addChannel(m_channel, m_channel.group);

    m_isChannel = false;
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