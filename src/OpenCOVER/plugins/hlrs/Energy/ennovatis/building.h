/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _BUILDING_H
#define _BUILDING_H
#include <sstream>
#include <string>
#include <array>
#include <set>

namespace ennovatis {
enum ChannelGroup { Strom, Wasser, Waerme, Kaelte, None }; // keep None at the end

/**
 * @brief Represents a channel in ennovatis.
 * 
 * The Channel struct represents a channel in ennovatis, which can be associated with a building.
 * It contains information such as the channel's name, ID, description, type, group and unit.
 */
struct Channel {
    std::string name;
    std::string id;
    std::string description;
    std::string type;
    std::string unit;
    ChannelGroup group = ChannelGroup::None;

    [[nodiscard]] bool empty() const
    {
        return name.empty() && id.empty() && description.empty() && type.empty() && unit.empty() &&
               group == ChannelGroup::None;
    }

    void clear()
    {
        name.clear();
        id.clear();
        description.clear();
        type.clear();
        unit.clear();
    }

    [[nodiscard]] const std::string to_string() const
    {
        std::stringstream ss;
        ss << "name: " << name << "\nid: " << id << "\ndescription" << description << "\ntype: " << type
           << "\nunit: " << unit << "\nChannelgroup: " << group;
        return ss.str();
    }
};

struct ChannelCmp {
    [[nodiscard]] bool operator()(const Channel &lhs, const Channel &rhs) const { return lhs.id < rhs.id; }
};

typedef std::set<Channel, ChannelCmp> ChannelList;
typedef std::array<ChannelList, static_cast<int>(ChannelGroup::None)> ChannelGroups;

/**
 * @brief Represents a building.
 * 
 * The Building class represents a building entity in ennovatis.
 * It contains information such as the building's name, ID, coordinates and channels.
 */
class Building {
public:
    Building(const std::string &name, const std::string &id): m_name(name), m_id(id){};
    Building(const std::string &name): m_name(name), m_id(""){};
    ~Building() = default;

    /**
     * @brief Adds a channel to the building.
     * 
     * This function adds a channel to the building object. The channel is specified by the `channel` parameter,
     * and the type of the channel is specified by the `type` parameter. Channels with existing IDs will be skipped.
     * 
     * @param channel The channel to be added.
     * @param type The type of the channel.
     */
    void addChannel(const Channel &channel, ChannelGroup type) { m_channels[static_cast<int>(type)].insert(channel); }

    [[nodiscard("Unused getter.")]] const auto &getChannels(ChannelGroup type) const
    {
        return m_channels[static_cast<int>(type)];
    }
    [[nodiscard("Unused getter.")]] const auto &getName() const { return m_name; }
    [[nodiscard("Unused getter.")]] const auto &getId() const { return m_id; }
    [[nodiscard("Unused getter.")]] const auto &getLat() const { return m_lat; }
    [[nodiscard("Unused getter.")]] const auto &getLon() const { return m_lon; }
    [[nodiscard("Unused getter.")]] const auto &getHeight() const { return m_height; }
    [[nodiscard("Unused str representation")]] const std::string to_string() const
    {
        return "Building: " + m_name + "\nID: " + m_id + "\n";
    }
    void setId(const std::string &id) { m_id = id; }
    void setLat(float lat) { m_lat = lat; }
    void setLon(float lon) { m_lon = lon; }
    void setHeight(float h) { m_height = h; }

private:
    std::string m_name;
    std::string m_id;
    float m_lat;
    float m_lon;
    float m_height;
    ChannelGroups m_channels;
};
} // namespace ennovatis
#endif