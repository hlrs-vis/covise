/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _BUILDING_H
#define _BUILDING_H
#include <string>
#include <array>
#include <set>

namespace ennovatis {
enum class ChannelGroup { Strom, Wasser, Waerme, Kaelte, None }; // keep None at the end

/**
 * @brief Represents a channel in ennovatis.
 * 
 * The Channel struct represents a channel in ennovatis, which can be associated with a building.
 * It contains information such as the channel's name, ID, description, type, and unit.
 */
struct Channel {
    std::string name;
    std::string id;
    std::string description;
    std::string type;
    std::string unit;
    ChannelGroup group = ChannelGroup::None;

    bool empty() const
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
};

struct ChannelCmp {
    bool operator()(const Channel &lhs, const Channel &rhs) const { return lhs.id < rhs.id; }
};

typedef std::set<Channel, ChannelCmp> ChannelList;
typedef std::array<ChannelList, static_cast<int>(ChannelGroup::None)> ChannelGroups;

/**
 * @brief Represents a building.
 * 
 * The Building class represents a building entity in ennovatis.
 * It contains information such as the building's name, ID, and channels.
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
    void addChannel(const Channel &channel, ChannelGroup type)
    {
        // m_channels[static_cast<int>(type)].push_back(channel);
        m_channels[static_cast<int>(type)].insert(channel);
    }

    const ChannelList &getChannels(ChannelGroup type) const { return m_channels[static_cast<int>(type)]; }
    const std::string &getName() const { return m_name; }
    const std::string &getId() const { return m_id; }
    void setId(const std::string &id) { m_id = id; }

private:
    std::string m_name;
    std::string m_id;
    ChannelGroups m_channels;
};
} // namespace ennovatis
#endif