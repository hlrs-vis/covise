#ifndef _BUILDING_H
#define _BUILDING_H

#include <string>
#include <vector>
#include <memory>
#include "channel.h"

namespace ennovatis {

/**
 * @brief Represents a building.
 * 
 * The Building class represents a building entity in ennovatis.
 * It contains information such as the building's name, ID, coordinates and channels.
 */
class Building {
public:
    Building(const std::string &name, const std::string &id, const std::string &street)
    : m_name(name), m_id(id), m_street(street), m_lat(0), m_lon(0), m_height(0){};
    Building(const std::string &name, const std::string &id): Building(name, id, ""){};
    Building(const std::string &name): Building(name, "", ""){};
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

    [[nodiscard]] const auto &getChannels(ChannelGroup type) const { return m_channels[static_cast<int>(type)]; }
    [[nodiscard]] const auto &getName() const { return m_name; }
    [[nodiscard]] const auto &getStreet() const { return m_street; }
    [[nodiscard]] const auto &getId() const { return m_id; }
    [[nodiscard]] const auto &getLat() const { return m_lat; }
    [[nodiscard]] const auto &getLon() const { return m_lon; }
    [[nodiscard]] const auto &getHeight() const { return m_height; }
    [[nodiscard]] const std::string to_string() const
    {
        return "Building: " + m_name + "\nID: " + m_id + "\n" + "\nStreet: " + m_street + "\n";
    }
    void setName(const std::string &name) { m_name = name; }
    void setId(const std::string &id) { m_id = id; }
    void setStreet(const std::string &street) { m_street = street; }
    void setLat(float lat) { m_lat = lat; }
    void setLon(float lon) { m_lon = lon; }
    void setHeight(float h) { m_height = h; }

private:
    std::string m_name;
    std::string m_street;
    std::string m_id;
    float m_lat;
    float m_lon;
    float m_height;
    ChannelGroups m_channels;
};

typedef std::vector<Building> Buildings;
typedef std::shared_ptr<Buildings> BuildingsPtr;
} // namespace ennovatis
#endif