#pragma once

#include <string>
#include <vector>

#include "channel.h"

namespace ennovatis {

/**
 * @brief Represents a building.
 *
 * The Building class represents a building entity in ennovatis.
 * It contains information such as the building's name, ID, coordinates and
 * channels.
 */
class Building {
 public:
  // typedef std::shared_ptr<Building> ptr;
  Building(const std::string &name, const std::string &id, const std::string &street)
      : m_name(name),
        m_id(id),
        m_street(street),
        m_y(0),
        m_x(0),
        m_height(0),
        m_area(0) {};
  Building(const std::string &name, const std::string &id)
      : Building(name, id, "") {};
  Building(const std::string &name) : Building(name, "", "") {};
  ~Building() = default;
  Building(const Building &other) = default;
  Building &operator=(const Building &other) = default;

  /**
   * @brief Adds a channel to the building.
   *
   * This function adds a channel to the building object. The channel is
   * specified by the `channel` parameter, and the type of the channel is
   * specified by the `type` parameter. Channels with existing IDs will be
   * skipped.
   *
   * @param channel The channel to be added.
   * @param type The type of the channel.
   */
  void addChannel(const Channel &channel, ChannelGroup type) {
    m_channels[static_cast<int>(type)].insert(channel);
  }

  [[nodiscard]] const auto &getChannels(ChannelGroup type) const {
    return m_channels[static_cast<int>(type)];
  }
  [[nodiscard]] const auto &getName() const { return m_name; }
  [[nodiscard]] const auto &getStreet() const { return m_street; }
  [[nodiscard]] const auto &getId() const { return m_id; }
  [[nodiscard]] const auto &getY() const { return m_y; }
  [[nodiscard]] const auto &getX() const { return m_x; }
  [[nodiscard]] const auto &getHeight() const { return m_height; }
  [[nodiscard]] const auto &getArea() const { return m_area; }
  [[nodiscard]] const std::string to_string() const {
    return "Building: " + m_name + "\nID: " + m_id + "\n" + "\nStreet: " + m_street +
           "\n";
  }
  void setName(const std::string &name) { m_name = name; }
  void setId(const std::string &id) { m_id = id; }
  void setStreet(const std::string &street) { m_street = street; }
  void setY(float lat) { m_y = lat; }
  void setX(float lon) { m_x = lon; }
  void setHeight(float h) { m_height = h; }
  void setArea(float a) { m_area = a; }

 private:
  std::string m_name;
  std::string m_street;
  std::string m_id;
  float m_y;
  float m_x;
  float m_height;
  float m_area;
  ChannelGroups m_channels;
};

typedef std::vector<Building> Buildings;
}  // namespace ennovatis
