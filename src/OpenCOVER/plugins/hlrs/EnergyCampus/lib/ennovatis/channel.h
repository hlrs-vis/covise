#pragma once
#include <array>
#include <set>
#include <string>

namespace ennovatis {
enum ChannelGroup { Strom, Wasser, Waerme, Kaelte, None };  // keep None at the end
constexpr const char *ChannelGroupToStringConstExpr(ChannelGroup group) {
  switch (group) {
    case ChannelGroup::Strom:
      return "Strom";
    case ChannelGroup::Wasser:
      return "Wasser";
    case ChannelGroup::Waerme:
      return "Waerme";
    case ChannelGroup::Kaelte:
      return "Kaelte";
    case ChannelGroup::None:
      return "None";
  }
  return "None";
}

inline const std::string ChannelGroupToString(ChannelGroup group) {
  switch (group) {
    case ChannelGroup::Strom:
      return "Strom";
    case ChannelGroup::Wasser:
      return "Wasser";
    case ChannelGroup::Waerme:
      return "Waerme";
    case ChannelGroup::Kaelte:
      return "Kaelte";
    case ChannelGroup::None:
      return "None";
  }
  return "None";
}

/**
 * @brief Represents a channel in ennovatis.
 *
 * The Channel struct represents a channel in ennovatis, which can be associated
 * with a building. It contains information such as the channel's name, ID,
 * description, type, group and unit.
 */
struct Channel {
  std::string name;
  std::string id;
  std::string description;
  std::string type;
  std::string unit;
  ChannelGroup group = ChannelGroup::None;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] const std::string to_string() const;
  void clear();
};

struct ChannelCmp {
  [[nodiscard]] bool operator()(const Channel &lhs, const Channel &rhs) const {
    return lhs.id < rhs.id;
  }
};

typedef std::set<Channel, ChannelCmp> ChannelList;
typedef std::array<ChannelList, static_cast<int>(ChannelGroup::None)> ChannelGroups;
}  // namespace ennovatis
