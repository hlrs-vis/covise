#include "building.h"
#include <sstream>

namespace ennovatis {
bool Channel::empty() const {
  return name.empty() && id.empty() && description.empty() && type.empty() &&
         unit.empty() && group == ChannelGroup::None;
}

void Channel::clear() {
  name.clear();
  id.clear();
  description.clear();
  type.clear();
  unit.clear();
}

const std::string Channel::to_string() const {
  std::stringstream ss;
  ss << "name: " << name << "\ndescription: " << description
     << "\ntype: " << type << "\nunit: " << unit
     << "\nChannelgroup: " << ChannelGroupToString(group);
  return ss.str();
}
} // namespace ennovatis
