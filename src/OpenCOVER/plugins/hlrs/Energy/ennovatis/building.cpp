#include "building.h"
#include <sstream>

namespace ennovatis {
bool Channel::empty() const
{
    return name.empty() && id.empty() && description.empty() && type.empty() && unit.empty() &&
           group == ChannelGroup::None;
}

void Channel::clear()
{
    name.clear();
    id.clear();
    description.clear();
    type.clear();
    unit.clear();
}

const std::string Channel::to_string() const
{
    std::stringstream ss;
    ss << "name: " << name << "\nid: " << id << "\ndescription: " << description << "\ntype: " << type
       << "\nunit: " << unit << "\nChannelgroup: " << ChannelGroupToStr(group);
    return ss.str();
}

std::string Channel::ChannelGroupToStr(ChannelGroup group)
{
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
} // namespace ennovatis