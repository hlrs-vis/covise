#include "csv.h"
#include <iostream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include "building.h"

namespace {

void channelgroup_switch(ennovatis::Channel &channel, const std::string &val)
{
    namespace algo = boost::algorithm;
    if (algo::contains(val, "POWER"))
        channel.group = ennovatis::ChannelGroup::Strom;
    else if (algo::contains(val, "WATER"))
        channel.group = ennovatis::ChannelGroup::Wasser;
    else if (algo::contains(val, "HEAT"))
        channel.group = ennovatis::ChannelGroup::Waerme;
    else if (algo::contains(val, "COOLING"))
        channel.group = ennovatis::ChannelGroup::Kaelte;
}
} // namespace


namespace ennovatis {

bool csv_channelid_parser::update_buildings_by_buildingid(std::basic_istream<char> &file, BuildingsPtr buildings)
{
    std::string row("");
    std::string lastBuildingId("");
    std::getline(file, row); // skip header
    std::vector<Building>::iterator buildingIt;
    while (file.good()) {
        std::getline(file, row);
        if (row.empty())
            continue;

        std::istringstream ss(row);
        std::string column("");
        std::vector<std::string> columns;
        while (std::getline(ss, column, ','))
            columns.push_back(column);

        const auto &building_id = columns[(int)CSV_ChannelID_Column::BUILDING_ID];
        if (building_id != lastBuildingId) {
            const auto &name_channel_dir = columns[(int)CSV_ChannelID_Column::BUILDING_CHANNEL_DIR];
            const auto name = name_channel_dir.substr(0, name_channel_dir.find_last_of('-'));

            buildingIt = std::find_if(buildings->begin(), buildings->end(),
                                      [&name](const Building &b) { return b.getId().compare(name) >= 0; });
            if (buildingIt == buildings->end()) {
                buildings->push_back(Building(name, building_id));
                buildingIt = buildings->end() - 1;
            }
        }
        lastBuildingId = building_id;

        const auto &channel_name = columns[(int)CSV_ChannelID_Column::CHANNEL];
        // channelid in csv is not the same as the channel id on server side
        const auto &channel_id =
            columns[(int)CSV_ChannelID_Column::GLOBAL_ID] + "_5"; // _5 needed to specify as channel
        const auto &channel_description = columns[(int)CSV_ChannelID_Column::DESCRIPTION];
        const auto &channel_type = columns[(int)CSV_ChannelID_Column::TYPE];

        Channel channel{channel_name, channel_id, channel_description, channel_type, "None", ChannelGroup::None};
        channelgroup_switch(channel, channel_type);
        if (channel.group == ChannelGroup::None)
            continue;

        buildingIt->addChannel(channel, channel.group);
    }
    return true;
}
} // namespace ennovatis
