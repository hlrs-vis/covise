#ifndef _CSV_H
#define _CSV_H

#include <iostream>
#include "building.h"

namespace ennovatis {

struct csv_channelid_parser {
    [[nodiscard]] static bool update_buildings_by_buildingid(std::basic_istream<char> &file, BuildingsPtr buildings);
    [[nodiscard]] static bool update_buildings_by_buildingid(const std::string &filename, BuildingsPtr buildings);

private:
    enum class CSV_ChannelID_Column {
        PROJECT = 0,
        PROJECT_ID = 1,
        BUILDING_CHANNEL_DIR = 2,
        BUILDING_ID = 3,
        DATASOURCE_SUBCHANNEL_DIR = 4,
        DATASOURCE_ID = 5,
        CHANNEL = 6,
        CHANNEL_ID = 7,
        GLOBAL_ID = 8,
        DESCRIPTION = 9,
        TYPE = 10,
        MAIN_SUB_TYPE = 11,
        BASELINE = 12,
        PREDICTION = 13,
        ACRONYM = 14,
        FUNCTION = 15,
    };
};

} // namespace ennovatis
#endif