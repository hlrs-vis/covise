#pragma once
#include "GridParser.h"
#include <utils/read/csv/csv.h>

struct PowerGridParser final : GridParser
{
    using GridParser::GridParser;
    grid_ptr operator()(CSVDataMap &map) override;
    grid_ptr operator()(const ArrowDataMap &map) override;
    grid_ptr operator()(const ArrowData &data) override;
    grid_ptr operator()(CSVData &data) override;

private:
    typedef std::unordered_map<int, std::string> IDLookupTable;

    std::vector<IDLookupTable> retrieveBusNameIdMapping(opencover::utils::read::CSVStream &stream);
    std::vector<grid::PointsMap> createPowerGridPoints(
        opencover::utils::read::CSVStream &stream, size_t &numPoints,
        const float &sphereRadius, const std::vector<IDLookupTable> &busNames, CSVDataMap &map);
    std::pair<std::vector<grid::Lines>, std::vector<grid::ConnectionDataList>>
    getPowerGridLines(opencover::utils::read::CSVStream &stream, const std::vector<grid::PointsMap> &points);
    osg::ref_ptr<grid::Line> createLine(
        const std::string &name, int &from, const std::string &geoBuses_comma_seperated,
        grid::Data &data, const std::vector<grid::PointsMap> &points);
    void helper_getAdditionalPowerGridPointData_addData(
        int busId, grid::PointDataList &additionalData, const grid::Data &data);
    void helper_getAdditionalPowerGridPointData_handleDuplicate(
        std::string &name, std::map<std::string, uint> &duplicateMap);
    std::unique_ptr<grid::PointDataList> getAdditionalPowerGridPointData(
        const std::size_t &numOfBus, CSVDataMap &map);
};
