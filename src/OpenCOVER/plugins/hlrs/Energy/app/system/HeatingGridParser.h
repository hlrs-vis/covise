#pragma once
#include "GridParser.h"
#include "app/osg/presentation/grid.h"
#include <utils/read/csv/csv.h>

using namespace opencover::utils::read;

struct HeatingGridParser final : GridParser
{
    using GridParser::GridParser;
    grid_ptr operator()(CSVDataMap &map) override;
    grid_ptr operator()(const ArrowDataMap &map) override;
    grid_ptr operator()(const ArrowData &data) override;
    grid_ptr operator()(CSVData &data) override;

private:
    void initHeatingGridStreams();
    void initHeatingGrid();
    void buildHeatingGrid();
    void readSimulationDataStream(CSVStream &heatingSimStream);
    void applySimulationDataToHeatingGrid();
    void readHeatingGridStream(CSVStream &heatingStream);
    std::vector<int> createHeatingGridIndices(
        const std::string &pointName,
        const std::string &connectionsStrWithCommaDelimiter,
        grid::ConnectionDataList &additionalData);

    osg::ref_ptr<grid::Line> createHeatingGridLine(
        const grid::Points &points, osg::ref_ptr<grid::Point> from,
        const std::string &connectionsStrWithCommaDelimiter,
        grid::ConnectionDataList &additionalData);
    std::pair<grid::Points, grid::Data> createHeatingGridPointsAndData(
        CSVStream &heatingStream, std::map<int, std::string> &connectionStrings);
    grid::Lines createHeatingGridLines(
        const grid::Points &points,
        const std::map<int, std::string> &connectionStrings,
        grid::ConnectionDataList &additionalData);
    osg::ref_ptr<grid::Point> searchHeatingGridPointById(const grid::Points &points,
        int id);
};
