#include "HeatingGridParser.h"
#include "app/osg/presentation/OsgTxtInfoboard.h"
#include "app/osg/presentation/EnergyGrid.h"
#include <lib/core/constants.h>
#include <util/string_util.h>
#include <proj.h>

grid_ptr HeatingGridParser::operator()(CSVDataMap &map)
{
    auto heatingIt = map.find("heating_network");
    if (heatingIt == map.end())
    {
        m_logger.error("Necessary heating network file heating_network cannot be found.");
        return nullptr;
    }
    auto heatingStream = heatingIt->second;
    CSVStream::CSVRow row;
    grid::ConnectionDataList additionalConnectionData {};

    OsgTxtBoxAttributes infoboardAttributes = OsgTxtBoxAttributes(
        { 0, 0, 0 }, "EnergyGridText", m_config.font, 50, 50, 2.0f, 0.1, 2);

    std::map<int, std::string> connectionStrings;
    auto [points, pointData] = createHeatingGridPointsAndData(*heatingStream, connectionStrings);
    auto lines = createHeatingGridLines(points, connectionStrings, additionalConnectionData);

    EnergyGridConfig config;
    config.name = "HEATING";
    config.points = points;
    config.parent = m_config.parent;
    config.connectionRadius = m_config.connectionsRadius;
    config.additionalConnectionData = additionalConnectionData;
    config.infoboardAttributes = infoboardAttributes;
    config.connectionType = EnergyGridConnectionType::Line;
    config.lines = lines;

    auto grid = std::make_shared<EnergyGrid>(config, m_logger);
    grid->initDrawable();

    return std::move(grid);
}

std::pair<grid::Points, grid::Data> HeatingGridParser::createHeatingGridPointsAndData(
    CSVStream &heatingStream, std::map<int, std::string> &connectionStrings)
{
    grid::Points points {};
    grid::Data pointData {};
    CSVStream::CSVRow row;
    std::string name = "", connections = "", label = "", type = "";
    float lat = 0.0f, lon = 0.0f;
    auto checkForInvalidValue = [](const std::string &value)
    {
        return value == INVALID_CELL_VALUE;
    };

    std::string projFrom = m_config.proj.projFrom;
    std::string projTo = m_config.proj.projTo;
    auto P = proj_create_crs_to_crs(PJ_DEFAULT_CTX, projFrom.c_str(), projTo.c_str(), NULL);
    PJ_COORD coord;
    coord.lpzt.z = 0.0;
    coord.lpzt.t = HUGE_VAL;
    bool mapdrape = true;

    if (!P)
    {
        m_logger.warn("Ignore mapping. No valid projection was "
             "found between given proj string in "
             "config EnergyCampus.toml");
        mapdrape = false;
    }

    auto addToPointData = [&checkForInvalidValue](grid::Data &pointData,
                              const std::string &key,
                              const std::string &value)
    {
        if (!checkForInvalidValue(value))
            pointData[key] = value;
    };

    while (heatingStream.readNextRow(row))
    {
        ACCESS_CSV_ROW(row, "connections", connections);
        ACCESS_CSV_ROW(row, "id", name);
        ACCESS_CSV_ROW(row, "Latitude", lat);
        ACCESS_CSV_ROW(row, "Longitude", lon);
        ACCESS_CSV_ROW(row, "Label", label);
        ACCESS_CSV_ROW(row, "Type", type);

        addToPointData(pointData, "name", name);
        addToPointData(pointData, "label", label);
        addToPointData(pointData, "type", type);

        coord.lpzt.lam = lon;
        coord.lpzt.phi = lat;

        coord = proj_trans(P, PJ_FWD, coord);

        lon = coord.xy.x + m_config.offset[0];
        lat = coord.xy.y + m_config.offset[1];
        auto height = m_config.offset[2];

        int strangeId = std::stoi(name);

        // create a point
        osg::ref_ptr<grid::Point> point = new grid::Point(name, lon, lat, height, 1.0f, m_logger, pointData);
        points.push_back(point);

        // needs cleanup because dataset is not final and has empty cells => no need to
        // display them
        pointData.clear();
        row.clear();
        if (connections.empty() || connections == INVALID_CELL_VALUE)
        {
            m_logger.warn("No connections for point: " + name);
            continue;
        }
        connectionStrings[strangeId] = connections;
    }

    return std::make_pair(points, pointData);
}

grid::Lines HeatingGridParser::createHeatingGridLines(
    const grid::Points &points, const std::map<int, std::string> &connectionStrings,
    grid::ConnectionDataList &additionalData)
{
    grid::Lines lines;
    for (auto it = connectionStrings.begin(); it != connectionStrings.end(); ++it)
    {
        int id = it->first;
        const std::string &connectionsStr = it->second;
        if (connectionsStr.empty() || connectionsStr == INVALID_CELL_VALUE)
            continue;
        // TODO: Really bad solution to find the point by id, but the id is not
        // necessarily the index in the points vector, so we need to find it by name =>
        // refactor the Points structure to use std::map later
        auto from = searchHeatingGridPointById(points, id);
        if (from == nullptr)
        {
            std::stringstream ss;
            ss << "Point with id " << id << " not found in points.";
            m_logger.warn(ss.str());
            continue;
        }
        auto line = createHeatingGridLine(points, from, connectionsStr, additionalData);
        if (line == nullptr)
        {
            m_logger.warn("Failed to create line for point: " + from->getName());
            continue;
        }
        lines.push_back(line);
    }
    return lines;
}

osg::ref_ptr<grid::Point> HeatingGridParser::searchHeatingGridPointById(
    const grid::Points &points, int id)
{
    auto pointIt = std::find_if(points.begin(), points.end(), [id](const auto &p)
        { return std::stoi(p->getName()) == id; });
    if (pointIt == points.end())
    {
        std::stringstream ss;
        ss << "Point with id " << id << " not found in points." << std::endl;
        m_logger.warn(ss.str());
    }
    return *pointIt; // returns nullptr if not found
}

osg::ref_ptr<grid::Line> HeatingGridParser::createHeatingGridLine(
    const grid::Points &points, osg::ref_ptr<grid::Point> from,
    const std::string &connectionsStrWithCommaDelimiter,
    grid::ConnectionDataList &additionalData)
{
    std::string connection("");
    grid::Connections gridConnections;
    auto pointName = from->getName();
    std::string lineName { pointName };
    auto connections = split(connectionsStrWithCommaDelimiter, ' ');
    for (const auto &connection : connections)
    {
        if (connection.empty() || connection == INVALID_CELL_VALUE)
            continue;
        grid::Data connectionData { { "name", pointName + "_" + connection } };
        additionalData.emplace_back(std::vector { connectionData });
        int toID(-1);
        try
        {
            toID = std::stoi(connection);
        }
        catch (...)
        {
            continue;
        }
        lineName += std::string(" ") + CONSTANTS::UIConstants::RIGHT_ARROW_UNICODE_HEX + " " + connection;

        // TODO: Really bad solution to find the point by id, but the id is not
        // necessarily the index in the points vector, so we need to find it by name =>
        // refactor the Points structure to use std::map later
        auto to = searchHeatingGridPointById(points, toID);
        if (to == nullptr)
        {
            std::stringstream ss;
            ss << "Point with id " << toID << " not found in points." << std::endl;
            m_logger.warn(ss.str());
            continue;
        }
        grid::ConnectionData connData {
            pointName + "_" + connection, from, to, 0.5f, true, nullptr, connectionData
        };
        grid::DirectedConnection directed(connData, m_logger,
            grid::ConnectionType::LineWithShader);
        gridConnections.push_back(new grid::DirectedConnection(directed));
    }

    return new grid::Line(lineName, gridConnections, m_logger);
}

grid_ptr HeatingGridParser::operator()(const ArrowDataMap &map)
{
    m_logger.error("HeatingGridParser cannot parse ArrowDataMap at the moment.");
    return nullptr;
}

grid_ptr HeatingGridParser::operator()(const ArrowData &data)
{
    m_logger.error("HeatingGridParser cannot parse ArrowData at the moment.");
    return nullptr;
}

grid_ptr HeatingGridParser::operator()(CSVData &data)
{
    m_logger.error("HeatingGridParser cannot parse CSVData at the moment.");
    return nullptr;
}
