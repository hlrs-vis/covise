#include "PowerGridParser.h"
#include "app/osg/presentation/EnergyGrid.h"

#include <filesystem>

using namespace opencover::utils::read;

grid_ptr PowerGridParser::operator()(CSVDataMap &map)
{
    using grid::Point;
    if (map.empty())
        return nullptr;

    size_t numPoints(0);

    // fetch bus names
    auto busData = map.find("bus");
    std::vector<IDLookupTable> busNames;
    if (busData != map.end())
    {
        auto &[name, busStream] = *busData;
        busNames = retrieveBusNameIdMapping(*busStream);
    }

    if (busNames.empty())
        return nullptr;

    // create points
    auto pointsData = map.find("bus_geodata");
    std::vector<grid::PointsMap> points;
    if (pointsData != map.end())
    {
        auto &[name, pointStream] = *pointsData;
        points = createPowerGridPoints(*pointStream, numPoints, m_config.sphereRadius, busNames, map);
    }

    // create line
    auto lineData = map.find("line");
    std::vector<grid::Lines> lines;
    std::vector<grid::ConnectionDataList> optData;
    if (lineData != map.end())
    {
        auto &[name, lineStream] = *lineData;
        std::tie(lines, optData) = getPowerGridLines(*lineStream, points);
    }

    // create grid
    if (lines[0].empty() || lines[1].empty() || points.empty())
        return nullptr;

    grid::PointsMap mergedPoints = points[0];
    mergedPoints.insert(points[1].begin(), points[1].end());
    // TODO: workaround for merging => PLS REFACTOR LATER
    grid::Lines mergedLines = lines[0];
    mergedLines.insert(mergedLines.end(), lines[1].begin(), lines[1].end());

    grid::ConnectionDataList mergedOptData = optData[0];
    mergedOptData.insert(mergedOptData.end(), optData[1].begin(), optData[1].end());

    auto powerGroup = m_config.parent;
    OsgTxtBoxAttributes infoboardAttributes = OsgTxtBoxAttributes(
        { 0, 0, 0 }, "EnergyGridText", m_config.font, 50, 50, 2.0f, 0.1, 2);
    powerGroup->setName("PowerGrid");

    EnergyGridConfig econfig;
    econfig.name = "POWER";
    econfig.pointsMap = mergedPoints;
    econfig.parent = powerGroup;
    econfig.connectionRadius = m_config.connectionsRadius;
    econfig.additionalConnectionData = mergedOptData;
    econfig.infoboardAttributes = infoboardAttributes;
    econfig.connectionType = EnergyGridConnectionType::Line;
    econfig.lines = mergedLines;

    auto powerGrid = std::make_shared<EnergyGrid>(econfig, getLogger(), false);
    powerGrid->initDrawable();
    return std::move(powerGrid);
}

std::vector<PowerGridParser::IDLookupTable>
PowerGridParser::retrieveBusNameIdMapping(opencover::utils::read::CSVStream &stream)
{
    auto busNames = IDLookupTable();
    auto busNamesSonder = IDLookupTable();
    CSVStream::CSVRow bus;
    std::string busName(""), type("");
    int id = 0;
    while (stream.readNextRow(bus))
    {
        ACCESS_CSV_ROW(bus, "name", busName);
        ACCESS_CSV_ROW(bus, "id", id);
        if (bus.find("grid") != bus.end())
            ACCESS_CSV_ROW(bus, "grid", type);
        else
            type = "Normalnetz"; // default type if not specified

        if (type == "Sondernetz")
        {
            busNamesSonder.insert({ id, busName });
            continue;
        }
        busNames.insert({ id, busName });
    }
    return { busNames, busNamesSonder };
}

std::vector<grid::PointsMap> PowerGridParser::createPowerGridPoints(
    CSVStream &stream, size_t &numPoints, const float &sphereRadius,
    const std::vector<IDLookupTable> &busNames, CSVDataMap &map)
{
    using PointsMap = grid::PointsMap;

    CSVStream::CSVRow point;
    float lat = 0, lon = 0;
    PointsMap points;
    PointsMap pointsSonder;
    std::string busName = "", type = "";
    int busID = 0;

    // TODO: need to be adjusted
    auto additionalData = getAdditionalPowerGridPointData(numPoints, map);

    while (stream.readNextRow(point))
    {
        ACCESS_CSV_ROW(point, "x", lon);
        ACCESS_CSV_ROW(point, "y", lat);
        ACCESS_CSV_ROW(point, "id", busID);

        // x = lon, y = lat
        lon += m_config.offset[0];
        lat += m_config.offset[1];
        auto height = m_config.offset[2];

        int i = 0;
        for (const auto &busNames : busNames)
        {
            if (auto it = busNames.find(busID); it != busNames.end())
            {
                if (i == 0)
                    type = "Normalnetz";
                else
                    type = "Sondernetz";
                busName = it->second;
                break;
            }
            else
            {
                busName = busName = "Base_" + std::to_string(busID);
            }
            ++i;
        }

        grid::Data busData;
        try
        {
            busData = additionalData->at(busID);
        }
        catch (const std::out_of_range &)
        {
            busData["base_point_data"] = "";
        }

        osg::ref_ptr<grid::Point> p = new grid::Point(busName, lon, lat, height, sphereRadius, getLogger(), busData);
        if (type == "Sondernetz")
            pointsSonder.insert({ busID, p });
        else
            points.insert({ busID, p });
        ++numPoints;
    }
    return { points, pointsSonder };
}

void PowerGridParser::helper_getAdditionalPowerGridPointData_addData(
    int busId, grid::PointDataList &additionalData, const grid::Data &data)
{
    if (busId == -1)
        return;
    auto &existingDataMap = additionalData[busId];
    if (existingDataMap.empty())
        additionalData[busId] = data;
    else
        existingDataMap.insert(data.begin(), data.end());
}

void PowerGridParser::helper_getAdditionalPowerGridPointData_handleDuplicate(
    std::string &name, std::map<std::string, uint> &duplicateMap)
{
    if (auto it = duplicateMap.find(name); it != duplicateMap.end())
        // if there is a similar entity, add the id to the name
        name = name + "_" + std::to_string(++it->second);
    else
        duplicateMap.insert({ name, 0 });
}

std::unique_ptr<grid::PointDataList>
PowerGridParser::getAdditionalPowerGridPointData(const std::size_t &numOfBus, CSVDataMap &map)
{
    using PDL = grid::PointDataList;

    // additional bus data
    PDL additionalData;

    for (auto &[tableName, tableStream] : map)
    {
        auto header = tableStream->getHeader();
        if (auto it = std::find(header.begin(), header.end(), "bus"); it == header.end())
            continue;
        auto it = std::find(header.begin(), header.end(), "bus");
        if (it == header.end())
            CSVStream::CSVRow busdata;
        int busId = -1;
        std::map<std::string, uint> duplicate {};
        CSVStream::CSVRow row;
        // row
        while (tableStream->readNextRow(row))
        {
            grid::Data data;
            // column
            for (auto &colName : header)
            {
                // TODO: move this into GridUIManager
                //  if (!checkBoxSelection_powergrid(tableName, colName))
                //      continue;
                //  get bus id without adding it
                if (colName == "bus")
                {
                    ACCESS_CSV_ROW(row, colName, busId);
                    continue;
                }
                std::string value;
                ACCESS_CSV_ROW(row, colName, value);

                // add the name of the table to the name
                std::string columnNameWithTable = tableName + " > " + colName;
                helper_getAdditionalPowerGridPointData_handleDuplicate(columnNameWithTable,
                    duplicate);
                data[columnNameWithTable] = value;
            }
            helper_getAdditionalPowerGridPointData_addData(busId, additionalData, data);
        }
    }
    return std::make_unique<PDL>(additionalData);
}

std::pair<std::vector<grid::Lines>, std::vector<grid::ConnectionDataList>>
PowerGridParser::getPowerGridLines(opencover::utils::read::CSVStream &stream, const std::vector<grid::PointsMap> &points)
{
    using Lines = grid::Lines;
    using CDL = grid::ConnectionDataList;
    Lines lines;
    CDL additionalData(points[0].size());
    Lines linesSonder;
    CDL additionalDataSonder(points[1].size());

    CSVStream::CSVRow row;
    int from = 0, to = 0;
    std::string geoBuses = "";
    std::string name = "", type = "";
    auto header = stream.getHeader();
    while (stream.readNextRow(row))
    {
        grid::Data data;

        for (auto colName : header)
        {
            auto filename = std::filesystem::path(stream.getFilename());
            //   fs::path filename(stream.getFilename());
            auto filename_without_ext = filename.stem().string();
            //   if (!checkBoxSelection_powergrid(filename_without_ext, colName)) continue;
            std::string value;
            ACCESS_CSV_ROW(row, colName, value);
            data[colName] = value;
        }

        ACCESS_CSV_ROW(row, "geo_buses", geoBuses);
        ACCESS_CSV_ROW(row, "from_bus", from);
        ACCESS_CSV_ROW(row, "name", name);
        if (row.find("grid") != row.end())
            ACCESS_CSV_ROW(row, "grid", type);
        else
            type = "Normalnetz"; // default type if not specified

        if (geoBuses.empty())
            continue;
        auto line = createLine(name, from, geoBuses, data, points);
        if (type == "Sondernetz")
        {
            linesSonder.push_back(line);
        }
        else
        {
            lines.push_back(line);
        }
    }

    return std::make_pair<std::vector<Lines>, std::vector<grid::ConnectionDataList>>(
        { lines, linesSonder }, { additionalData, additionalDataSonder });
}

osg::ref_ptr<grid::Line> PowerGridParser::createLine(
    const std::string &name, int &from, const std::string &geoBuses_comma_seperated,
    grid::Data &data, const std::vector<grid::PointsMap> &points)
{
    std::stringstream ss(geoBuses_comma_seperated);
    std::string bus("");

    int from_last = from;

    grid::Connections connections;
    while (std::getline(ss, bus, ','))
    {
        auto to_new = std::stoi(bus);
        if (from_last == to_new)
            continue;

        osg::ref_ptr<grid::Point> fromPoint = nullptr;
        osg::ref_ptr<grid::Point> toPoint = nullptr;
        for (auto points : points)
        {
            auto toIt = points.find(to_new);
            if (!toPoint && toIt != points.end())
                toPoint = toIt->second;

            auto fromIt = points.find(from_last);
            if (!fromPoint && fromIt != points.end())
                fromPoint = fromIt->second;
        }
        if (!fromPoint || !toPoint)
        {
            std::stringstream ss;
            ss << "Invalid bus ID: " << from_last << " or " << to_new << std::endl;
            warn(ss.str());
            continue;
        }

        std::string name = fromPoint->getName() + " > " + toPoint->getName();
        float radius = 0.5f;

        grid::ConnectionData conData { name, fromPoint, toPoint, radius,
            false, nullptr, data };
        connections.push_back(
            new grid::DirectedConnection(conData, getLogger(), grid::ConnectionType::LineWithShader));
        from_last = to_new;
    }
    return new grid::Line(name, connections, getLogger());
}

grid_ptr PowerGridParser::operator()(const ArrowDataMap &map)
{
    return nullptr;
}

grid_ptr PowerGridParser::operator()(const ArrowData &data)
{
    return nullptr;
}

grid_ptr PowerGridParser::operator()(CSVData &data)
{
    return nullptr;
}
