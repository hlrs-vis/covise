#include "Parser.h"

std::unique_ptr<core::interface::IEnergyGrid> PowerGridParser::operator()(const CSVDataMap &map)
{
//   using grid::Point;
//   if (m_powerGridStreams.empty()) return nullptr;

//   constexpr float connectionsRadius(1.0f);
//   constexpr float sphereRadius(2.0f);
//   size_t numPoints(0);
//   // fetch bus names
//   auto busData = m_powerGridStreams.find("bus");
//   std::vector<IDLookupTable> busNames;
//   if (busData != m_powerGridStreams.end()) {
//     auto &[name, busStream] = *busData;
//     busNames = retrieveBusNameIdMapping(busStream);
//   }

//   if (busNames.empty()) return nullptr;

//   // create points
//   auto pointsData = m_powerGridStreams.find("bus_geodata");
//   std::vector<grid::PointsMap> points;
//   if (pointsData != m_powerGridStreams.end()) {
//     auto &[name, pointStream] = *pointsData;
//     points = createPowerGridPoints(pointStream, numPoints, sphereRadius, busNames);
//   }

//   // create line
//   auto lineData = m_powerGridStreams.find("line");
//   std::vector<grid::Lines> lines;
//   std::vector<grid::ConnectionDataList> optData;
//   if (lineData != m_powerGridStreams.end()) {
//     auto &[name, lineStream] = *lineData;
//     std::tie(lines, optData) = getPowerGridLines(lineStream, points);
//   }

//   // create grid
//   if (lines[0].empty() || lines[1].empty() || points.empty()) return nullptr;

//   grid::PointsMap mergedPoints = points[0];
//   mergedPoints.insert(points[1].begin(), points[1].end());
//   // TODO: workaround for merging => PLS REFACTOR LATER
//   grid::Lines mergedLines = lines[0];
//   mergedLines.insert(mergedLines.end(), lines[1].begin(), lines[1].end());

//   grid::ConnectionDataList mergedOptData = optData[0];
//   mergedOptData.insert(mergedOptData.end(), optData[1].begin(), optData[1].end());

// //   auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
// //   auto &egrid = m_energyGrids[idx];
//   osg::ref_ptr<osg::MatrixTransform> powerGroup = new osg::MatrixTransform;
//   powerGroup = new osg::MatrixTransform;
//   auto font = m_plugin->configString("Billboard", "font", "default")->value();
//   OsgTxtBoxAttributes infoboardAttributes = OsgTxtBoxAttributes(
//       {0, 0, 0}, "EnergyGridText", font, 50, 50, 2.0f, 0.1, 2);
//   powerGroup->setName("PowerGrid");

//   EnergyGridConfig econfig("POWER", {}, grid::Indices(), mergedPoints, powerGroup,
//                            connectionsRadius, mergedOptData, infoboardAttributes,
//                            EnergyGridConnectionType::Line, mergedLines);

//   auto powerGrid = std::make_unique<EnergyGrid>(econfig, getLogger(), false);
//   powerGrid->initDrawable();
//   egrid.grid = std::move(powerGrid);
//   addEnergyGridToGridSwitch(egrid.group);

//   // TODO:
//   //  [ ] set trafo as 3d model or block

//   // how to implement this generically?
//   // - fixed grid structure for discussion in AK Software
//   // - look into Energy ADE
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> PowerGridParser::operator()(const ArrowDataMap &map)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> PowerGridParser::operator()(const ArrowData &data)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> PowerGridParser::operator()(const CSVData &data)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> HeatingGridParser::operator()(const CSVDataMap &map)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> HeatingGridParser::operator()(const ArrowDataMap &map)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> HeatingGridParser::operator()(const ArrowData &data)
{
    return nullptr;
}

std::unique_ptr<core::interface::IEnergyGrid> HeatingGridParser::operator()(const CSVData &data)
{
    return nullptr;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const CSVDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const CSVData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const CSVDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const CSVData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}
