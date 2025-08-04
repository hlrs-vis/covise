#include "EnergyGrid.h"

#include <PluginUtil/colors/ColorBar.h>
#include <PluginUtil/colors/coColorMap.h>
#include <cover/coVRSelectionManager.h>
#include <lib/core/constants.h>
// #include <lib/core/simulation/grid.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

#include <cassert>
#include <memory>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osgText/Text>
#include <sstream>
#include <utility>
#include <variant>

#include "cover/VRViewer.h"

namespace {

auto get_string = [](const auto &data) {
  std::stringstream ss;
  ss << data << "\n\n";
  return ss.str();
};

}  // namespace

InfoboardSensor::InfoboardSensor(
    osg::ref_ptr<osg::Group> parent,
    std::unique_ptr<interface::IInfoboard<std::string>> &&infoboard,
    const std::string &content)
    : coPickSensor(parent), m_enabled(false), m_infoBoard(std::move(infoboard)) {
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  m_infoBoard->updateInfo(content);

  parent->addChild(m_infoBoard->getDrawable());
}

void InfoboardSensor::activate() {
  auto selectionManager = opencover::coVRSelectionManager::instance();
  selectionManager->clearSelection();
  auto selectedNode = getNode();
  if (!selectedNode) {
    std::cerr << "InfoboardSensor: No node selected for activation." << std::endl;
    return;
  }
  auto parent = selectedNode->getParent(0);
  if (!parent) {
    std::cerr << "InfoboardSensor: No parent node found for selected node."
              << std::endl;
    return;
  }

  constexpr float R = 173.0f / 255.0f;
  constexpr float G = 216.0f / 255.0f;
  constexpr float B = 230.0f / 255.0f;
  selectionManager->setSelectionColor(R, G, B);

  if (!m_enabled) {
    m_infoBoard->showInfo();
    m_enabled = true;
    selectionManager->addSelection(parent, getNode());
  } else {
    m_infoBoard->hideInfo();
    m_enabled = false;
  }

  coPickSensor::activate();
}

void InfoboardSensor::update() {
  updateDrawable();
  coPickSensor::update();
}

// EnergyGrid::EnergyGrid(EnergyGridConfig &&data) : m_config(std::move(data)) {
EnergyGrid::EnergyGrid(const EnergyGridConfig &data, bool ignoreOverlap)
    : m_config(data), m_ignoreOverlap(ignoreOverlap) {
  if (!m_config.parent.valid()) {
    m_config.parent = new osg::MatrixTransform;
    m_config.parent->setName(m_config.name);
  }
  initConnections();
};

void EnergyGrid::initConnections() {
  assert(m_config.valid() && "EnergyGridConfig is not valid");
  if (m_config.connectionType == EnergyGridConnectionType::Index)
    initConnectionsByIndex(m_config.indices, m_config.connectionRadius,
                           m_config.additionalConnectionData);
}

void EnergyGrid::initConnectionsByIndex(
    const grid::Indices &indices, const float &radius,
    const grid::ConnectionDataList &additionalConnectionData) {
  bool hasAdditionalData = !additionalConnectionData.empty();

  const auto &points = m_config.points;
  for (auto i = 0; i < indices.size(); ++i) {
    auto from = points[i];
    for (auto j = 0; j < indices[i].size(); ++j) {
      if (i < 0 || i >= points.size()) {
        std::cerr << "Invalid Index for points: " << i << "\n";
        continue;
      }

      const auto indice = indices[i][j];

      if (indice >= points.size() || indice < 0) {
        std::cerr << "Invalid Index for points: " << indice << "\n";
        continue;
      }
      auto to = points[indice];

      std::string name(from->getName() + " " + UIConstants::RIGHT_ARROW_UNICODE_HEX +
                       " " + to->getName());

      grid::Data additionalData{};
      if (hasAdditionalData)
        if (additionalConnectionData.size() > i)
          if (additionalConnectionData[i].size() > j)
            additionalData = additionalConnectionData[i][j];
      grid::ConnectionData data{name,    from,          to, radius, false,
                                nullptr, additionalData};
      m_connections.push_back(new grid::DirectedConnection(data));
    }
  }
}

void EnergyGrid::findCorrectHeightForLine(float radius,
                                          osg::ref_ptr<grid::Line> line,
                                          grid::Lines &processedLines) {
  if (m_ignoreOverlap) return;
  int redundantCount = 0;
  bool overlap = true;

  std::set<osg::ref_ptr<grid::Line>> lastMatch;
  while (overlap) {
    overlap = false;  // Reset overlap flag for each iteration

    for (auto otherLine : processedLines) {
      if (line == otherLine) continue;  // Skip comparing the line with itself.
      if (lastMatch.find(otherLine) != lastMatch.end())
        continue;  // Skip already checked line
      if (!line.valid() || !otherLine.valid()) continue;
      if (line->overlap(*otherLine)) {
        line->move(osg::Vec3(0, 0, -2 * radius * redundantCount));
        ++redundantCount;
        overlap = true;  // Set overlap flag to repeat the loop
        lastMatch.insert(
            otherLine);  // Store the last line to avoid redundant checks
        break;           // No need to check other lines in this iteration
      }
    }
  }
}

void EnergyGrid::initDrawableLines() {
  using namespace grid;
  osg::ref_ptr<osg::Group> linesGroup = new osg::Group;
  linesGroup->setName("Lines");
  const auto &sphereRadius =
      m_config.lines[0]->getConnections().begin()->second->getStart()->getRadius();
  grid::Lines overlappingLines;

  for (auto line : m_config.lines) {
    // move redundant line below the first one
    findCorrectHeightForLine(sphereRadius, line, overlappingLines);
    overlappingLines.push_back(line);
    initDrawableGridObject(linesGroup, line);
  }
  m_config.parent->addChild(linesGroup);
}

std::string EnergyGrid::createDataString(const grid::Data &data) const {
  std::string result;
  for (const auto &[key, value] : data) {
    result += UIConstants::TAB_SPACES + key + ": " + std::visit(get_string, value);
  }
  return result;
}

void EnergyGrid::initDrawablePoints() {
  osg::ref_ptr<osg::Group> points = new osg::Group;
  points->setName("Points");
  if (!m_config.points.empty()) {
    for (auto &point : m_config.points) {
      initDrawableGridObject(points, point);
    }
  } else {
    for (auto &[id, point] : m_config.pointsMap) {
      if (point.valid()) {
        initDrawableGridObject(points, point);
      }
    }
  }

  m_config.parent->addChild(points);
}

osg::ref_ptr<grid::DirectedConnection> EnergyGrid::getConnectionByName(
    const std::string &name) {
  for (auto &connection : m_connections)
    if (connection->getName() == name) return connection;
  return nullptr;
}

const osg::ref_ptr<grid::Point> EnergyGrid::getPointByName(
    const std::string &name) const {
  // TODO: need to rework this after workshop => for now it works
  for (auto &point : m_config.points)
    if (point->getName() == name) return point;
  for (auto &[_, point] : m_config.pointsMap)
    if (point->getName() == name) return point;
  return nullptr;
}

void EnergyGrid::initDrawableConnections() {
  osg::ref_ptr<osg::Group> connections = new osg::Group;
  connections->setName("Connections");

  for (auto connection : m_connections) {
    initDrawableGridObject(connections, connection);
  }

  m_config.parent->addChild(connections);
}

void EnergyGrid::initDrawables() {
  switch (m_config.connectionType) {
    case EnergyGridConnectionType::Index:
      initDrawableConnections();
      break;
    case EnergyGridConnectionType::Line:
      initDrawableLines();
      break;
    default:
      std::cerr << "Invalid connection type\n";
  }
  initDrawablePoints();
}

void EnergyGrid::updateColor(const osg::Vec4 &color) {
  for (auto &connection : m_connections)
    utils::color::overrideGeodeColor(connection->getGeode(), color);
  for (auto &point : m_config.points)
    utils::color::overrideGeodeColor(point->getGeode(), color);
  for (auto &[_, point] : m_config.pointsMap)
    utils::color::overrideGeodeColor(point->getGeode(), color);
}

void EnergyGrid::updateDrawables() {
  for (auto &infoboard : m_infoboards) {
    infoboard->updateDrawable();
  }
}
// toDo: streamline update for m_connections, m_lines and m_config.lines
void EnergyGrid::updateTime(int timestep) {
  for (auto point : m_config.points) point->updateTimestepInShader(timestep);

  for (auto &[_, point] : m_config.pointsMap)
    point->updateTimestepInShader(timestep);

  for (auto &conn : m_connections) conn->updateTimestepInShader(timestep);
  for (auto &line : m_lines)
    for (auto &[_, conn] : line->getConnections())
      conn->updateTimestepInShader(timestep);
  for (auto &line : m_config.lines)
    for (auto &[_, conn] : line->getConnections())
      conn->updateTimestepInShader(timestep);
}

void EnergyGrid::setColorMap(const opencover::ColorMap &colorMap, const opencover::ColorMap &vm_pu_colormap) {
  for (auto &point : m_config.points) point->updateColorMapInShader(colorMap);

  for (auto &[_, point] : m_config.pointsMap)
    point->updateColorMapInShader(vm_pu_colormap);

  for (auto &conn : m_connections) conn->updateColorMapInShader(colorMap);
  for (auto &line : m_lines)
    for (auto &[_, conn] : line->getConnections())
      conn->updateColorMapInShader(colorMap);
  for (auto &line : m_config.lines)
    for (auto &[_, conn] : line->getConnections())
      conn->updateColorMapInShader(colorMap);
}

void EnergyGrid::setData(const core::simulation::Simulation &sim,
                         const std::string &species, bool interpolate) {
  for (auto &point : m_config.points) {
    auto data = sim.getTimedependentScalar(species, point->getName());
    auto [min, max] = sim.getMinMax(species);
    if (data) {
      point->updateDataInShader(*data, min, max);
    } else {
      std::cerr << "No data found for point: " << point->getName() << "\n";
    }
  }
  for (auto &[_, point] : m_config.pointsMap) {
    // TODO: remove this later => workaround for workshop
    // Make selector a buttongroupd which allows to select multiple species
    auto data = sim.getTimedependentScalar("vm_pu", point->getName());
    auto [min, max] = sim.getMinMax("vm_pu");
    // auto data = sim.getTimedependentScalar(species, point->getName());
    // auto [min, max] = sim.getMinMax(species);
    if (data) {
      point->updateDataInShader(*data, min, max);
    } else {
      std::cerr << "No data found for point: " << point->getName() << "\n";
    }
  }
  for (auto &conn : m_connections) {
    auto fromData = sim.getTimedependentScalar(species, conn->getStart()->getName());
    auto toData = fromData;
    if (interpolate)
      toData = sim.getTimedependentScalar(species, conn->getEnd()->getName());
    if (fromData && toData) {
      conn->setDataInShader(*fromData, *toData);
    } else {
      std::cerr << "No data found for connection: " << conn->getName() << "\n";
    }
  }
  for (auto &line : m_lines) {
    for (auto &[_, conn] : line->getConnections()) {
      auto fromData =
          sim.getTimedependentScalar(species, conn->getStart()->getName());
      auto toData = fromData;
      if (interpolate)
        toData = sim.getTimedependentScalar(species, conn->getEnd()->getName());
      if (fromData && toData) {
        conn->setDataInShader(*fromData, *toData);
      } else {
        std::cerr << "No data found for connection: " << conn->getName() << "\n";
      }
    }
  }
  for (auto &line : m_config.lines) {
    if (interpolate) {
      for (auto &[_, conn] : line->getConnections()) {
        auto fromData =
            sim.getTimedependentScalar(species, conn->getStart()->getName());
        auto toData = sim.getTimedependentScalar(species, conn->getEnd()->getName());
        if (fromData && toData) {
          conn->setDataInShader(*fromData, *toData);
        } else {
          std::cerr << "No data found for connection: " << conn->getName() << "\n";
        }
      }
    } else {
      // TODO: If not interpolating, use the line name to get the data => pls rework
      // later this is a workaround for the current data structure
      auto lineName = line->getName();
      std::replace(lineName.begin(), lineName.end(), ' ', '_');
      auto data = sim.getTimedependentScalar(species, lineName);
      const auto [min, max] = sim.getMinMax(species);
      std::cout << "Min: " << min << ", Max: " << max << "\n";
      if (!data) {
        std::cerr << "No data found for line: " << lineName << "\n";
        continue;
      }
      for (auto &[_, conn] : line->getConnections())
        // conn->setDataInShader(*data, *data);
        conn->setData1DInShader(*data, min, max);
    }
  }
}
