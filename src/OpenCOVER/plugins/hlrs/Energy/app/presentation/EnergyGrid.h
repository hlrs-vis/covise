#ifndef _CORE_ENERGYGRIND_H
#define _CORE_ENERGYGRIND_H

#include "grid.h"
#include <PluginUtil/colors/coColorMap.h>
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/interfaces/IInfoboard.h>
// #include <lib/core/simulation/grid.h>


#include <memory>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/ref_ptr>

#include "PluginUtil/coSensor.h"
#include "TxtInfoboard.h"

using namespace core;
// using namespace core::simulation;

enum class EnergyGridConnectionType { Index, Line };

/**
 * @struct EnergyGridConfig
 * @brief A struct representing the data needed to create an energy grid.
 *
 * This struct is used to store the data needed to create an energy grid.
 *
 * @param name The name of the energy grid.
 * @param points The points that define the grid.
 * @param indices The indices that define the connections between points.
 * @param parent The parent OSG group node (default is nullptr).
 * @param connectionRadius The radius for connections (default is 1.0f).
 * @param additionalConData Additional connection data (default is an empty list).
 */
struct EnergyGridConfig {
  //   EnergyGridConfig(const std::string &gridName, const grid::Points &gridPoints,
  EnergyGridConfig(const std::string &gridName, const grid::Points &gridPoints,
                   const grid::Indices &gridIndices,
                   const grid::PointsMap &gridPointsMap = {},
                   osg::ref_ptr<osg::MatrixTransform> gridParent = nullptr,
                   const float &gridConnectionRadius = 1.0f,
                   const grid::ConnectionDataList &extraConnectionData =
                       grid::ConnectionDataList(),
                   const TxtBoxAttributes &gridInfoAttributes =
                       TxtBoxAttributes(osg::Vec3(0, 0, 0), "EnergyGridText",
                                        "DejaVuSans-Bold.ttf", 50, 50, 2.0f, 0.1, 2),
                   const EnergyGridConnectionType &gridConnectionType =
                       EnergyGridConnectionType::Index,
                   const grid::Lines &gridLines = grid::Lines())
      : name(gridName),
        points(gridPoints),
        indices(gridIndices),
        pointsMap(gridPointsMap),
        parent(gridParent),
        connectionRadius(gridConnectionRadius),
        additionalConnectionData(extraConnectionData),
        infoboardAttributes(gridInfoAttributes),
        connectionType(gridConnectionType),
        lines(gridLines) {}

  // mandatory
  std::string name;
  grid::Points points;
  grid::Indices indices;
  // optional
  grid::PointsMap pointsMap;  // for faster access
  osg::ref_ptr<osg::MatrixTransform> parent;
  float connectionRadius;
  grid::ConnectionDataList additionalConnectionData;
  TxtBoxAttributes infoboardAttributes;
  EnergyGridConnectionType connectionType;
  grid::Lines lines;

  bool valid() const {
    // bool isMandatoryValid = !name.empty() || (!points.empty() &&
    // pointsMap.empty()) || !indices.empty();
    bool isMandatoryValid = !name.empty() ||
                            ((points.empty() || pointsMap.empty()) &&
                             (points.empty() && pointsMap.empty())) ||
                            !indices.empty();
    return connectionType == EnergyGridConnectionType::Index ? isMandatoryValid
                                                             : !lines.empty();
  }
};

class InfoboardSensor : public coPickSensor {
 public:
  InfoboardSensor(osg::ref_ptr<osg::Group> parent,
                  std::unique_ptr<interface::IInfoboard<std::string>> &&infoboard,
                  const std::string &content = "");

  void updateDrawable() { m_infoBoard->updateDrawable(); }
  void activate() override;
  void update() override;

 private:
  bool m_enabled = false;
  std::unique_ptr<interface::IInfoboard<std::string>> m_infoBoard;
};

/**
 * @class EnergyGrid
 * @brief A class representing an energy grid, inheriting from interface::EnergyGrid.
 *
 * This class is responsible for managing and visualizing an energy grid using
 * OpenSceneGraph.
 *
 */
class EnergyGrid : public interface::IEnergyGrid {
 public:
  //   EnergyGrid(EnergyGridConfig &&data);
  EnergyGrid(const EnergyGridConfig &data, bool ignoreOverlap = true);
  void initDrawables() override;
  void update() override {
    for (auto &infoboard : m_infoboards) infoboard->update();
  }
  void updateColor(const osg::Vec4 &color) override;
  void updateDrawables() override;
  void updateTime(int timestep) override;

//   void setColorMap(const opencover::ColorMap &colorMap) override;
//   TODO: remove this later => what a fucking mess
//   HACK: bullshit code
  void setColorMap(const opencover::ColorMap &colorMap, const opencover::ColorMap &vm_pu_Colormap) override;
  void setData(const core::simulation::Simulation& sim, const std::string & species, bool interpolate = false) override;
  osg::ref_ptr<grid::DirectedConnection> getConnectionByName(
      const std::string &name);
  osg::ref_ptr<grid::DirectedConnection> getConnectionByIdx(int idx) {
    if (idx < 0 || idx >= m_connections.size()) return nullptr;
    return m_connections[idx];
  }
  const osg::ref_ptr<grid::Point> getPointByName(const std::string &name) const;
  osg::ref_ptr<grid::Point> getPointByIdx(int idx) {
    if (idx < 0 || idx >= m_config.points.size()) return nullptr;
    return m_config.points[idx];
  }

 private:
  template <typename T>
  void initDrawableGridObject(osg::ref_ptr<osg::Group> parent, const T &gridObj) {
    m_drawables.push_back(gridObj);
    parent->addChild(gridObj);
    std::string toPrint(createDataString(gridObj->getAdditionalData()));
    auto center = gridObj->getCenter();
    center.z() += 30;
    auto name = gridObj->getName();

    m_config.infoboardAttributes.position = center;
    m_config.infoboardAttributes.title = name;
    TxtInfoboard infoboard(m_config.infoboardAttributes);
    m_infoboards.push_back(std::make_unique<InfoboardSensor>(
        gridObj, std::make_unique<TxtInfoboard>(infoboard), toPrint));
  }

  std::string createDataString(const grid::Data &data) const;

  void initConnections();
  void initConnectionsByIndex(
      const grid::Indices &indices, const float &radius,
      const grid::ConnectionDataList &additionalConnectionData);
  void initDrawableConnections();
  void initDrawableLines();
  void initDrawablePoints();
  bool validPointIdx(int idx) { return idx < 0 || idx >= m_config.points.size(); }
  void findCorrectHeightForLine(float radius, osg::ref_ptr<grid::Line> line,
                                grid::Lines &lines);

  EnergyGridConfig m_config;
  grid::Connections m_connections;
  grid::Lines m_lines;
  std::vector<std::unique_ptr<InfoboardSensor>> m_infoboards;
  bool m_ignoreOverlap;
};
#endif
