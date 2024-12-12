#ifndef _CORE_ENERGYGRIND_H
#define _CORE_ENERGYGRIND_H

#include <memory>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/ref_ptr>

#include "PluginUtil/coSensor.h"
#include "TxtInfoboard.h"
#include "grid.h"
#include "interfaces/IEnergyGrid.h"
#include "interfaces/IInfoboard.h"

namespace core {

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
  // mandatory
  std::string name;
  grid::Points points;
  grid::Indices indices;
  // optional
  osg::ref_ptr<osg::Group> parent = nullptr;
  float connectionRadius = 1.0f;
  grid::DataList additionalConnectionData = grid::DataList();
  TxtBoxAttributes infoboardAttributes =
      TxtBoxAttributes(osg::Vec3(0, 0, 0), "EnergyGridText", "DroidSans-Bold.ttf",
                       50, 50, 2.0f, 0.1, 2);
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
  EnergyGrid(EnergyGridConfig &&data);
  void initDrawables() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateDrawables() override;

 private:
  class InfoboardSensor : public coPickSensor {
   public:
    InfoboardSensor(osg::ref_ptr<osg::Group> parent,
                    std::unique_ptr<interface::IInfoboard<std::string>> &&infoboard,
                    const std::string &content = "");

    void updateDrawable() { m_infoBoard->updateDrawable(); }
    int hit(vrui::vruiHit *hit) override;

    void update() override {
      updateDrawable();
      coPickSensor::update();
    }

    interface::IInfoboard<std::string> *getInfoboard() { return m_infoBoard.get(); }

   private:
    std::unique_ptr<interface::IInfoboard<std::string>> m_infoBoard;
  };

  void initConnections(const grid::Indices &indices, const float &radius,
                       const grid::DataList &additionalConnectionData);
  void initDrawableConnections();
  void initDrawablePoints();
  EnergyGridConfig m_config;
  grid::Connections m_connections;
  std::vector<std::unique_ptr<InfoboardSensor>> m_infoboards;
};
}  // namespace core
#endif
