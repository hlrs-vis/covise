/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_PLUGIN_OCT_H
#define COVER_PLUGIN_OCT_H
#include "DataTable.h"
#include <PluginUtil/ColorBar.h>
#include <PluginUtil/coColorMap.h>
#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <exprtk.hpp>
#include <array>
#include <thread>

// This plugin was developed to visualize laser cladding measured by an oct scanner as an animated point cloud.
// Therefore it reads an ".oct" file (a renamed csv file with semicolon delimiter and an 6 line ignored header).
// x,y,z and the point color can be configured in the ui via mathematical expressions that use the variables listed in the ".oct" file

namespace osg
{
  class Group;
}

using namespace opencover;

class CsvPointCloudPlugin : public coVRPlugin, public ui::Owner
{
public:
  CsvPointCloudPlugin();
  ~CsvPointCloudPlugin();
  const CsvPointCloudPlugin *instance() const;
  bool init() override;
  static int load(const char *filename, osg::Group *loadParent, const char *covise_key);
  static int unload(const char *filename, const char *covise_key);
  float pointSize() const;
  void setTimestep(int t) override;

private:
  typedef exprtk::expression<float> expression_t;
  typedef exprtk::parser<float> parser_t;
  struct Expression
  {
    expression_t &operator()() { return expression; }
    expression_t expression;
    parser_t parser;
  };

  static CsvPointCloudPlugin *m_plugin;
  osg::Geometry* m_pointCloud = nullptr;
  osg::Geometry *m_reducedPointCloud = nullptr;
  osg::Geode *m_currentGeode = nullptr;
  osg::MatrixTransform* m_transform = nullptr;
  ui::Menu *m_CsvPointCloudMenu, *m_colorMenu;
  ui::EditField* m_dataScale;
  std::array<ui::EditField*, 3> m_coordTerms;
  ui::EditField *m_colorTerm, *m_timeScaleIndicator, *m_delimiter, *m_offset;
  std::array<ui::EditField*, 3> m_machinePositionsTerms;
  ui::EditField* m_pointReductionCriteria;
  covise::ColorMapSelector m_colorMapSelector;
  ui::Slider *m_pointSizeSlider, *m_numPointsSlider;
  ui::Button *m_applyBtn; //button only to allow sharing
  ui::Group *m_colorsGroup;
  opencover::ColorBar *m_colorBar;
  const std::array<ui::EditField*, 12> m_editFields;
  std::vector<vrml::VrmlSFVec3f> m_machinePositions;
  bool m_animSpeedSet = false, m_animSkipSet = false;
  std::vector<size_t> m_reducedIndices;
  std::unique_ptr<DataTable> m_dataTable;
  time_t m_readSettingsTime = 0;
  std::array<std::string, 3> m_machineSpeedNames{"dx", "dy", "dz"};
  std::array<float, 3> m_currentMachineSpeeds{0, 0, 0};
  std::vector<std::unique_ptr<std::thread>> m_threads;
  const int m_numThreads;
  void createGeodes(osg::Group *, const std::string &);
  void createOsgPoints(DataTable &symbols, std::ofstream& f);
  osg::Geometry* createOsgPoints(osg::Vec3Array* points, osg::Vec4Array* colors);

  std::vector<vrml::VrmlSFVec3f> readMachinePositions(DataTable& symbols);

  int unloadFile(const std::string &filename);
  bool compileSymbol(DataTable &symbols, const std::string &symbol, Expression &expr);
  void readSettings(const std::string& filename);
  void writeSettings(const std::string& filename);
  std::unique_ptr<std::ifstream> cacheFileUpToDate(const std::string& filename);
  void writeCacheFileHeader(std::ofstream& f);
  void addMachineSpeedSymbols(DataTable &symbols, std::array<float, 3> &currentMachineSpeed);
  void resetMachineSpeed(std::array<float, 3> &machineSpeed);

  void advanceMachineSpeed(std::array<float, 3> &machineSpeed, size_t i);
  void updateColorMap(float min, float max);

  struct Colors
  {
    osg::Vec4Array *reduced = nullptr, *other = nullptr;
    float min = 0, max = 0;
  };
  struct Coords
  {
    osg::Vec3Array* reduced = nullptr, *other = nullptr;
  };
  Colors getColors(DataTable &symbols);
  Coords getCoords(DataTable &symbols);
};

#endif // COVER_PLUGIN_OCT_H
