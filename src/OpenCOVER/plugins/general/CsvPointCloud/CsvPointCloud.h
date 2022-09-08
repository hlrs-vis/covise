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
#include <exprtk.hpp>
#include <array>

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
  osg::Geometry *m_pointCloud = nullptr;
  osg::Geode *m_currentGeode = nullptr;
  ui::Menu *m_CsvPointCloudMenu, *m_colorMenu;
  std::array<ui::EditField*, 3> m_coordTerms;
  ui::EditField *m_colorTerm, *m_timeScaleIndicator, *m_delimiter, *m_offset;
  covise::ColorMapSelector m_colorMapSelector;
  ui::Slider *m_pointSizeSlider;
  ui::Slider *m_animationSpeedMulti;
  ui::Action *m_reloadBtn;
  ui::Group *m_colorsGroup;
  opencover::ColorBar *m_colorBar;
  void createGeodes(osg::Group *, const std::string &);
  osg::Geometry *createOsgPoints(DataTable &symbols);
  int unloadFile();
  bool compileSymbol(DataTable &symbols, const std::string &symbol, Expression &expr);
};

#endif // COVER_PLUGIN_OCT_H
