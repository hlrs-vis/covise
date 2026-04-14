
#ifndef COVISE_UTIL_COLOR_MAP_H
#define COVISE_UTIL_COLOR_MAP_H

#include <map>
#include <memory>
#include <osg/Vec4>
#include <string>
#include <vector>


#include "util/coExport.h"

namespace opencover {

struct PLUGIN_UTILEXPORT BaseColorMap {
  std::vector<osg::Vec4> colors;
  std::vector<float> samplingPoints;
  std::string name;
};

typedef std::vector<BaseColorMap> ColorMaps;
PLUGIN_UTILEXPORT ColorMaps &ConfigColorMaps();


class PLUGIN_UTILEXPORT ColorMap {
public:
  ColorMap(const std::string &species, const std::string &unit = std::string(),
    float min = 0.0, float max = 1.0, int steps = 64);
  ColorMap(const BaseColorMap &map, float min = 0.0, float max = 1.0);

  ColorMap(const ColorMap &other) = delete;
  ColorMap &operator=(const ColorMap &other) = delete;
  ColorMap(ColorMap &&other);
  ColorMap &operator=(ColorMap &&other);
  ~ColorMap() = default;

  void setMinMax(float min, float max);
  void setSteps(int steps);
  void setSpecies(const std::string &species);
  void setUnit(const std::string &unit);

  osg::Vec4 getColor(float val) const;
  osg::Vec4 getColorPerStep(int step) const;
  osg::Vec4 getColorPercent(float percentile) const;
  float min() const;
  float max() const;
  int steps() const;

  const std::string &species() const;
  const std::string &unit() const;
  const std::string &name() const;
private:
  float m_min = 0.0, m_max = 1.0;
  int m_steps = 32;
  std::string m_species = "default";
  std::string m_unit;
  BaseColorMap m_interpolatedMap;
  BaseColorMap *m_colorMap = nullptr;

};

PLUGIN_UTILEXPORT ColorMaps readColorMaps();
// osg::Vec4 PLUGIN_UTILEXPORT getColor(float val, const ColorMap &colorMap);
//                                                 RotationType type = HPR);

// class PLUGIN_UTILEXPORT ColorMapSelector {
//  public:
//   ColorMapSelector(opencover::ui::Menu &menu);
//   ColorMapSelector(opencover::ui::Group &menu);

//   bool setValue(const std::string &colorMapName);
//   osg::Vec4 getColor(float val);
//   const ColorMap &selectedMap() const;
//   void setCallback(const std::function<void(const ColorMap &)> &f);
//   void showHud(bool show);
//   bool hudVisible() const;
//   void setHudPosition(const opencover::ColorBar::HudPosition &position);
//   void setUnit(const std::string &unit);
//   void setSpecies(const std::string &species);
//   void setMin(float min);
//   void setMax(float max);
//   void setMinBounds(float min, float max);
//   void setMaxBounds(float min, float max);
//   void setName(const std::string &name);
//  private:
//   opencover::ui::SelectionList *m_selector;
//   ColorMaps m_colors;
//   ColorMaps::iterator m_selectedMap;
//   std::unique_ptr<opencover::ColorBar> m_colorBar;
//   std::string m_unit, m_species;
//   void updateSelectedMap();
//   void init();
// };


}  // namespace opencover

#endif
