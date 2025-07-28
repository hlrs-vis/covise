#include "coColorMap.h"

#include <config/coConfig.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Slider.h>
#include <cover/VRViewer.h>
#include <cover/VRVruiRenderInterface.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <osg/Array>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Math>
#include <osg/MatrixTransform>
#include <osg/Multisample>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osg/ref_ptr>
#include <osgText/Text>
#include <sstream>

#include "osg/Camera"

using namespace std;
using namespace opencover;

ColorMaps &opencover::ConfigColorMaps()
{
  static ColorMaps maps;
  if (maps.empty())
    maps = readColorMaps();
  return maps;
}

ColorMaps opencover::readColorMaps() {
  // read the name of all colormaps in file

  auto colorMapEntries = covise::coCoviseConfig::getScopeEntries("Colormaps");
  ColorMaps colorMaps;
#ifdef NO_COLORMAP_PARAM
  colorMapEntries["COVISE"];
#else
  // colorMapEntries["Editable"];
#endif

  for (const auto &map : colorMapEntries) {
    string name = "Colormaps." + map.first;

    auto no = covise::coCoviseConfig::getScopeEntries(name).size();
    std::vector<osg::Vec4> colors;
    std::vector<float> samplingPoints;
    // read all sampling points
    float diff = 1.0f / (no - 1);
    float pos = 0;
    for (int j = 0; j < no; j++) {
      string tmp = name + ".Point:" + std::to_string(j);
      osg::Vec4 color;
      color.r() = covise::coCoviseConfig::getFloat("r", tmp, 0);
      color.g() = covise::coCoviseConfig::getFloat("g", tmp, 0);
      color.b() = covise::coCoviseConfig::getFloat("b", tmp, 0);
      color.a() = covise::coCoviseConfig::getFloat("a", tmp, 1);
      colors.push_back(color);
      samplingPoints.push_back(covise::coCoviseConfig::getFloat("x", tmp, pos));
      pos += diff;
    }
    colorMaps.push_back(BaseColorMap{colors, samplingPoints, map.first});
  }
  return colorMaps;
}

ColorMap::ColorMap(const std::string &species, const std::string &unit,
  float min, float max, int steps)
: m_species(species), m_unit(unit), m_min(min), m_max(max), m_steps(steps), m_colorMap(&ConfigColorMaps()[0]) {}

ColorMap::ColorMap(const BaseColorMap &map, float min, float max)
: m_min(min), m_max(max), m_steps(map.colors.size()), m_interpolatedMap(map), m_colorMap(&m_interpolatedMap) {}

ColorMap::ColorMap(ColorMap &&other)
: m_min(std::move(other.m_min)) 
, m_max(std::move(other.m_max)) 
, m_steps(std::move(other.m_steps) )
, m_species(std::move(other.m_species) )
, m_unit(std::move(other.m_unit) )
, m_colorMap(other.m_colorMap)
{
  auto otherMap = &other.m_interpolatedMap;
  m_interpolatedMap = std::move(other.m_interpolatedMap);
  if(other.m_colorMap == otherMap)
  {
    m_colorMap = &m_interpolatedMap;
  }
}

ColorMap &ColorMap::operator=(ColorMap &&other)
{
  m_min = std::move(other.m_min);
  m_max = std::move(other.m_max);
  m_steps = std::move(other.m_steps);
  m_species = std::move(other.m_species);
  m_unit = std::move(other.m_unit);
  auto otherMap = &other.m_interpolatedMap;
  m_interpolatedMap = std::move(other.m_interpolatedMap);
  other.m_colorMap == otherMap ? m_colorMap = &m_interpolatedMap : m_colorMap = other.m_colorMap;
  return *this;
}

void ColorMap::setMinMax(float min, float max)
{
  m_min = min;
  m_max = max;
}

void ColorMap::setSteps(int steps)
{
  m_steps = steps;
}

void ColorMap::setSpecies(const std::string &species)
{
  m_species = species;
}

void ColorMap::setUnit(const std::string &unit)
{
  m_unit = unit;
}

float ColorMap::min() const { return m_min; }
float ColorMap::max() const { return m_max; }
int ColorMap::steps() const { return m_steps; }
const std::string &ColorMap::species() const { return m_species; }
const std::string &ColorMap::unit() const { return m_unit; }
const std::string &ColorMap::name() const { return m_colorMap->name; }

osg::Vec4 ColorMap::getColor(float val) const
{
  val = std::clamp(val, m_min, m_max);
  //transform to [0,1] range
  val = 1 / (m_max - m_min) * (val - m_min);
  return getColorPercent(val);
}

osg::Vec4 ColorMap::getColorPerStep(int step) const
{
  assert(step >= 0 && step < m_steps);
  if(m_colorMap->colors.size() == m_steps)
    return m_colorMap->colors[step];
  return getColorPercent(((float)step / (m_steps - 1)));
}


osg::Vec4 ColorMap::getColorPercent(float percentile) const
{
  auto stepsize = 1.0 / (m_steps - 1);
  if(stepsize > 0) // set to a multiple of stepsize
  {
    auto numSteps = std::floor(percentile / stepsize);
    percentile =numSteps * stepsize;
  }

  const auto &samplingPoints = m_colorMap->samplingPoints;
  auto numBaseColors = samplingPoints.size();
  if(m_steps < numBaseColors)
  {
    //find solution
  }

  //todo: use binary search
  size_t idx = 0;
  for (; idx < numBaseColors && samplingPoints[idx + 1] < percentile; idx++) {
  }
  //interpolate val between two sampling points
  auto d = (percentile - samplingPoints[idx]) /
            (samplingPoints[idx + 1] - samplingPoints[idx]);
  return m_colorMap->colors[idx] * (1 - d) +
         m_colorMap->colors[idx + 1] * d;
}


// osg::Vec4 getSamplingColor(const opencover::ColorMap &colorMap, size_t index) {
//   return osg::Vec4(colorMap.r[index], colorMap.g[index], colorMap.b[index], colorMap.a[index]);
// }


// //when calling this often consider to resample the map first to make color calculation more efficient
// osg::Vec4 opencover::getColor(float val, const opencover::ColorMap &colorMap) {
//   auto min = colorMap.min;
//   auto max = colorMap.max;
//   assert(val >= min && val <= max);
//   val = 1 / (max - min) * (val - min);

//   auto stepsize = 1.0 / colorMap.steps;
//   if(stepsize > 0) // set to a multiple of stepsize
//   {
//     auto numSteps = std::floor((val - min) / stepsize);
//     val =numSteps * stepsize;
//   }

//   if(colorMap.steps < colorMap.numColors())
//   {
//     //find solution
//   }


//   size_t idx = 0;
//   for (; idx < colorMap.samplingPoints.size() &&
//          colorMap.samplingPoints[idx + 1] < val;
//        idx++) {
//   }
//   auto d = (val - colorMap.samplingPoints[idx]) /
//             (colorMap.samplingPoints[idx + 1] - colorMap.samplingPoints[idx]);
//   return getSamplingColor(colorMap, idx) * (1 - d) + getSamplingColor(colorMap, idx + 1) * d;
// }

// opencover::ColorMapSelector::ColorMapSelector(opencover::ui::Group &group)
//     : m_selector(new opencover::ui::SelectionList(&group, "mapChoice"))
//     , m_colors(readColorMaps())
//     , m_colorBar(std::make_unique<opencover::ColorBar>(&group)) {
//   init();
// }

// opencover::ColorMapSelector::ColorMapSelector(opencover::ui::Menu &menu)
//     : m_selector(new opencover::ui::SelectionList{&menu, "mapChoice"})
//     , m_colors(readColorMaps())
//     , m_colorBar(std::make_unique<opencover::ColorBar>(&menu)) {
//   init();
// }

// bool opencover::ColorMapSelector::setValue(const std::string &colorMapName) {
//   auto it = m_colors.find(colorMapName);
//   if (it == m_colors.end()) return false;

//   m_selector->select(std::distance(m_colors.begin(), it));
//   updateSelectedMap();
//   return true;
// }

// osg::Vec4 opencover::ColorMapSelector::getColor(float val) {
//   return opencover::getColor(val, m_selectedMap->second);
// }

// const opencover::ColorMap &opencover::ColorMapSelector::selectedMap() const {
//   return m_selectedMap->second;
// }

// void opencover::ColorMapSelector::setCallback(
//     const std::function<void(const ColorMap &)> &f) {
//   m_selector->setCallback([this, f](int index) {
//     updateSelectedMap();
//     f(selectedMap());
//   });
//   if(!m_colorBar)
//     return;
//   m_colorBar->setCallback([this, f](const ColorMap &map) {
//     // m_selectedMap->second = map;
//     f(map);
//   });
// }


// void opencover::ColorMapSelector::showHud(bool show)
// {
//  m_colorBar->show(show);
// }
// bool opencover::ColorMapSelector::hudVisible() const
// {
//   return m_colorBar->hudVisible();
// }
// void opencover::ColorMapSelector::setHudPosition(const opencover::ColorBar::HudPosition &pos)
// {
//   m_colorBar->setHudPosition(pos);
// }

// void opencover::ColorMapSelector::setUnit(const std::string &unit)
// {
//   m_unit = unit;
//   m_selectedMap->second.unit = m_unit;
//   m_colorBar->update(m_selectedMap->second);
// }

// void opencover::ColorMapSelector::setSpecies(const std::string &species)
// {
//   m_species = species;
//   m_selectedMap->second.species = m_species;
//   m_colorBar->update(m_selectedMap->second);

// }

// void opencover::ColorMapSelector::setMin(float min)
// {
//   m_selectedMap->second.min = min;
//   m_colorBar->update(m_selectedMap->second);

// }
// void opencover::ColorMapSelector::setMax(float max)
// {
//   m_selectedMap->second.max = max;
//   m_colorBar->update(m_selectedMap->second);

// }
// void opencover::ColorMapSelector::setMinBounds(float min, float max)
// {
//   m_colorBar->setMinBounds(min, max);

// }
// void opencover::ColorMapSelector::setMaxBounds(float min, float max)
// {
//   m_colorBar->setMaxBounds(min, max);
// }

// void opencover::ColorMapSelector::setName(const std::string &name)
// {
//   m_colorBar->setName(name);
// }

// void opencover::ColorMapSelector::updateSelectedMap() {
//   m_selectedMap = m_colors.begin();
//   std::advance(m_selectedMap, m_selector->selectedIndex());
//   assert(m_selectedMap != m_colors.end());
//   m_selectedMap->second.unit = m_unit;
//   m_selectedMap->second.species = m_species;
//   m_colorBar->update(m_selectedMap->second);
// }

// void opencover::ColorMapSelector::init() {
//   for (auto &n : m_colors) m_selector->append(n.first);
//   m_selector->select(0);
//   m_selectedMap = m_colors.begin();

//   m_selector->setCallback([this](int index) { updateSelectedMap(); });
//   if(!m_colorBar)
//     return;
//     m_colorBar->setMaxNumSteps(m_selectedMap->second.numColors() *10);
//     m_colorBar->update(m_selectedMap->second);
// }

//***************************************************************************** */

