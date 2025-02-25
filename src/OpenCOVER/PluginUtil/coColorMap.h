
#ifndef COVUSE_UTIL_COMORMAP_H
#define COVUSE_UTIL_COMORMAP_H

#include <OpenVRUI/coUpdateManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>

#include <map>
#include <memory>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Vec4>
#include <string>
#include <vector>

#include "osg/Math"
#include "osg/Vec3d"
#include "util/coExport.h"

namespace covise {
namespace shader {
constexpr const char *COLORMAP_VERTEX_EMISSION_SHADER =
    "void main() { "
    "   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; "
    "   gl_TexCoord[0] = gl_MultiTexCoord0; "
    "}";
constexpr const char *COLORMAP_FRAGMENT_EMISSION_SHADER =
    "uniform sampler2D emissionMap;"
    "void main() { "
    "   gl_FragColor = texture2D(emissionMap, gl_TexCoord[0]);"
    "}";
}  // namespace shader

struct ColorMap {
  std::vector<float> r, g, b, a, samplingPoints;
  float min = 0.0, max = 0.0;
  int steps = 32;
};

struct ColorMapLabelConfig {
  std::string font = "DejaVuSans-Bold.ttf";
  osg::Vec4 color = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
  float charSize = 0.01f;
};

class ColorMapRenderConfig {
  auto computeColorMapRotation() const {
    return osg::Quat(osg::DegreesToRadians(rotationAngleX), osg::Vec3(1, 0, 0)) *
           osg::Quat(osg::DegreesToRadians(rotationAngleY), osg::Vec3(0, 1, 0)) *
           osg::Quat(osg::DegreesToRadians(rotationAngleZ), osg::Vec3(0, 0, 1));
  }

 public:
  ColorMapRenderConfig()
      : multisample(true),
        rotationAngleX(90.0f),
        rotationAngleY(0.0f),
        rotationAngleZ(25.0f),
        objectPositionInBase(-0.6f, 1.6f, -0.4f),
        colorMapRotation(computeColorMapRotation()) {}
  void setRotationAngleX(float rotationX) {
    this->rotationAngleX = rotationX;
    recomputeColorMapRotation();
  }
  void setRotationAngleY(float rotationY) {
    this->rotationAngleY = rotationY;
    recomputeColorMapRotation();
  }
  void setRotationAngleZ(float rotationZ) {
    this->rotationAngleZ = rotationZ;
    recomputeColorMapRotation();
  }

  auto &Mutlisample() { return multisample; }
  auto &LabelConfig() { return labelConfig; }
  auto &DistanceX() { return objectPositionInBase.x(); }
  auto &DistanceY() { return objectPositionInBase.y(); }
  auto &DistanceZ() { return objectPositionInBase.z(); }
  const auto &RotationAngleX() const { return rotationAngleX; }
  const auto &RotationAngleY() const { return rotationAngleY; }
  const auto &RotationAngleZ() const { return rotationAngleZ; }
  const auto &ColorMapRotation() const { return colorMapRotation; }
  const auto &ObjectPositionInBase() const { return objectPositionInBase; }

 private:
  void recomputeColorMapRotation() { colorMapRotation = computeColorMapRotation(); }

  bool multisample;
  float rotationAngleX, rotationAngleY, rotationAngleZ;
  ColorMapLabelConfig labelConfig;

  // object position in plugins base coordinates => -z is forward => normally; in
  // COVER y is forward
  osg::Vec3d objectPositionInBase;
  osg::Vec3d objectUp;
  osg::Quat colorMapRotation;
};

typedef std::map<std::string, ColorMap> ColorMaps;
PLUGIN_UTILEXPORT ColorMaps readColorMaps();
osg::Vec4 PLUGIN_UTILEXPORT getColor(float val, const ColorMap &colorMap,
                                     float min = 0, float max = 1);
// same logic as colors module, but sets linear sampling points
ColorMap PLUGIN_UTILEXPORT interpolateColorMap(const ColorMap &cm, int numSteps);
ColorMap PLUGIN_UTILEXPORT upscale(const ColorMap &baseMap, size_t numSteps);

class PLUGIN_UTILEXPORT ColorMapSelector {
 public:
  ColorMapSelector(opencover::ui::Menu &menu);
  ColorMapSelector(opencover::ui::Group &menu);

  bool setValue(const std::string &colorMapName);
  osg::Vec4 getColor(float val, float min = 0, float max = 1);
  const ColorMap &selectedMap() const;
  void setCallback(const std::function<void(const ColorMap &)> &f);

 private:
  opencover::ui::SelectionList *m_selector;
  const ColorMaps m_colors;
  ColorMaps::const_iterator m_selectedMap;
  void updateSelectedMap();
  void init();
};

class PLUGIN_UTILEXPORT ColorMapRenderObject : public vrui::coUpdateable {
 public:
  ColorMapRenderObject(std::shared_ptr<ColorMap> colorMap)
      : m_colormap(colorMap), m_config() {
    opencover::cover->getUpdateManager()->add(this);
  }
  ~ColorMapRenderObject() override {
    opencover::cover->getUpdateManager()->remove(this);
  }
  void show(bool on = false);

  bool update() override {
    render();
    return true;
  }

  ColorMapRenderConfig &getConfig() { return m_config; }
  const ColorMapRenderConfig &getConfig() const { return m_config; }

 private:
  void render();
  osg::ref_ptr<osg::Geode> createColorMapPlane(const ColorMap &colorMap);
  osg::ref_ptr<osg::Texture2D> createVerticalColorMapTexture(
      const ColorMap &colorMap);
  osg::ref_ptr<osg::Geode> createTextGeode(const std::string &text,
                                           const osg::Vec3 &position);
  void applyEmissionShader(osg::ref_ptr<osg::StateSet> objectStateSet,
                           osg::ref_ptr<osg::Texture2D> colormapTexture);
  void initShader();

  std::weak_ptr<ColorMap> m_colormap;
  osg::ref_ptr<osg::MatrixTransform> m_colormapTransform;
  osg::ref_ptr<osg::Program> m_shader;
  ColorMapRenderConfig m_config;
};

class PLUGIN_UTILEXPORT ColorMapUI {
 public:
  ColorMapUI(opencover::ui::Group &group);

  void setMinBounds(float min, float max) {
    setSliderBounds(m_minAttribute, min, max);
  }

  void setMaxBounds(float min, float max) {
    setSliderBounds(m_maxAttribute, min, max);
  }

  void setNumStepsBounds(int min, int max) { setSliderBounds(m_numSteps, min, max); }
  void setCallback(const std::function<void(const ColorMap &)> &f);

  auto getColor(float val);
  auto getColorMap() { return m_colorMap; }
  const auto getMin() const { return m_minAttribute->value(); }
  const auto getMax() const { return m_maxAttribute->value(); }
  const auto getNumSteps() const { return m_numSteps->value(); }
  const ColorMapSelector &getSelector() const { return *m_selector; }

 private:
  void init();
  void initUI();
  void initColorMapSettings();
  void initShow();
  void initSteps();
  void initColorMap();
  void initRenderObject();
  void rebuildColorMap();
  opencover::ui::Slider *createSlider(
      const std::string &name, const opencover::ui::Slider::ValueType &min,
      const opencover::ui::Slider::ValueType &max,
      const opencover::ui::Slider::Presentation &presentation,
      const opencover::ui::Slider::ValueType &initial,
      std::function<void(float, bool)> callback,
      opencover::ui::Group *group = nullptr);
  void setSliderBounds(opencover::ui::Slider *slider, float min, float max) {
    slider->setBounds(min, max);
  }
  void sliderCallback(opencover::ui::Slider *slider, float &toSet, float value,
                      bool moving, bool predicateCheck = false);

  void show(bool show);

  // dont delete these pointers, they are managed by the ui
  opencover::ui::Menu *m_colorMapSettingsMenu = nullptr;
  opencover::ui::Group *m_colorMapGroup = nullptr;
  opencover::ui::Slider *m_minAttribute = nullptr;
  opencover::ui::Slider *m_maxAttribute = nullptr;
  opencover::ui::Slider *m_numSteps = nullptr;
  opencover::ui::Button *m_show = nullptr;

  opencover::ui::Slider *m_distance_x = nullptr;
  opencover::ui::Slider *m_distance_y = nullptr;
  opencover::ui::Slider *m_distance_z = nullptr;

  opencover::ui::Slider *m_rotation_x = nullptr;
  opencover::ui::Slider *m_rotation_y = nullptr;
  opencover::ui::Slider *m_rotation_z = nullptr;

  opencover::ui::Slider *m_charSize = nullptr;

  std::unique_ptr<ColorMapSelector> m_selector;
  std::unique_ptr<ColorMapRenderObject> m_renderObject;
  //   std::unique_ptr<opencover::config::File> m_config = nullptr;
  std::shared_ptr<ColorMap> m_colorMap;
};

}  // namespace covise

#endif
