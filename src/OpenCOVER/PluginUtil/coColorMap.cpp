#include "coColorMap.h"

#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Slider.h>

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

using namespace std;
using namespace opencover;

covise::ColorMaps covise::readColorMaps() {
  // read the name of all colormaps in file

  covise::coCoviseConfig::ScopeEntries colorMapEntries =
      coCoviseConfig::getScopeEntries("Colormaps");
  ColorMaps colorMaps;
#ifdef NO_COLORMAP_PARAM
  colorMapEntries["COVISE"];
#else
  // colorMapEntries["Editable"];
#endif

  for (const auto &map : colorMapEntries) {
    string name = "Colormaps." + map.first;

    auto no = coCoviseConfig::getScopeEntries(name).size();
    ColorMap &colorMap = colorMaps.emplace(map.first, ColorMap()).first->second;
    // read all sampling points
    float diff = 1.0f / (no - 1);
    float pos = 0;
    for (int j = 0; j < no; j++) {
      string tmp = name + ".Point:" + std::to_string(j);
      ColorMap cm;
      colorMap.r.push_back(coCoviseConfig::getFloat("r", tmp, 0));
      colorMap.g.push_back(coCoviseConfig::getFloat("g", tmp, 0));
      colorMap.b.push_back(coCoviseConfig::getFloat("b", tmp, 0));
      colorMap.a.push_back(coCoviseConfig::getFloat("a", tmp, 1));
      colorMap.samplingPoints.push_back(coCoviseConfig::getFloat("x", tmp, pos));
      pos += diff;
    }
  }
  return colorMaps;
}

osg::Vec4 covise::getColor(float val, const covise::ColorMap &colorMap, float min,
                           float max) {
  assert(val >= min && val <= max);
  val = 1 / (max - min) * (val - min);

  size_t idx = 0;
  for (; idx < colorMap.samplingPoints.size() &&
         colorMap.samplingPoints[idx + 1] < val;
       idx++) {
  }

  double d = (val - colorMap.samplingPoints[idx]) /
             (colorMap.samplingPoints[idx + 1] - colorMap.samplingPoints[idx]);
  osg::Vec4 color;
  color[0] = ((1 - d) * colorMap.r[idx] + d * colorMap.r[idx + 1]);
  color[1] = ((1 - d) * colorMap.g[idx] + d * colorMap.g[idx + 1]);
  color[2] = ((1 - d) * colorMap.b[idx] + d * colorMap.b[idx + 1]);
  color[3] = ((1 - d) * colorMap.a[idx] + d * colorMap.a[idx + 1]);

  return color;
}

covise::ColorMap covise::interpolateColorMap(const covise::ColorMap &cm,
                                             int numSteps) {
  covise::ColorMap interpolatedMap;
  interpolatedMap.r.resize(numSteps);
  interpolatedMap.g.resize(numSteps);
  interpolatedMap.b.resize(numSteps);
  interpolatedMap.a.resize(numSteps);
  interpolatedMap.samplingPoints.resize(numSteps);
  auto numColors = cm.samplingPoints.size();
  double delta = 1.0 / (numSteps - 1) * (numColors - 1);
  double x;
  int i;

  delta = 1.0 / (numSteps - 1);
  int idx = 0;
  for (i = 0; i < numSteps - 1; i++) {
    x = i * delta;
    while (cm.samplingPoints[(idx + 1)] <= x) {
      idx++;
      if (idx > numColors - 2) {
        idx = numColors - 2;
        break;
      }
    }

    double d = (x - cm.samplingPoints[idx]) /
               (cm.samplingPoints[idx + 1] - cm.samplingPoints[idx]);
    interpolatedMap.r[i] = (float)((1 - d) * cm.r[idx] + d * cm.r[idx + 1]);
    interpolatedMap.g[i] = (float)((1 - d) * cm.g[idx] + d * cm.g[idx + 1]);
    interpolatedMap.b[i] = (float)((1 - d) * cm.b[idx] + d * cm.b[idx + 1]);
    interpolatedMap.a[i] = (float)((1 - d) * cm.a[idx] + d * cm.a[idx + 1]);
    interpolatedMap.samplingPoints[i] = (float)i / (numSteps - 1);
  }
  interpolatedMap.r[numSteps - 1] = cm.r[(numColors - 1)];
  interpolatedMap.g[numSteps - 1] = cm.g[(numColors - 1)];
  interpolatedMap.b[numSteps - 1] = cm.b[(numColors - 1)];
  interpolatedMap.a[numSteps - 1] = cm.a[(numColors - 1)];
  interpolatedMap.samplingPoints[numSteps - 1] = 1;

  interpolatedMap.min = cm.min;
  interpolatedMap.max = cm.max;
  interpolatedMap.steps = numSteps;

  return interpolatedMap;
}

covise::ColorMapSelector::ColorMapSelector(opencover::ui::Group &group)
    : m_selector(new opencover::ui::SelectionList(&group, "mapChoice")),
      m_colors(readColorMaps()) {
  init();
}

covise::ColorMapSelector::ColorMapSelector(opencover::ui::Menu &menu)
    : m_selector(new opencover::ui::SelectionList{&menu, "mapChoice"}),
      m_colors(readColorMaps()) {
  init();
}

bool covise::ColorMapSelector::setValue(const std::string &colorMapName) {
  auto it = m_colors.find(colorMapName);
  if (it == m_colors.end()) return false;

  m_selector->select(std::distance(m_colors.begin(), it));
  updateSelectedMap();
  return true;
}

osg::Vec4 covise::ColorMapSelector::getColor(float val, float min, float max) {
  return covise::getColor(val, m_selectedMap->second, min, max);
}

const covise::ColorMap &covise::ColorMapSelector::selectedMap() const {
  return m_selectedMap->second;
}

void covise::ColorMapSelector::setCallback(
    const std::function<void(const ColorMap &)> &f) {
  m_selector->setCallback([this, f](int index) {
    updateSelectedMap();
    f(selectedMap());
  });
}

void covise::ColorMapSelector::updateSelectedMap() {
  m_selectedMap = m_colors.begin();
  std::advance(m_selectedMap, m_selector->selectedIndex());
  assert(m_selectedMap != m_colors.end());
}

void covise::ColorMapSelector::init() {
  for (auto &n : m_colors) m_selector->append(n.first);
  m_selector->select(0);
  m_selectedMap = m_colors.begin();

  m_selector->setCallback([this](int index) { updateSelectedMap(); });
}

osg::ref_ptr<osg::Texture2D>
covise::ColorMapRenderObject::createVerticalColorMapTexture(
    const ColorMap &colorMap) {
  if (colorMap.r.empty() || colorMap.g.empty() || colorMap.b.empty() ||
      colorMap.a.empty() || colorMap.samplingPoints.empty()) {
    return nullptr;
  }

  int width = 1;  // 1D texture, now vertical
  int height = colorMap.steps;

  osg::ref_ptr<osg::Image> image = new osg::Image;
  image->allocateImage(width, height, 1, GL_RGBA, GL_FLOAT);

  float *imageData = (float *)image->data();

  for (int y = 0; y < height; ++y) {
    float samplePoint = (float)y / (height - 1);

    for (size_t i = 1; i < colorMap.samplingPoints.size(); ++i) {
      if (samplePoint <= colorMap.samplingPoints[i]) {
        float t = (samplePoint - colorMap.samplingPoints[i - 1]) /
                  (colorMap.samplingPoints[i] - colorMap.samplingPoints[i - 1]);

        float r = colorMap.r[i - 1] + t * (colorMap.r[i] - colorMap.r[i - 1]);
        float g = colorMap.g[i - 1] + t * (colorMap.g[i] - colorMap.g[i - 1]);
        float b = colorMap.b[i - 1] + t * (colorMap.b[i] - colorMap.b[i - 1]);
        float a = colorMap.a[i - 1] + t * (colorMap.a[i] - colorMap.a[i - 1]);

        imageData[y * 4 + 0] = r;
        imageData[y * 4 + 1] = g;
        imageData[y * 4 + 2] = b;
        imageData[y * 4 + 3] = a;

        break;
      }
    }
  }

  osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
  texture->setImage(image);
  texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
  texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
  texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
  texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);

  return texture;
}

osg::ref_ptr<osg::Geode> covise::ColorMapRenderObject::createColorMapPlane(
    const covise::ColorMap &colorMap) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;

  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  vertices->push_back(osg::Vec3(0, 0, 0));
  vertices->push_back(osg::Vec3(1, 0, 0));
  vertices->push_back(osg::Vec3(1, 1, 0));
  vertices->push_back(osg::Vec3(0, 1, 0));

  geometry->setVertexArray(vertices);

  osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array;
  texcoords->push_back(osg::Vec2(0, 0));
  texcoords->push_back(osg::Vec2(1, 0));
  texcoords->push_back(osg::Vec2(1, 1));
  texcoords->push_back(osg::Vec2(0, 1));
  geometry->setTexCoordArray(0, texcoords);

  osg::ref_ptr<osg::DrawElementsUInt> quad =
      new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
  for (auto i = 0; i < 4; ++i) quad->push_back(i);

  geometry->addPrimitiveSet(quad);

  geode->addDrawable(geometry);

  osg::ref_ptr<osg::Texture2D> texture = createVerticalColorMapTexture(colorMap);
  osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
  if (texture) {
    applyEmissionShader(stateset, texture);
  }

  if (m_config.Mutlisample()) {
    osg::ref_ptr<osg::Multisample> multisample = new osg::Multisample;
    stateset->setAttributeAndModes(multisample, osg::StateAttribute::ON);
  }

  return geode;
}

osg::ref_ptr<osg::Geode> covise::ColorMapRenderObject::createTextGeode(
    const std::string &text, const osg::Vec3 &position) {
  osg::ref_ptr<osgText::Text> osgText = new osgText::Text;
  osgText->setText(text);
  osgText->setFont(m_config.LabelConfig().font);
  osgText->setCharacterSize(m_config.LabelConfig().charSize);  // Adjust size
  osgText->setPosition(position);
  osgText->setColor(m_config.LabelConfig().color);  // White text
  osgText->setAlignment(osgText::Text::LEFT_CENTER);

  osg::ref_ptr<osg::Geode> textGeode = new osg::Geode;
  textGeode->addDrawable(osgText);
  return textGeode;
}

void covise::ColorMapRenderObject::initShader() {
  // Add a shader to apply the texture as emission.
  osg::ref_ptr<osg::Program> program = new osg::Program;
  osg::ref_ptr<osg::Shader> vertexShader = new osg::Shader(osg::Shader::VERTEX);
  vertexShader->setShaderSource(shader::COLORMAP_VERTEX_EMISSION_SHADER);
  osg::ref_ptr<osg::Shader> fragmentShader = new osg::Shader(osg::Shader::FRAGMENT);
  fragmentShader->setShaderSource(shader::COLORMAP_FRAGMENT_EMISSION_SHADER);
  program->addShader(vertexShader);
  program->addShader(fragmentShader);
  m_shader = program;
}

void covise::ColorMapRenderObject::applyEmissionShader(
    osg::ref_ptr<osg::StateSet> stateSet,
    osg::ref_ptr<osg::Texture2D> colormapTexture) {
  assert(stateSet && "Cannot apply emission shader to uninitialized stateSet");
  if (colormapTexture) {
    auto colormap = m_colormap.lock();
    assert(colormap &&
           "Given colormapTexture is not valid and colormap weak_ptr is locked.");
    colormapTexture = createVerticalColorMapTexture(*colormap);
  }
  stateSet->setTextureAttributeAndModes(0, colormapTexture, osg::StateAttribute::ON);
  stateSet->addUniform(
      new osg::Uniform("emissionMap", 0));  // Assuming texture unit 0
  stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

  // Add a shader to apply the texture as emission.
  if (!m_shader) initShader();

  stateSet->setAttributeAndModes(m_shader, osg::StateAttribute::ON);
}

void covise::ColorMapRenderObject::show(bool on) {
  if (on) {
    auto colorMap = m_colormap.lock();
    if (!colorMap) {
      std::cerr << "ColorMapRenderObject: ColorMap is not set or in use."
                << std::endl;
      return;
    }

    auto colormapPlane = createColorMapPlane(*m_colormap.lock());

    // position colormap relative to the object
    osg::ref_ptr<osg::PositionAttitudeTransform> pat =
        new osg::PositionAttitudeTransform();
    pat->setPosition(osg::Vec3(0.0f, 0.0f, 0.0f));
    pat->setScale(osg::Vec3(0.1f, 0.8f, 1.0f));
    pat->addChild(colormapPlane);

    osg::ref_ptr<osg::Group> colormapGroup = new osg::Group();
    colormapGroup->addChild(pat);

    // add text labels for the sampling points
    for (size_t i = 0; i < colorMap->samplingPoints.size(); ++i) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << colorMap->samplingPoints[i];
      osg::ref_ptr<osg::Geode> textGeode = createTextGeode(
          ss.str(), osg::Vec3(-0.1f, colorMap->samplingPoints[i] * 0.8f, 0.0f));
      colormapGroup->addChild(textGeode);
    }

    // create a transform node to move the colormap to the right position
    m_colormapTransform = new osg::MatrixTransform();
    m_colormapTransform->addChild(colormapGroup);
    m_colormapTransform->setName("ColorMap");

    cover->getObjectsRoot()->addChild(m_colormapTransform);
  } else {
    cover->getObjectsRoot()->removeChild(m_colormapTransform);
  }
}

void covise::ColorMapRenderObject::render() {
  if (m_colormapTransform) {
    // First, cover->getInvBaseMat() transforms world coordinates into the plugin's
    // base coordinate system.
    //
    // Then, cover->getViewerMat() transforms the results of
    // the previous operation, from the plugins base coordinate system, into the
    // viewers coordinate system.
    //
    // Therefore, the combined result is a matrix that transforms coordinates from
    // the plugin's base coordinate system directly into the viewer's coordinate
    // system.
    auto transformMatrix = cover->getViewerMat() * cover->getInvBaseMat();
    osg::Vec3d scale, translation;
    osg::Quat rotationNoScale, scaleOrientation;
    transformMatrix.decompose(translation, rotationNoScale, scale, scaleOrientation);
    auto transformMatrixNoScale =
        osg::Matrixd::rotate(rotationNoScale) * osg::Matrixd::translate(translation);

    // transform to viewer coordinates
    auto objectPositionInViewer =
        m_config.ObjectPositionInBase() * transformMatrixNoScale;

    // apply transformation to object
    osg::Matrixd matrix;
    // matrix.makeRotate(colorMapRotation * rotationNoScale);
    matrix.makeRotate(m_config.ColorMapRotation() * rotationNoScale);
    matrix.setTrans(objectPositionInViewer);
    m_colormapTransform->setMatrix(matrix);
  }
}

covise::ColorMapUI::ColorMapUI(opencover::ui::Group &group)
    : m_colorMapGroup(new opencover::ui::Group(&group, "ColorMap")),
      m_colorMapSettingsMenu(new opencover::ui::Menu(&group, "ColorMapSettings")),
      //   m_config(new opencover::config::File("ColorMapConfig")),
      m_selector(std::make_unique<ColorMapSelector>(*m_colorMapGroup)) {
  init();
}

void covise::ColorMapUI::sliderCallback(opencover::ui::Slider *slider, float &toSet,
                                        float value, bool moving,
                                        bool predicateCheck) {
  if (!moving) return;
  if (predicateCheck) {
    slider->setValue(toSet);
    return;
  }
  toSet = value;
}

opencover::ui::Slider *covise::ColorMapUI::createSlider(
    const std::string &name, const ui::Slider::ValueType &min,
    const ui::Slider::ValueType &max, const ui::Slider::Presentation &presentation,
    const ui::Slider::ValueType &initial, std::function<void(float, bool)> callback,
    opencover::ui::Group *group) {
  if (!group) group = m_colorMapGroup;
  auto slider = new ui::Slider(group, name);
  slider->setBounds(min, max);
  slider->setPresentation(presentation);
  slider->setValue(initial);
  slider->setCallback(callback);
  return slider;
}

void covise::ColorMapUI::initSteps() {
  m_numSteps = createSlider(
      "steps", 1, 1024, ui::Slider::AsDial, 1, [this](float value, bool moving) {
        if (value < 1) return;
        if (!moving) return;
        *m_colorMap = covise::interpolateColorMap(m_selector->selectedMap(), value);
        rebuildColorMap();
      });
  m_numSteps->setScale(ui::Slider::Linear);
  m_numSteps->setIntegral(true);
  m_numSteps->setLinValue(32);
}

void covise::ColorMapUI::initColorMap() {
  assert(m_selector && "ColorMapSelector must be initialized before ColorMap");
  m_colorMap = std::make_shared<ColorMap>(m_selector->selectedMap());
}

void covise::ColorMapUI::initShow() {
  m_show = new ui::Button(m_colorMapGroup, "Show");
  m_show->setCallback([this](bool on) { show(on); });
}

void covise::ColorMapUI::initColorMapSettings() {
  assert(m_renderObject && "RenderObject need to be initialized before calling it.");
  auto renderConfig = m_renderObject->getConfig();
  m_distance_x = createSlider(
      "colormap_distance_x", -5.0f, 5.0f, ui::Slider::AsDial,
      renderConfig.DistanceX(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().DistanceX() = value;
      },
      m_colorMapSettingsMenu);
  m_distance_y = createSlider(
      "colormap_distance_y", -5.0f, 5.0f, ui::Slider::AsDial,
      renderConfig.DistanceY(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().DistanceY() = value;
      },
      m_colorMapSettingsMenu);
  m_distance_z = createSlider(
      "colormap_distance_z", -5.0f, 5.0f, ui::Slider::AsDial,
      renderConfig.DistanceZ(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().DistanceZ() = value;
      },
      m_colorMapSettingsMenu);

  m_rotation_x = createSlider(
      "colormap_rotation_x", 0.0f, 360.0f, ui::Slider::AsDial,
      renderConfig.RotationAngleX(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().setRotationAngleX(value);
      },
      m_colorMapSettingsMenu);
  m_rotation_y = createSlider(
      "colormap_rotation_y", 0.0f, 360.0f, ui::Slider::AsDial,
      renderConfig.RotationAngleY(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().setRotationAngleY(value);
      },
      m_colorMapSettingsMenu);
  m_rotation_z = createSlider(
      "colormap_rotation_z", 0.0f, 360.0f, ui::Slider::AsDial,
      renderConfig.RotationAngleZ(),
      [this](float value, bool moving) {
        m_renderObject->getConfig().setRotationAngleZ(value);
      },
      m_colorMapSettingsMenu);
  m_charSize = createSlider(
      "charSize", 0.01, 0.04, ui::Slider::AsDial,
      renderConfig.LabelConfig().charSize,
      [this](float value, bool moving) {
        if (!moving) return;
        m_renderObject->getConfig().LabelConfig().charSize = value;
        rebuildColorMap();
      },
      m_colorMapSettingsMenu);
}

void covise::ColorMapUI::initUI() {
  initShow();
  initColorMap();
  m_minAttribute = createSlider(
      "min", 0, 1, ui::Slider::AsDial, 0, [this](float value, bool moving) {
        sliderCallback(m_minAttribute, m_colorMap->min, value, moving,
                       value > m_maxAttribute->value());
      });
  m_maxAttribute = createSlider(
      "max", 0, 1, ui::Slider::AsDial, 1, [this](float value, bool moving) {
        sliderCallback(m_maxAttribute, m_colorMap->max, value, moving,
                       value < m_minAttribute->value());
      });
  initSteps();
  initRenderObject();
  initColorMapSettings();
}

void covise::ColorMapUI::initRenderObject() {
  assert(m_colorMap && "ColorMap must be initialized before render object");
  m_renderObject = std::make_unique<ColorMapRenderObject>(m_colorMap);
}

void covise::ColorMapUI::init() { initUI(); }

void covise::ColorMapUI::rebuildColorMap() {
  assert(m_colorMap && "ColorMap must be initialized before rebuilding");
  assert(m_renderObject && "RenderObject must be initialized before rebuilding");
  m_renderObject->show(false);
  m_renderObject->show(true);
}

void covise::ColorMapUI::setCallback(
    const std::function<void(const ColorMap &)> &f) {
  m_selector->setCallback([this, f](const ColorMap &cm) {
    *m_colorMap = interpolateColorMap(cm, m_numSteps->value());
    f(*m_colorMap);
    rebuildColorMap();
  });
}

auto covise::ColorMapUI::getColor(float val) {
  return covise::getColor(val, *m_colorMap, m_colorMap->min, m_colorMap->max);
}

void covise::ColorMapUI::show(bool show) { m_renderObject->show(show); }
