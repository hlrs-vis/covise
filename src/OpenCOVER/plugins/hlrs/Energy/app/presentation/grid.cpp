#include "grid.h"

#include <PluginUtil/colors/coColorMap.h>
#include <lib/core/utils/osgUtils.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/StateAttribute>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Vec4>

namespace {
void updateMinMax(osg::Vec3 &minExtends, osg::Vec3 &maxExtends,
                  const osg::Vec3 &point) {
  minExtends.x() = std::min(minExtends.x(), point.x());
  minExtends.y() = std::min(minExtends.y(), point.y());
  minExtends.z() = std::min(minExtends.z(), point.z());

  maxExtends.x() = std::max(maxExtends.x(), point.x());
  maxExtends.y() = std::max(maxExtends.y(), point.y());
  maxExtends.z() = std::max(maxExtends.z(), point.z());
}

constexpr int SHADER_SCALAR_TIMESTEP_MAPPING_INDEX =
    0;  // index of the texture that maps from energy grid node index to timestep
        // value
}  // namespace
   //
using namespace core;

namespace grid {

// namespace core::simulation::grid {
Point::Point(const std::string &name, const float &x, const float &y, const float &z,
             const float &radius, const Data &additionalData)
    : osg::MatrixTransform(),
      m_point(new osg::Sphere(osg::Vec3(x, y, z), radius)),
      m_additionalData(additionalData),
      m_radius(radius),
      m_shader(nullptr) {
  init(name);
}

Point::Point(const Point &other)
    : m_additionalData(other.getAdditionalData()), m_radius(other.getRadius()) {
  m_point = new osg::Sphere(other.getPosition(), m_radius);
  init(other.getName());
}

void Point::init(const std::string &name) {
  osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
  hints->setDetailRatio(1.5f);
  m_shape = new osg::ShapeDrawable(m_point, hints);
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  geode->addChild(m_shape);
  addChild(geode);
  setName(name);
}

void Point::updateColorMapInShader(const opencover::ColorMap &colorMap,
                                   const std::string &shaderName) {
  osg::ref_ptr<osg::Geode> geode = getChild(0)->asGeode();
  m_shader = opencover::applyShader(geode, colorMap, shaderName);
  m_shader->setIntUniform("numNodes", 1);

  auto state = geode->getOrCreateStateSet();
  m_shader->apply(state);
  geode->setStateSet(state);
}

void Point::updateDataInShader(const std::vector<double> &data, float min,
                               float max) {
  if (!m_shader) {
    std::cerr << "Point::updateDataInShader: No shader set for point " << getName()
              << "\n";
    return;
  }

  m_shader->setIntUniform("numTimesteps", data.size());
  m_shader->setIntUniform("numNodes", 1);

  auto uniform = m_shader->getcoVRUniform("timestepToData");
  assert(uniform);
  uniform->setValue(std::to_string(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX).c_str());

  auto texture = core::utils::osgUtils::createValue1DTexture(data);
  auto geode = getChild(0);
  auto state = geode->getOrCreateStateSet();
  state->setTextureAttribute(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX, texture,
                             osg::StateAttribute::ON);

  m_shader->apply(state);
  geode->setStateSet(state);
}

void Point::updateTimestepInShader(int timestep) {
  if (!m_shader) {
    std::cerr << "Point::updateTimestep: No shader set for connection "
              << getName() << "\n";
    return;
  }

  m_shader->setIntUniform("timestep", timestep);
  auto geode = getChild(0);
  auto state = geode->getOrCreateStateSet();
  m_shader->apply(state);
  geode->setStateSet(state);
}

constexpr int NUM_CIRCLE_POINTS = 20;

DirectedConnection::DirectedConnection(const std::string &name,
                                       osg::ref_ptr<Point> start,
                                       osg::ref_ptr<Point> end, const float &radius,
                                       bool colorInterpolation,
                                       osg::ref_ptr<osg::TessellationHints> hints,
                                       const Data &additionalData,
                                       ConnectionType type)
    : osg::MatrixTransform(),
      m_start(start),
      m_end(end),
      m_additionalData(additionalData),
      m_colorInterpolation(colorInterpolation) {
  switch (type) {
    case ConnectionType::Line:
      m_geode = utils::osgUtils::createOsgCylinderBetweenPoints(
          start->getPosition(), end->getPosition(), radius,
          osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f), hints);
      break;
    case ConnectionType::LineWithColorInterpolation:
      m_geode = utils::osgUtils::createCylinderBetweenPointsColorInterpolation(
          start->getPosition(), end->getPosition(), radius * 2.0f, radius,
          NUM_CIRCLE_POINTS, 1, osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f),
          osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f), hints);
      break;
    case ConnectionType::LineWithShader: {
      auto geometry = utils::osgUtils::createCylinderBetweenPoints(
          start->getPosition(), end->getPosition(), radius, NUM_CIRCLE_POINTS, 1,
          hints, m_colorInterpolation);
      m_geode = new osg::Geode();
      m_geode->addDrawable(geometry);

    } break;
    case ConnectionType::Arc:
      m_geode = utils::osgUtils::createBezierTube(
          start->getPosition(), end->getPosition(), radius * 2.0f, radius, 50,
          osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));
      break;
    case ConnectionType::Arrow:
      assert(false && "Arrow type not implemented");
  }
  addChild(m_geode);
  setName(name);
}

void DirectedConnection::setDataInShader(const std::vector<double> &fromData,
                                         const std::vector<double> &toData) {
  if (!m_shader) {
    std::cerr << "DirectedConnection::setData: No shader set for connection "
              << getName() << "\n";
    return;
  }
  std::cerr << "Setting data shader for connection: " << getName() << "\n";
  m_shader->setIntUniform("numTimesteps", fromData.size());
  // might be unnecessary, default should be 0 anyway
  auto uniform = m_shader->getcoVRUniform("timestepToData");
  assert(uniform);
  uniform->setValue(std::to_string(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX).c_str());

  auto texture = core::utils::osgUtils::createValueTexture(fromData, toData);
  auto drawable = m_geode->getDrawable(0);
  auto state = drawable->getOrCreateStateSet();
  state->setTextureAttribute(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX, texture,
                             osg::StateAttribute::ON);

  m_shader->apply(state);
  drawable->setStateSet(state);
}

void DirectedConnection::setData1DInShader(const std::vector<double> &data,
                                           float min, float max) {
  if (!m_shader) {
    std::cerr << "DirectedConnection::setData: No shader set for connection "
              << getName() << "\n";
    return;
  }
  std::cerr << "Setting 1D data shader for connection: " << getName() << "\n";
  m_shader->setIntUniform("numTimesteps", data.size());
  m_shader->setIntUniform("numNodes", 1);

  auto uniform = m_shader->getcoVRUniform("timestepToData");
  assert(uniform);
  uniform->setValue(std::to_string(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX).c_str());

  auto texture = core::utils::osgUtils::createValue1DTexture(data);
  auto drawable = m_geode->getDrawable(0);
  auto state = drawable->getOrCreateStateSet();
  state->setTextureAttribute(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX, texture,
                             osg::StateAttribute::ON);

  m_shader->apply(state);
  drawable->setStateSet(state);
}

void DirectedConnection::updateColorMapInShader(const opencover::ColorMap &colorMap,
                                                const std::string &shaderName) {
  auto drawable = m_geode->getDrawable(0);
  m_shader = opencover::applyShader(drawable, colorMap, shaderName);
  m_shader->setIntUniform("numNodes", m_numNodes);

  auto state = drawable->getOrCreateStateSet();
  m_shader->apply(state);
  drawable->setStateSet(state);
}

void DirectedConnection::updateTimestepInShader(int timestep) {
  if (!m_shader) {
    std::cerr << "DirectedConnection::updateTimestep: No shader set for connection "
              << getName() << "\n";
    return;
  }

  m_shader->setIntUniform("timestep", timestep);
  auto drawable = m_geode->getDrawable(0);
  auto state = drawable->getOrCreateStateSet();
  m_shader->apply(state);
  drawable->setStateSet(state);
}

Line::Line(std::string name, const Connections &connections) : m_name(name) {
  init(connections);
}

void Line::init(const Connections &connections) {
  if (connections.empty()) {
    std::cerr << "Line: No connections provided\n";
    return;
  }
  setName(m_name);
  // TODO: for now only the first connection is used to get the additional data
  m_additionalData = connections[0]->getAdditionalData();
  for (const auto &connection : connections) {
    const auto &name = connection->getName();
    m_connections[name] = connection;
    addChild(connection);
  }
  computeBoundingBox();
}

bool Line::operator==(const Line &other) const {
  return m_name == other.m_name ||
         std::any_of(m_connections.begin(), m_connections.end(),
                     [&other](const auto &pair) {
                       return other.m_connections.find(pair.first) !=
                              other.m_connections.end();
                     });
}

bool Line::overlap(const Line &other) const {
  for (const auto &[name, connection] : m_connections) {
    auto otherConnections = other.getConnections();
    auto start = connection->getStart()->getPosition();
    auto end = connection->getEnd()->getPosition();
    if (std::find_if(otherConnections.begin(), otherConnections.end(),
                     [&name, &start, &end](const auto &otherPair) {
                       const auto &[otherName, otherCon] = otherPair;

                       if (name == otherName) return true;

                       auto otherStart = otherCon->getStart();
                       auto otherEnd = otherCon->getEnd();

                       // check if the name is simply in reverse order
                       std::string otherReverseName =
                           otherEnd->getName() + " > " + otherStart->getName();

                       if (name == otherReverseName) return true;

                       const auto &otherStartPos = otherStart->getPosition();
                       const auto &otherEndPos = otherEnd->getPosition();

                       // check if the start and end points overlap
                       return (start == otherStartPos && end == otherEndPos) ||
                              (end == otherStartPos || start == otherEndPos);
                     }) != otherConnections.end()) {
      return true;
    }
  }
  return false;
}

osg::Vec3 Line::getCenter() const {
  osg::Vec3 center(0.0f, 0.0f, 0.0f);
  for (const auto &[_, connection] : m_connections)
    center += connection->getCenter();
  return center / m_connections.size();
}

void Line::computeBoundingBox() {
  assert(!m_connections.empty() && "No connections to compute bounding box");
  auto firstConnection = m_connections.begin()->second;
  const auto &start = firstConnection->getStart()->getPosition();
  osg::Vec3 minExtends(start);
  osg::Vec3 maxExtends(start);
  for (const auto &[_, connection] : m_connections) {
    auto start = connection->getStart();
    auto end = connection->getEnd();
    const auto &startPosition = start->getPosition();
    const auto &endPosition = end->getPosition();
    const auto &startRadius = start->getRadius();
    const auto &endRadius = end->getRadius();
    auto startRadiusVec = osg::Vec3(startRadius, startRadius, startRadius);
    auto endRadiusVec = osg::Vec3(endRadius, endRadius, endRadius);
    auto direction = connection->getDirection();
    direction.normalize();

    updateMinMax(minExtends, maxExtends,
                 startPosition - startRadiusVec * direction.length());
    updateMinMax(minExtends, maxExtends,
                 endPosition + endRadiusVec * direction.length());
  }
  m_boundingBox.set(minExtends, maxExtends);
}
}  // namespace grid
