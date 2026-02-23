#pragma once

#include <PluginUtil/coShaderUtil.h>
#include <lib/core/utils/color.h>

#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/ref_ptr>
#include <variant>


namespace grid {
typedef std::map<std::string, std::variant<float, int, std::string>> Data;
/**
 * @class Point
 * @brief 3D point with visual representation and associated data.
 *
 * Inherits osg::MatrixTransform for spatial transforms.
 * Encapsulates sphere geometry, data, and shader support.
 *
 * Main methods:
 * - move(offset): Move point.
 * - getRadius(), getCenter(), getPosition(): Access geometry.
 * - getAdditionalData(): Access extra data.
 * - getGeode(): Get visual node.
 * - updateColor(), updateColorMapInShader(), updateTimestepInShader(), updateDataInShader(): Visualization updates.
 */
class Point : public osg::MatrixTransform {
 public:
  Point(const std::string &name, const float &x, const float &y, const float &z,
        const float &radius, const Data &additionalData = Data());
  Point(const Point &p);

  void move(const osg::Vec3 &offset) { setMatrix(osg::Matrix::translate(offset)); }

  const auto &getRadius() const { return m_radius; }
  const auto &getCenter() const { return m_point->getCenter(); }
  const auto &getPosition() const { return m_point->getCenter(); }
  const auto &getAdditionalData() const { return m_additionalData; }
  osg::ref_ptr<osg::Geode> getGeode() {
    return dynamic_cast<osg::Geode *>(osg::Group::getChild(0));
  }

  void updateColor(const osg::Vec4 &color) {
    core::utils::color::overrideGeodeColor(getGeode(), color);
  }

  void updateColorMapInShader(const opencover::ColorMap &colormap,
                              const std::string &shaderName = "EnergyGrid");
  void updateTimestepInShader(int timestep);
  void updateDataInShader(const std::vector<double> &data, float min, float max);

 private:
  void init(const std::string &name);

  float m_radius;
  osg::ref_ptr<osg::ShapeDrawable> m_shape;
  osg::ref_ptr<osg::Sphere> m_point;
  Data m_additionalData;
  opencover::coVRShader *m_shader;
};

struct ConnectionData {
  std::string name;
  osg::ref_ptr<Point> start;
  osg::ref_ptr<Point> end;
  float radius;
  bool colorInterpolation;
  osg::ref_ptr<osg::TessellationHints> hints;
  Data additionalData;
};

enum class ConnectionType {
  Line,
  LineWithColorInterpolation,
  LineWithShader,
  Arc,
  Arrow
};

/**
 * @class DirectedConnection
 * @brief Directed connection between two grid points, visualized with OSG.
 *
 * Inherits osg::MatrixTransform for spatial transforms.
 * Supports color interpolation, shader updates, and geometric queries.
 *
 * Main methods:
 * - move(offset): Move connection and endpoints.
 * - getDirection(), getCenter(): Geometry access.
 * - getStart(), getEnd(), getGeode(): Node access.
 * - updateColor(), updateColorMapInShader(), updateTimestepInShader(), setDataInShader(): Visualization updates.
 *
 * Members:
 * - m_geode: Geometry node.
 * - m_start, m_end: Endpoints.
 * - m_additionalData: Extra data.
 * - m_type: Connection type.
 * - m_shader: Shader pointer.
 * - m_colorInterpolation: Color interpolation flag.
 * - m_numNodes: Number of nodes.
 */
class DirectedConnection : public osg::MatrixTransform {
  DirectedConnection(const std::string &name, osg::ref_ptr<Point> start,
                     osg::ref_ptr<Point> end, const float &radius,
                     bool colorInterpolation,
                     osg::ref_ptr<osg::TessellationHints> hints,
                     const Data &additionalData = Data(),
                     ConnectionType type = ConnectionType::Line);

 public:
  DirectedConnection(const ConnectionData &data,
                     ConnectionType type = ConnectionType::Line)
      : DirectedConnection(data.name, data.start, data.end, data.radius,
                           data.colorInterpolation, data.hints, data.additionalData,
                           type) {};

  void move(const osg::Vec3 &offset) {
    setMatrix(osg::Matrix::translate(offset));
    m_start->move(offset);
    m_end->move(offset);
  }
  osg::Vec3 getDirection() const {
    return m_end->getPosition() - m_start->getPosition();
  }
  osg::Vec3 getCenter() const {
    return (m_start->getPosition() + m_end->getPosition()) / 2;
  }
  osg::ref_ptr<Point> getStart() const { return m_start; }
  osg::ref_ptr<Point> getEnd() const { return m_end; }
  osg::ref_ptr<osg::Geode> getGeode() const { return m_geode; }
  const auto &getAdditionalData() const { return m_additionalData; }
  void updateColor(const osg::Vec4 &color) {
    core::utils::color::overrideGeodeColor(m_geode, color);
  }
  void setDataInShader(const std::vector<double> &fromData,
                       const std::vector<double> &toData);
  void setData1DInShader(const std::vector<double> &data, float min, float max);
  // shader needs to have same uniform buffer like
  // share/covise/materials/EnergyGrid_Line.xml
  void updateColorMapInShader(const opencover::ColorMap &colorMap,
                              const std::string &shaderName = "EnergyGrid");
  void updateTimestepInShader(int timestep);

 private:
  osg::ref_ptr<osg::Geode> m_geode;
  osg::ref_ptr<Point> m_start;
  osg::ref_ptr<Point> m_end;

  Data m_additionalData;
  ConnectionType m_type;
  // to idea who owns the shader
  opencover::coVRShader *m_shader = nullptr;
  bool m_colorInterpolation;
  int m_numNodes = 2;
};

// list of directed connections between points
typedef std::vector<osg::ref_ptr<DirectedConnection>> Connections;

/**
 * @class Line
 * @brief Represents a line made of DirectedConnections in an OSG scene.
 *
 * Inherits osg::MatrixTransform. Manages connections, movement, bounding box, and overlap checks.
 */
class Line : public osg::MatrixTransform {
 public:
  Line(std::string name, const Connections &connections);

  void move(const osg::Vec3 &offset) {
    for (const auto &[_, connection] : m_connections) connection->move(offset);
  }

  const auto &getConnections() const { return m_connections; }
  auto &getConnections() { return m_connections; }
  const auto &getAdditionalData() const { return m_additionalData; }
  const auto &getName() const { return m_name; }
  osg::Vec3 getCenter() const;
  const osg::BoundingBox &getBoundingBox() const { return m_boundingBox; }
  void recomputeBoundingBox() { computeBoundingBox(); }
  bool overlap(const Line &other) const;
  bool operator==(const Line &other) const;

 private:
  void init(const Connections &connections);
  void computeBoundingBox();
  std::string m_name;
  osg::BoundingBox m_boundingBox;
  std::map<std::string, osg::ref_ptr<DirectedConnection>> m_connections;
  Data m_additionalData;
};

// TODO: write a concept for PointType
// template <typename PointType>
// using ConnectivityList = std::vector<ConnectionData<PointType>>;
typedef std::vector<osg::ref_ptr<Point>> Points;
typedef std::map<int, osg::ref_ptr<Point>> PointsMap;
typedef std::vector<osg::ref_ptr<Line>> Lines;
typedef std::vector<std::vector<int>> Indices;
typedef std::map<int, Data> PointDataList;
typedef std::vector<std::vector<Data>> ConnectionDataList;
}  // namespace grid
