#ifndef _CORE_GRID_H
#define _CORE_GRID_H

#include <osg/Geode>
#include <osg/Group>
#include <osg/ShapeDrawable>
#include <variant>

#include "../utils/color.h"

namespace core::simulation::grid {
// grid::Data is a vector of variants that can hold int, float, or string
typedef std::map<std::string, std::variant<float, int, std::string>> Data;
class Point : public osg::Group {
 public:
  Point(const std::string &name, const float &x, const float &y, const float &z,
        const float &radius, const Data &additionalData = Data());

  const auto &getPosition() const { return m_point->getCenter(); }
  const Data &getAdditionalData() const { return m_additionalData; }
  osg::ref_ptr<osg::Geode> getGeode() {
    return dynamic_cast<osg::Geode *>(osg::Group::getChild(0));
  }

  void updateColor(const osg::Vec4 &color) {
    core::utils::color::overrideGeodeColor(getGeode(), color);
  }

 private:
  osg::ref_ptr<osg::ShapeDrawable> m_shape;
  osg::ref_ptr<osg::Sphere> m_point;
  Data m_additionalData;
};

template <typename CoordType>
struct ConnectionData {
  ConnectionData(const std::string &name, const CoordType &start,
                 const CoordType &end, const float &radius,
                 osg::ref_ptr<osg::TessellationHints> hints = nullptr,
                 const Data &additionalData = Data())
      : name(name),
        start(start),
        end(end),
        radius(radius),
        hints(hints),
        additionalData(additionalData) {};
  std::string name;
  CoordType start;
  CoordType end;
  float radius;
  osg::ref_ptr<osg::TessellationHints> hints;
  Data additionalData;
};

class DirectedConnection : public osg::Group {
  DirectedConnection(const std::string &name, const osg::Vec3 &start,
                     const osg::Vec3 &end, const float &radius,
                     osg::ref_ptr<osg::TessellationHints> hints,
                     const Data &additionalData = Data());

 public:
  DirectedConnection(const ConnectionData<osg::Vec3> &data)
      : DirectedConnection(data.name, data.start, data.end, data.radius, data.hints,
                           data.additionalData) {};

  DirectedConnection(const ConnectionData<Point> &data)
      : DirectedConnection(data.name, data.start.getPosition(),
                           data.end.getPosition(), data.radius, data.hints,
                           data.additionalData) {};

  osg::Vec3 getDirection() const { return *m_end - *m_start; }
  osg::Vec3 getCenter() const { return (*m_start + *m_end) / 2; }
  osg::ref_ptr<osg::Geode> getGeode() const { return m_geode; }
  const Data &getAdditionalData() const { return m_additionalData; }
  void updateColor(const osg::Vec4 &color) {
    core::utils::color::overrideGeodeColor(m_geode, color);
  }

 private:
  osg::ref_ptr<osg::Geode> m_geode;
  osg::Vec3 *m_start;
  osg::Vec3 *m_end;
  Data m_additionalData;
};

typedef std::vector<osg::ref_ptr<Point>> Points;
typedef std::vector<osg::ref_ptr<DirectedConnection>> Connections;
// list of directed connections between points
// TODO: write a concept for PointType
// template <typename PointType>
// using ConnectivityList = std::vector<ConnectionData<PointType>>;
typedef std::vector<std::vector<int>> Indices;
typedef std::vector<Data> DataList;
}  // namespace core::simulation::grid

#endif
