#pragma once
#include <osg/Vec3>

namespace prototype::core::interface {
class IMoveable {
 public:
  virtual ~IMoveable() = default;
  virtual void move(const osg::Vec3 &pos) = 0;
};
}  // namespace core::interface
