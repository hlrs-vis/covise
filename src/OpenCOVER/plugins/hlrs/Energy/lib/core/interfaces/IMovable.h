#pragma once
#include <osg/Vec3>

namespace core::interface {
class IMoveable {
 public:
  virtual ~IMoveable() = default; // only for polymorphe destruction and not for ressource management
  IMoveable(const IMoveable&) = delete;
  IMoveable& operator=(const IMoveable&) = delete;
  virtual void move(const osg::Vec3 &pos) = 0;
};
}  // namespace core::interface
