#pragma once
#include <osg/Vec4>

namespace core::interface {
class IColorable {
 public:
  virtual ~IColorable() = default;
  virtual void updateColor(const osg::Vec4 &color) = 0;
};
}  // namespace core::interface
