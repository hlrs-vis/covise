#pragma once
#include <osg/Vec4>

namespace core::interface {
class IColorable {
 public:
  virtual ~IColorable() = default;
  IColorable(const IColorable&) = delete;
  IColorable& operator=(const IColorable&) = delete;
  virtual void updateColor(const osg::Vec4 &color) = 0;
};
}  // namespace core::interface
