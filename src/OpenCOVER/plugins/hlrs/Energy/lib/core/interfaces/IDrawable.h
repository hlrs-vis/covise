#pragma once
#include <osg/Node>
#include <osg/ref_ptr>

namespace core::interface {
class IDrawable {
 public:
  virtual void initDrawable() = 0;
  virtual ~IDrawable() = default;
  IDrawable(const IDrawable&) = delete;
  IDrawable& operator=(const IDrawable&) = delete;
};
}  // namespace core::interface
