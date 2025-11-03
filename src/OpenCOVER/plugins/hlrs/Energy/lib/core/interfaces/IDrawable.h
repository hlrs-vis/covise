#pragma once
#include <osg/Node>
#include <osg/ref_ptr>

namespace core::interface {
class IDrawable {
 public:
  virtual void initDrawable() = 0;
  virtual ~IDrawable() = default;
};
}  // namespace core::interface
