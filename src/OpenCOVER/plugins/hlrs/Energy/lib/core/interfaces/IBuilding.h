#pragma once
#include <osg/Vec4>

#include "IColorable.h"
#include "IDrawable.h"

namespace core::interface {
template <typename DrawableType, template <typename> class Container>
class IBuilding : public IDrawable, public IColorable {
 public:
  virtual ~IBuilding() = default;
  virtual Container<DrawableType>& getDrawables() = 0;
  virtual DrawableType& getDrawable(size_t idx) = 0;
};
}  // namespace core::interface
