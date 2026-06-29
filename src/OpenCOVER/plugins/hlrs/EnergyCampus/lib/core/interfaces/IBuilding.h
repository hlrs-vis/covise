#pragma once
#include <memory>
#include <osg/Vec4>

#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"

namespace core::interface {
class IBuilding : public IDrawables, public IColorable, public ITimedependable {
 public:
  virtual ~IBuilding() = default;
  /**
   * Returns the color in the range [0, maxValue] based on the given value based
   * on colorRange of building instance.
   *
   * @param value The value to determine the color for.
   * @param maxValue The maximum value in the range.
   * @return The color represented as an osg::Vec4.
   */
  virtual std::unique_ptr<osg::Vec4> getColorInRange(float value,
                                                     float maxValue) = 0;
};
}  // namespace core::interface
