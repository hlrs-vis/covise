#ifndef _CORE_INTERFACES_IBUILDING_H
#define _CORE_INTERFACES_IBUILDING_H

#include "IDrawable.h"
#include "IColorable.h"
#include "IMovable.h"
#include "ITimedependable.h"
#include <memory>
#include <osg/ref_ptr>
#include <osg/Vec4>

namespace core {
namespace interface {
class IBuilding: public IDrawable, public IColorable, public IMoveable, public ITimedependable {
public:
    virtual void move(const osg::Vec3 &pos) = 0;
    virtual void initDrawable() = 0;
    virtual void updateDrawable() = 0;
    virtual void updateColor(const osg::Vec4 &color) = 0;
    virtual void updateTime(int timestep) = 0;
    /**
     * Returns the color in the range [0, maxValue] based on the given value based on colorRange of building instance.
     *
     * @param value The value to determine the color for.
     * @param maxValue The maximum value in the range.
     * @return The color represented as an osg::Vec4.
     */
    virtual std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) = 0;
};
} // namespace interface
} // namespace core

#endif