#ifndef _CORE_INTERFACES_IMOVEABLE_H
#define _CORE_INTERFACES_IMOVEABLE_H

#include <osg/Vec3>

namespace core {
namespace interface {
class IMoveable {
public:
    virtual void move(const osg::Vec3 &pos) = 0;
};
} // namespace interface
} // namespace core

#endif