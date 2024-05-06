#ifndef _CORE_INTERFACES_ISEPERABLE_H
#define _CORE_INTERFACES_ISEPERABLE_H

#include <osg/Vec3>

namespace core {
namespace interface {
class ISeperable {
    virtual void seperate() = 0;
};
} // namespace interface
} // namespace core

#endif