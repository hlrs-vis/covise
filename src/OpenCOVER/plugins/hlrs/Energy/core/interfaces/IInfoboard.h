#ifndef _CORE_INTERFACES_IINFOBOARD_H
#define _CORE_INTERFACES_IINFOBOARD_H

#include "ITimedependable.h"
#include "IInformable.h"
#include "IMovable.h"
#include <osg/Vec3>

namespace core {
namespace interface {
template<typename Info>
class IInfoboard: public IInformable<Info>, public ITimedependable, public IMoveable {
public:
    virtual void updateTime(int timestep) = 0;
    virtual void showInfo() = 0;
    virtual void hideInfo() = 0;
    virtual void initDrawable() = 0;
    virtual void initInfoboard() = 0;
    virtual void updateDrawable() = 0;
    virtual void updateInfo(const Info &info) = 0;
    virtual void move(const osg::Vec3 &pos) = 0;
    bool enabled() { return m_enabled; }

protected:
    bool m_enabled = false;
};
} // namespace interface
} // namespace core

#endif