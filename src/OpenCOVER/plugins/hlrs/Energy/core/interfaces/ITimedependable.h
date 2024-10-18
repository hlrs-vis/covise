#ifndef _CORE_INTERFACES_ITIMEDEPENDABLE_H
#define _CORE_INTERFACES_ITIMEDEPENDABLE_H

namespace core {
namespace interface {
class ITimedependable {
public:
    virtual void updateTime(int timestep) = 0;
    virtual int getCurrentTimeStep() const { return m_timestep; }
protected:
    int m_timestep = 0;
};
} // namespace interface
} // namespace core

#endif
