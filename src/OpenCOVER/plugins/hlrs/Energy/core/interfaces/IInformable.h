#ifndef _CORE_INTERFACES_IINFORMABLE_H
#define _CORE_INTERFACES_IINFORMABLE_H

#include <osg/Geode>
#include <osg/ref_ptr>
#include "IDrawable.h"

namespace core {
namespace interface {
template<typename Info>
class IInformable: public IDrawable {
public:
    virtual void showInfo() = 0;
    virtual void hideInfo() = 0;
    virtual void initDrawable() = 0;
    virtual void updateDrawable() = 0;
    virtual void updateInfo(const Info &info) = 0;
protected:
    Info m_info;
};
} // namespace interface
} // namespace core

#endif