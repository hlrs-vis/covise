#ifndef _CORE_INTERFACES_ICOLORABLE_H
#define _CORE_INTERFACES_ICOLORABLE_H

#include <osg/Vec4>

namespace core {
namespace interface {
class IColorable {
 public:
  virtual ~IColorable() = default;
  virtual void updateColor(const osg::Vec4 &color) = 0;
};
}  // namespace interface
}  // namespace core

#endif
