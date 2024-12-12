#ifndef _CORE_INTERFACES_IDRAWABLE_H
#define _CORE_INTERFACES_IDRAWABLE_H

#include <osg/Node>
#include <osg/ref_ptr>

namespace core {
namespace interface {
class IDrawable {
 public:
  virtual void initDrawable() = 0;
  virtual void updateDrawable() = 0;
  virtual ~IDrawable() = default;
  auto getDrawable() { return m_drawable; }

 protected:
  osg::ref_ptr<osg::Node> m_drawable;
};
}  // namespace interface
}  // namespace core

#endif
