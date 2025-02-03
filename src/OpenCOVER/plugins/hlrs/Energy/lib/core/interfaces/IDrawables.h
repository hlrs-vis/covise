#ifndef _CORE_INTERFACES_IDRAWABLES_H
#define _CORE_INTERFACES_IDRAWABLES_H

#include <osg/Node>

namespace core::interface {
class IDrawables {
 public:
  virtual ~IDrawables() = default;
  virtual void initDrawables() = 0;
  virtual void updateDrawables() = 0;
  auto getDrawables() { return m_drawables; }
  auto getDrawable(size_t index) { return m_drawables.at(index); }

 protected:
  std::vector<osg::ref_ptr<osg::Node>> m_drawables;
};
}  // namespace core::interface

#endif
