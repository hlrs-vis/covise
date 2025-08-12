#pragma once
#include <osg/Node>
#include <osg/ref_ptr>

namespace core::interface {
class IDrawable {
 public:
  virtual void initDrawable() = 0;
  virtual void updateDrawable() = 0;
  virtual ~IDrawable() = default;
  auto getDrawable() { return m_drawable; }

 protected:
  osg::ref_ptr<osg::Node> m_drawable;
};
}  // namespace core::interface
