#pragma once
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/interfaces/ITimedependable.h>

#include <osg/Node>
#include <osg/ref_ptr>

typedef osg::ref_ptr<osg::Node> Drawable;
typedef std::vector<Drawable> Drawables;

class OsgBuildingImpl
    : public core::interface::IBuilding<Drawable, std::vector> {
 public:
  Drawables &getDrawables() override { return m_drawables; };
  Drawable &getDrawable(size_t idx) override {
    if (idx >= m_drawables.size()) {
      throw std::out_of_range("Drawable idx out of range for citygmlbuilding.");
    }
    return m_drawables[idx];
  }

 protected:
  Drawables m_drawables;
};

class OsgBuildingTimedependImpl : public OsgBuildingImpl, public core::interface::ITimedependable {
    public:
     virtual ~OsgBuildingTimedependImpl() = default;
};
