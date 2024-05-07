#ifndef _CORE_PROTOTYPEBUILDING_H
#define _CORE_PROTOTYPEBUILDING_H

#include "interfaces/IBuilding.h"
#include <memory>
#include <osg/Vec3>
#include <osg/Vec4>

namespace core {
struct CylinderColormap {
    CylinderColormap(const osg::Vec4 &max, const osg::Vec4 &min, const osg::Vec4 &def)
    : max(max), min(min), defaultColor(def)
    {}
    osg::Vec4 max;
    osg::Vec4 min;
    osg::Vec4 defaultColor;
};

struct CylinderAttributes {
    CylinderAttributes(const float &rad, const float &height, const osg::Vec3 &pos, const CylinderColormap &colorMap)
    : radius(rad), height(height), position(pos), colorMap(colorMap)
    {}
    CylinderAttributes(const float &rad, const float &height, const osg::Vec4 &maxCol, const osg::Vec4 &minCol,
                       const osg::Vec3 &pos, const osg::Vec4 &defaultCol)
    : CylinderAttributes(rad, height, pos, CylinderColormap(maxCol, minCol, defaultCol))
    {}
    CylinderAttributes(const float &rad, const float &height, const osg::Vec4 &maxCol, const osg::Vec4 &minCol,
                       const osg::Vec4 &defaultCol)
    : CylinderAttributes(rad, height, osg::Vec3(0, 0, 0), CylinderColormap(maxCol, minCol, defaultCol))
    {}
    float radius;
    float height;
    osg::Vec3 position;
    CylinderColormap colorMap;
};

class PrototypeBuilding: public interface::IBuilding {
public:
    PrototypeBuilding(const CylinderAttributes &cylinderAttributes): m_attributes(cylinderAttributes){};
    void initDrawable() override;
    void move(const osg::Vec3 &pos) override;
    void updateColor(const osg::Vec4 &color) override;
    void updateTime(int timestep) override;
    void updateDrawable() override;
    std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override;

private:
    auto getColor(float val, float max) const;

    CylinderAttributes m_attributes;
};
} // namespace core

#endif