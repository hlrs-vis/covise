#include "color.h"
#include <osg/Vec4>

namespace core::utils::color
{

core::interface::Color getTrafficLightColor(float val, float max)
{
    core::interface::Color red(1.0f, 0.0f, 0.0f, 1.0f);
    core::interface::Color yellow(1.0f, 1.0f, 0.0f, 1.0f);
    core::interface::Color green(0.0f, 1.0f, 0.0f, 1.0f);

    float greenThreshold = max * 0.33f;
    float yellowThreshold = max * 0.66f;

    if (val <= greenThreshold)
    {
        return green;
    }
    else if (val <= yellowThreshold)
    {
        return yellow;
    }
    else
    {
        return red;
    }
};

auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask)
{
    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setDiffuse(faceMask, color);
    mat->setAmbient(faceMask, color);
    return mat;
}
void overrideGeodeColor(osg::Geode *geode, const core::interface::Color &color,
    osg::Material::Face faceMask)
{
    overrideGeodeColor(geode, osg::Vec4(color.r, color.g, color.b, color.a));
}

void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
    osg::Material::Face faceMask)
{
    auto mat = createMaterial(color, faceMask);
    overrideGeodeMaterial(geode, mat);
}

void overrideGeodeMaterial(osg::Geode *geode, osg::Material *material)
{
    if (!geode)
        return;
    auto stateSet = geode->getOrCreateStateSet();
    if (!stateSet)
        return;
    stateSet->setAttribute(material, osg::StateAttribute::OVERRIDE);
}
} // namespace core::utils::color
