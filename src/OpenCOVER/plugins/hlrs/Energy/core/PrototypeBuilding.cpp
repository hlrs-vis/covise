#include "PrototypeBuilding.h"
#include "utils/color.h"
#include <memory>
#include <osg/Geode>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/Vec4>
#include <osg/ref_ptr>

namespace {

/**
 * @brief Adds a cylinder between two points.
 * Source: http://www.thjsmith.com/40/cylinder-between-two-points-opengl-c
 *
 * @param start The starting point of the cylinder.
 * @param end The ending point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param cylinderColor The color of the cylinder.
 * @param group The group to which the cylinder will be added.
 */
auto createCylinderBetweenPoints(osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cylinderColor)
{
    osg::ref_ptr geode = new osg::Geode;
    osg::Vec3 center;
    float height;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Material> pMaterial;

    height = (start - end).length();
    center = osg::Vec3((start.x() + end.x()) / 2, (start.y() + end.y()) / 2, (start.z() + end.z()) / 2);

    // This is the default direction for the cylinders to face in OpenGL
    osg::Vec3 z = osg::Vec3(0, 0, 1);

    // Get diff between two points you want cylinder along
    osg::Vec3 p = start - end;

    // Get CROSS product (the axis of rotation)
    osg::Vec3 t = z ^ p;

    // Get angle. length is magnitude of the vector
    double angle = acos((z * p) / p.length());

    // Create a cylinder between the two points with the given radius
    cylinder = new osg::Cylinder(center, radius, height);
    cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));

    cylinderDrawable = new osg::ShapeDrawable(cylinder);
    geode->addDrawable(cylinderDrawable);

    // Set the color of the cylinder that extends between the two points.
    core::utils::color::overrideGeodeColor(geode, cylinderColor);

    return geode;
}
} // namespace

namespace core {

auto PrototypeBuilding::getColor(float val, float max) const
{
    // RGB Colors 1,1,1 = white, 0,0,0 = black
    const auto &colMax = m_attributes.colorMap.max;
    const auto &colMin = m_attributes.colorMap.min;
    max = std::max(max, 1.f);
    float valN = val / max;

    auto col = std::make_unique<osg::Vec4>(colMax.r() * valN + colMin.r() * (1 - valN), colMax.g() * valN + colMin.g() * (1 - valN),
                             colMax.b() * valN + colMin.b() * (1 - valN), colMax.a() * valN + colMin.a() * (1 - valN));
    return col;
}

void PrototypeBuilding::updateColor(const osg::Vec4 &color)
{
    if (auto geode = dynamic_cast<osg::Geode *>(m_drawable.get()))
        utils::color::overrideGeodeColor(geode, color);
}

void PrototypeBuilding::initDrawable()
{
    m_drawable = new osg::Geode;
    const osg::Vec3f bottom(m_attributes.position);
    osg::Vec3f top(bottom);
    top.z() += m_attributes.height;
    m_drawable = createCylinderBetweenPoints(bottom, top, m_attributes.radius, m_attributes.colorMap.defaultColor);
}

std::unique_ptr<osg::Vec4> PrototypeBuilding::getColorInRange(float value, float maxValue)
{
    return getColor(value, maxValue);
}

void PrototypeBuilding::updateDrawable()
{}

void PrototypeBuilding::updateTime(int timestep)
{
    //TODO: update for example the height of the cylinder with each timestep
}

} // namespace core
