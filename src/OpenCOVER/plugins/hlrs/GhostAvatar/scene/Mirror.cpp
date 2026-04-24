#include <osg/ShapeDrawable>

#include <cover/coVRPluginSupport.h> // includes cover

#include "Mirror.h"

using namespace opencover;

Mirror::Mirror(const osg::Vec3 &position, float sizeX, float sizeZ)
    : m_position(position)
    , m_sizeX(sizeX)
    , m_sizeY(0.01)
    , m_sizeZ(sizeZ)
    , m_geode(new osg::Geode())
    , m_rttCamera(new RenderToTextureCamera(true))
{
    m_geode->setName("GhostAvatarMirror");
    addMirrorToGeode();

    if (cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->addChild(m_geode);

    m_rttCamera->initialize();
}

Mirror::~Mirror()
{
    if (m_geode && cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->removeChild(m_geode);

    if (m_rttCamera)
        m_rttCamera->deinitialize();
}

void Mirror::addMirrorToGeode()
{
    auto boxDrawable = new osg::ShapeDrawable(createMirror());
    m_geode->addDrawable(boxDrawable);
}

osg::ref_ptr<osg::Box> Mirror::createMirror() const
{
    return new osg::Box(getMirrorCenter(), m_sizeX, m_sizeY, m_sizeZ);
}

osg::Vec3 Mirror::getMirrorCenter() const
{
    return { m_position.x() + (m_sizeX / 2.f), m_position.y() + (m_sizeY / 2.0f), m_position.z() + (m_sizeZ / 2.0f) };
}