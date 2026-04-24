#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>

#include <cover/coVRPluginSupport.h> // includes cover

#include "Mirror.h"

using namespace opencover;

Mirror::Mirror(const osg::Vec3 &position, float sizeX, float sizeZ)
    : m_position(position)
    , m_sizeX(sizeX)
    , m_sizeY(0.01)
    , m_sizeZ(sizeZ)
    , m_mirrorTransform(new osg::MatrixTransform())
    , m_rttCamera(new RenderToTextureCamera({ 0, -1, 0 }, { 0, 0, 1 }, 1024, 45.0, 1.0, 1.0, 1000.0, true, false))
{
    m_rttCamera->initialize();
    updateView();

    m_mirrorTransform->setMatrix(osg::Matrix::translate(m_position));
    m_mirrorTransform->setName("GhostAvatarMirror");
    addMirrorToTransform();

    if (cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->addChild(m_mirrorTransform);
}

Mirror::~Mirror()
{
    if (m_mirrorTransform && cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->removeChild(m_mirrorTransform);

    if (m_rttCamera)
        m_rttCamera->deinitialize();
}

void Mirror::updateView()
{
    auto translation = m_mirrorTransform->getMatrix().getTrans() + getMirrorCenter();
    m_rttCamera->update(osg::Matrix::translate(translation));
}

void Mirror::addMirrorToTransform() const
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->setName("GhostAvatarMirrorGeode");
    geode->addDrawable(new osg::ShapeDrawable(createMirror()));

    // Create texture from RTT camera's live image
    osg::ref_ptr<osg::Texture2D> mirrorTexture = new osg::Texture2D(m_rttCamera->getImage());
    mirrorTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    mirrorTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);

    // Apply texture to the drawable
    osg::ref_ptr<osg::StateSet> stateSet = geode->getOrCreateStateSet();
    stateSet->setTextureAttributeAndModes(0, mirrorTexture, osg::StateAttribute::ON);

    m_mirrorTransform->addChild(geode);
}

osg::ref_ptr<osg::Box> Mirror::createMirror() const
{
    return new osg::Box(getMirrorCenter(), m_sizeX, m_sizeY, m_sizeZ);
}

osg::Vec3 Mirror::getMirrorCenter() const
{
    return { m_sizeX / 2.f, m_sizeY / 2.0f, m_sizeZ / 2.0f };
}