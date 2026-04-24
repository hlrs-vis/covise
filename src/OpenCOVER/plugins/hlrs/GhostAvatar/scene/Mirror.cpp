#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Texture2D>
#include <osg/Geometry>

#include <cover/coVRPluginSupport.h> // includes cover

#include "Mirror.h"

using namespace opencover;

Mirror::Mirror(const osg::Vec3 &position, float sizeX, float sizeZ)
    : m_position(position)
    , m_sizeX(sizeX)
    , m_sizeY(0.01)
    , m_sizeZ(sizeZ)
    , m_mirrorTransform(new osg::MatrixTransform())
    , m_rttCamera(new RenderToTextureCamera({ 0, -1, 0 }, { 0, 0, 1 }, 1024, 60.0, 1.0, 1.0, 1000.0, true, false))
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
    {
        if (m_reflectedNode)
            m_rttCamera->removeSceneNode(m_reflectedNode);
        m_rttCamera->deinitialize();
    }
}

void Mirror::setReflectedNode(osg::Node *node)
{
    if (m_reflectedNode == node)
        return;

    if (m_rttCamera && m_reflectedNode)
        m_rttCamera->removeSceneNode(m_reflectedNode);

    m_reflectedNode = node;

    if (m_rttCamera && m_reflectedNode)
        m_rttCamera->addSceneNode(m_reflectedNode);
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
    geode->addDrawable(createMirror());

    // create texture from RTT camera's live image
    osg::ref_ptr<osg::Texture2D> mirrorTexture = new osg::Texture2D(m_rttCamera->getImage());
    mirrorTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    mirrorTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);

    // apply texture to the drawable
    osg::ref_ptr<osg::StateSet> stateSet = geode->getOrCreateStateSet();
    stateSet->setTextureAttributeAndModes(0, mirrorTexture, osg::StateAttribute::ON);
    stateSet->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);

    m_mirrorTransform->addChild(geode);
}

osg::ref_ptr<osg::Geometry> Mirror::createMirror() const
{
    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    vertices->push_back(osg::Vec3(0.0f, 0.0f, 0.0f));
    vertices->push_back(osg::Vec3(m_sizeX, 0.0f, 0.0f));
    vertices->push_back(osg::Vec3(m_sizeX, 0.0f, m_sizeZ));
    vertices->push_back(osg::Vec3(0.0f, 0.0f, m_sizeZ));
    geometry->setVertexArray(vertices);

    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array();
    normals->push_back(osg::Vec3(0.0f, -1.0f, 0.0f));
    geometry->setNormalArray(normals, osg::Array::BIND_OVERALL);

    osg::ref_ptr<osg::Vec2Array> texCoords = new osg::Vec2Array();
    texCoords->push_back(osg::Vec2(1.0f, 0.0f));
    texCoords->push_back(osg::Vec2(0.0f, 0.0f));
    texCoords->push_back(osg::Vec2(0.0f, 1.0f));
    texCoords->push_back(osg::Vec2(1.0f, 1.0f));
    geometry->setTexCoordArray(0, texCoords);

    geometry->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    return geometry;
}

osg::Vec3 Mirror::getMirrorCenter() const
{
    return { m_sizeX / 2.f, m_sizeY / 2.0f, m_sizeZ / 2.0f };
}