#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Texture2D>
#include <osg/Geometry>

#include <cover/coVRPluginSupport.h> // includes cover

#include "Mirror.h"

using namespace opencover;

Mirror::Mirror(const osg::Vec3 &position, float sizeX, float sizeZ, const osg::Quat &rotation)
    : m_position(position)
    , m_sizeX(sizeX)
    , m_sizeY(0.01)
    , m_sizeZ(sizeZ)
    , m_rotation(rotation)
    , m_mirrorTransform(new osg::MatrixTransform())
    ,  m_rttCamera(new RenderToTextureCamera(m_rotation * osg::Vec3{ 0, -1, 0 }, m_rotation * osg::Vec3{ 0, 0, 1 }, 1024, 60.0, 1.0, 1.0, 1000.0, true, false))
{
    m_mirrorTransform->setMatrix(osg::Matrix::rotate(m_rotation) * osg::Matrix::translate(m_position));
    m_mirrorTransform->setName("GhostAvatarMirror");
    addMirrorToTransform();

    m_rttCamera->initialize();
    updateView();

    if (cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->addChild(m_mirrorTransform);
}

Mirror::Mirror(Mirror &&other) noexcept
    : m_position(other.m_position)
    , m_sizeX(other.m_sizeX)
    , m_sizeY(other.m_sizeY)
    , m_sizeZ(other.m_sizeZ)
    , m_rotation(other.m_rotation)
    , m_mirrorTransform(std::move(other.m_mirrorTransform))
    , m_rttCamera(std::move(other.m_rttCamera))
    , m_reflectedNode(std::move(other.m_reflectedNode))
{
    other.m_position = osg::Vec3();
    other.m_sizeX = 0.f;
    other.m_sizeY = 0.f;
    other.m_sizeZ = 0.f;
    other.m_rotation = osg::Quat();
    other.m_mirrorTransform = nullptr;
    other.m_rttCamera = nullptr;
    other.m_reflectedNode = nullptr;
}

Mirror &Mirror::operator=(Mirror &&other) noexcept
{
    if (this == &other)
        return *this;

    if (m_mirrorTransform && cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->removeChild(m_mirrorTransform);

    if (m_rttCamera)
    {
        if (m_reflectedNode)
            m_rttCamera->removeSceneNode(m_reflectedNode);
        m_rttCamera->deinitialize();
    }

    m_position = other.m_position;
    m_sizeX = other.m_sizeX;
    m_sizeY = other.m_sizeY;
    m_sizeZ = other.m_sizeZ;
    m_rotation = other.m_rotation;
    m_mirrorTransform = std::move(other.m_mirrorTransform);
    m_rttCamera = std::move(other.m_rttCamera);
    m_reflectedNode = std::move(other.m_reflectedNode);

    other.m_position = osg::Vec3();
    other.m_sizeX = 0.f;
    other.m_sizeY = 0.f;
    other.m_sizeZ = 0.f;
    other.m_rotation = osg::Quat();
    other.m_mirrorTransform = nullptr;
    other.m_rttCamera = nullptr;
    other.m_reflectedNode = nullptr;

    return *this;
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
    /*
        While the mirror does not change position, we need to check if the sky node is available and render it
        every frame (since the user can also choose to load the GeoData plugin while COVER is up and running).
    */
    auto translation = m_mirrorTransform->getMatrix().getTrans() + m_rotation * getMirrorCenter();
    m_rttCamera->update(osg::Matrix::translate(translation));
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

osg::Vec3 Mirror::getMirrorCenter() const
{
    return { m_sizeX / 2.f, m_sizeY / 2.0f, m_sizeZ / 2.0f };
}