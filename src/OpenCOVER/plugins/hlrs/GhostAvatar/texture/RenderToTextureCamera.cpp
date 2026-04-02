#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h> // includes cover
#include <cover/VRSceneGraph.h>

#include "RenderToTextureCamera.h"

RenderToTextureCamera::RenderToTextureCamera(bool enableDefaultCamera)
    : RenderToTextureCamera(512, 90.0, 1.0, 1.0, 1000.0, enableDefaultCamera) { };

RenderToTextureCamera::RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear,
    double zFar, bool enableDebugCamera)
    : m_viewPortSize { viewPortSize }
    , m_fovy { fovy }
    , m_aspectRatio { aspectRatio }
    , m_zNear { zNear }
    , m_zFar { zFar }
    , m_enableDebugCamera { enableDebugCamera }
{
    configureCamera();
    configureDebugCamera();
    configureImage();
}

void RenderToTextureCamera::initialize()
{
    addChild(opencover::cover->getObjectsRoot());
    opencover::cover->getScene()->addChild(this);
    setCullMask(opencover::Isect::NoMirror);

    if (m_debugCamera)
    {
        m_debugCamera->addChild(opencover::cover->getObjectsRoot());
        opencover::cover->getScene()->addChild(m_debugCamera);
        m_debugCamera->setCullMask(opencover::Isect::NoMirror);
    }
}

osg::ref_ptr<osg::Image> RenderToTextureCamera::getImage() const
{
    return m_image.get();
}

osg::ref_ptr<osg::Image> RenderToTextureCamera::getScreenshot() const
{
    if (auto image = getImage())
        return new osg::Image(*image, osg::CopyOp::DEEP_COPY_ALL);
    return nullptr;
}

osg::ref_ptr<osg::Texture2D> RenderToTextureCamera::getScreenshotAsTexture() const
{
    if (auto screenshot = getScreenshot())
    {
        osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D(screenshot);

        // make sure there is no undefined content at the texture's edges
        texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
        texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);

        return texture;
    }

    return nullptr;
}

void RenderToTextureCamera::setZFarToClippingPlane(float scale)
{
    auto newZFar = opencover::coVRConfig::instance()->farClip() * scale;
    if (newZFar != m_zFar)
    {
        m_zFar = newZFar;
        setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
    }
}

void RenderToTextureCamera::update(const osg::Matrix &transform, const osg::Vec3 &offset, const osg::Vec3 &lookAt, const osg::Vec3 &baseUp)
{
    osg::Vec3 eye = transform.preMult(offset);
    osg::Vec3 centerWorld = transform.preMult(offset + lookAt);
    osg::Vec3 up = osg::Matrix::transform3x3(baseUp, transform);

    setViewMatrixAsLookAt(eye, centerWorld, up);

    // Since the user can choose to change the far clipping plane value at any time
    // we have to check for changes every update.
    // TODO: find a more efficient way to do this
    setZFarToClippingPlane(10.);

    // Since the GeoData plugin (which adds the sky node to the scene) can also be loaded after OpenCover
    // has finished loading, we have to add it to the camera here as soon as it becomes available.
    if (!m_addedSkyNode)
        addSkyNode("sky");

    if (m_enableDebugCamera)
    {
        m_debugCamera->setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
        m_debugCamera->setViewMatrixAsLookAt(eye, centerWorld, up);
    }
}

void RenderToTextureCamera::configureCamera()
{
    setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    setRenderOrder(osg::Camera::PRE_RENDER);
    setName("RenderToTextureCamera");
    setViewport(0, 0, m_viewPortSize, m_viewPortSize);
    setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
}

void RenderToTextureCamera::configureImage()
{
    m_image = new osg::Image();
    m_image->allocateImage(m_viewPortSize, m_viewPortSize, 1, GL_RGBA, GL_FLOAT);
    attach(osg::Camera::COLOR_BUFFER, m_image.get());
}

void RenderToTextureCamera::configureDebugCamera()
{
    if (m_enableDebugCamera)
    {
        m_debugCamera = new DebugCamera();
        m_debugCamera->setViewport(0, 0, m_viewPortSize, m_viewPortSize);
        m_debugCamera->setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
    }
}

void RenderToTextureCamera::addChildNode(osg::Node *node)
{
    if (!node)
        return;

    addChild(node);

    if (m_enableDebugCamera)
        m_debugCamera->addChild(node);
}

void RenderToTextureCamera::addSkyNode(const char *skyNodeName)
{
    // The node containing the skyspheres from the GeoData plugin is not part of OBJECTS_ROOT, but a child
    // node of the main scene. This is why we have to pass the scene explictly to the findFirstNode method
    // here (otherwise the node is not found and thus also not rendered by the camera).
    if (auto skyNode = opencover::VRSceneGraph::instance()->findFirstNode<osg::Node>(skyNodeName, false,
            opencover::VRSceneGraph::instance()->getScene()))
    {
        addChildNode(skyNode);
        m_addedSkyNode = true;
    }
}