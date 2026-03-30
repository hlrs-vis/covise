#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h> // includes cover

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

    if (m_debugCamera)
    {
        m_debugCamera->addChild(opencover::cover->getObjectsRoot());
        opencover::cover->getScene()->addChild(m_debugCamera);
    }
}

osg::Image *RenderToTextureCamera::getImage() const
{
    return m_image.get();
}

osg::Image *RenderToTextureCamera::getScreenshot() const
{
    if (auto image = getImage())
        return new osg::Image(*image, osg::CopyOp::DEEP_COPY_ALL);
    return nullptr;
}

void RenderToTextureCamera::setZFarToClippingPlane()
{
    auto newZFar = opencover::coVRConfig::instance()->farClip() * 10;
    if (newZFar != m_zFar)
    {
        m_zFar = newZFar;
        setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
    }
}

void RenderToTextureCamera::updateCameraPosition(const osg::Matrix &transform, const osg::Vec3 &offset, const osg::Vec3 &lookAt)
{
    osg::Vec3 eye = transform.preMult(offset);
    osg::Vec3 centerWorld = transform.preMult(offset + lookAt);
    osg::Vec3 up = osg::Matrix::transform3x3(osg::Vec3(0.0, 0.0, 1.0), transform);

    setViewMatrixAsLookAt(eye, centerWorld, up);

    // since the user can choose to change the far clipping plane value at any time
    // we have to check for changes in every frame
    // TODO: find a more efficient way to do this
    setZFarToClippingPlane();

    if (m_enableDebugCamera)
    {
        m_debugCamera->setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
        m_debugCamera->setViewMatrixAsLookAt(eye, centerWorld, up);
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
