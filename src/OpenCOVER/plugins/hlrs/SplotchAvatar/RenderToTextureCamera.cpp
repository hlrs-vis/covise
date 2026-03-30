#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h> // includes cover

#include "RenderToTextureCamera.h"

RenderToTextureCamera::RenderToTextureCamera()
    : RenderToTextureCamera(512, 90.0, 1.0, 1.0, 1000.0) { };

RenderToTextureCamera::RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear,
    double zFar)
    : m_viewPortSize { viewPortSize }
    , m_fovy { fovy }
    , m_aspectRatio { aspectRatio }
    , m_zNear { zNear }
    , m_zFar { zFar }
{
    // configure the camera
    setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    setRenderOrder(osg::Camera::PRE_RENDER);
    setName("RenderToTextureCamera");
    setViewport(0, 0, m_viewPortSize, m_viewPortSize);
    setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);

    // configure the texture to render to
    // TODO: do we need both texture and image?
    m_texture = new osg::Texture2D();

    m_texture->setSourceFormat(GL_RGBA);
    m_texture->setInternalFormat(GL_RGBA32F_ARB);
    m_texture->setSourceType(GL_FLOAT);

    m_texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    m_texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);

    attach(osg::Camera::COLOR_BUFFER, m_texture.get());

    // configure the image to render to
    m_image = new osg::Image();
    m_image->allocateImage(m_viewPortSize, m_viewPortSize, 1, GL_RGBA, GL_FLOAT);
    attach(osg::Camera::COLOR_BUFFER, m_image.get());

    m_debugCamera = new DebugCamera();
    m_debugCamera->setViewport(0, 0, m_viewPortSize, m_viewPortSize);
    m_debugCamera->setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
}

osg::Image *RenderToTextureCamera::getImage() const
{
    return m_image.get();
}

osg::Texture2D *RenderToTextureCamera::getTexture() const
{
    return m_texture.get();
}

DebugCamera *RenderToTextureCamera::getDebugCamera() const
{
    return m_debugCamera.get();
}

osg::Image *RenderToTextureCamera::createScreenshot() const
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

void RenderToTextureCamera::initialize()
{
    // let camera render the scene
    this->addChild(opencover::cover->getObjectsRoot());
    this->getDebugCamera()->addChild(opencover::cover->getObjectsRoot());

    // add the camera to the scene graph
    opencover::cover->getScene()->addChild(this);
    opencover::cover->getScene()->addChild(this->getDebugCamera());
}

void RenderToTextureCamera::updateCameraPosition(const osg::Matrix &transform, const osg::Vec3 &offset, const osg::Vec3 &lookAt)
{
    osg::Vec3 eye = transform.preMult(offset);
    osg::Vec3 centerWorld = transform.preMult(offset + lookAt);

    osg::Vec3 up = osg::Matrix::transform3x3(osg::Vec3(0.0, 0.0, 1.0), transform);

    this->setViewMatrixAsLookAt(eye, centerWorld, up);
    this->getDebugCamera()->setViewMatrixAsLookAt(eye, centerWorld, up);

    // since the user can choose to change the far clipping plane value at any time
    // we have to check for changes in every frame
    // TODO: find a more efficient way to do this
    this->setZFarToClippingPlane();
    this->getDebugCamera()->setProjectionMatrixAsPerspective(this->getFovy(),
        this->getAspectRatio(),
        this->getZNear(), this->getZFar());
}

void RenderToTextureCamera::addChildNode(osg::Node *node)
{
    if (!node)
        return;

    this->addChild(node);
    this->getDebugCamera()->addChild(node);
}
