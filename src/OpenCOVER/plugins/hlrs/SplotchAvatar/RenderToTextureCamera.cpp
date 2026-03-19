#include <cover/coVRConfig.h>

#include "RenderToTextureCamera.h"

RenderToTextureCamera::RenderToTextureCamera(): RenderToTextureCamera(512, 90.0, 1.0, 1.0, 1000.0) {};

RenderToTextureCamera::RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear,
                                             double zFar)
: m_viewPortSize{viewPortSize}, m_fovy{fovy}, m_aspectRatio{aspectRatio}, m_zNear{zNear}, m_zFar{zFar}
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
}

osg::Image *RenderToTextureCamera::getImage() const
{
    return m_image.get();
}

osg::Texture2D *RenderToTextureCamera::getTexture() const
{
    return m_texture.get();
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
    if (newZFar != m_zFar) {
        m_zFar = newZFar;
        setProjectionMatrixAsPerspective(m_fovy, m_aspectRatio, m_zNear, m_zFar);
    }
}
