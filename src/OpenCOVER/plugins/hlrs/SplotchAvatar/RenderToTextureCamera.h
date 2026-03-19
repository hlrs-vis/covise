#ifndef COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
#define COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>

// Approach taken from: https://thermalpixel.github.io/osg/2014/02/15/rtt-with-slave-cameras.html
class RenderToTextureCamera: public osg::Camera {
private:
    osg::ref_ptr<osg::Image> m_image;
    osg::ref_ptr<osg::Texture2D> m_texture;

    // camera settings
    int m_viewPortSize;
    double m_fovy;
    double m_aspectRatio;
    double m_zNear;
    double m_zFar;

public:
    RenderToTextureCamera();
    RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear, double zFar);
    virtual ~RenderToTextureCamera() {}

    osg::Image *getImage() const;
    osg::Texture2D *getTexture() const;
    osg::Image *createScreenshot() const;

    void setZFarToClippingPlane();

    int getViewPortSize() const { return m_viewPortSize; }
    double getFovy() const { return m_fovy; }
    double getAspectRatio() const { return m_aspectRatio; }
    double getZNear() const { return m_zNear; }
    double getZFar() const { return m_zFar; }
};

#endif // COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
