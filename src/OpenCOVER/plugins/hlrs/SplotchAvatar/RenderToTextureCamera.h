#ifndef COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
#define COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H

#include <osg/Camera>
#include <osg/Image>
#include <osg/Matrix>
#include <osg/Node>
#include <osg/Texture2D>
#include <osg/Vec3>

// TODO: delete if no longer necessary
class DebugCamera : public osg::Camera
{
public:
    DebugCamera()
    {
        setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        setRenderTargetImplementation(osg::Camera::FRAME_BUFFER);
        setRenderOrder(osg::Camera::POST_RENDER);

        setDrawBuffer(GL_BACK);
        setReadBuffer(GL_BACK);

        setReferenceFrame(osg::Transform::ABSOLUTE_RF);

        setName("RenderToTextureDebugCamera");
    }
    virtual ~DebugCamera() { }
};

// Approach taken from: https://thermalpixel.github.io/osg/2014/02/15/rtt-with-slave-cameras.html
class RenderToTextureCamera : public osg::Camera
{
private:
    osg::ref_ptr<osg::Image> m_image;
    osg::ref_ptr<osg::Texture2D> m_texture;
    osg::ref_ptr<DebugCamera> m_debugCamera;

    // camera settings
    int m_viewPortSize;
    double m_fovy;
    double m_aspectRatio;
    double m_zNear;
    double m_zFar;

public:
    RenderToTextureCamera();
    RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear, double zFar);
    virtual ~RenderToTextureCamera() { }

    osg::Image *getImage() const;
    osg::Texture2D *getTexture() const;
    DebugCamera *getDebugCamera() const;
    osg::Image *createScreenshot() const;

    void setZFarToClippingPlane();

    int getViewPortSize() const { return m_viewPortSize; }
    double getFovy() const { return m_fovy; }
    double getAspectRatio() const { return m_aspectRatio; }
    double getZNear() const { return m_zNear; }
    double getZFar() const { return m_zFar; }

    void initialize();

    /*
       Sets the position and orientation of the cameras based on a given `transform` matrix.
       Note that the camera will be moved relative to `transform` by `offset`.
       Moreover, the camera will look at the point defined by `lookDirection`,
       which is relative to `transform` as well.
   */
    void updateCameraPosition(const osg::Matrix &transform, const osg::Vec3 &offset, const osg::Vec3 &lookAt);

    void addChildNode(osg::Node *node);
};

#endif // COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
