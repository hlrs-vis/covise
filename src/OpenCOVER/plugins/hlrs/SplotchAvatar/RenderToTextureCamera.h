#ifndef COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
#define COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H

#include <osg/Camera>
#include <osg/Image>
#include <osg/Matrix>
#include <osg/Node>
#include <osg/ref_ptr>
#include <osg/Texture2D>
#include <osg/Vec3>

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
public:
    RenderToTextureCamera(bool enableDefaultCamera = false);
    RenderToTextureCamera(int viewPortSize, double fovy, double aspectRatio, double zNear, double zFar, bool enableDebugCamera = false);
    virtual ~RenderToTextureCamera() { }

    /*
        Makes camera render the scene and adds it to the scene graph.
    */
    void initialize();

    osg::ref_ptr<osg::Image> getImage() const;

    /*
        Returns a copy of `m_image`.
    */
    osg::ref_ptr<osg::Image> getScreenshot() const;

    osg::ref_ptr<osg::Texture2D> getScreenshotAsTexture() const;

    /*
        Sets the camera's z far distance to the far clipping plane distance set in COVER.
    */
    void setZFarToClippingPlane(float scale = 1.0);

    /*
       Sets the position and orientation of the cameras based on a given `transform` matrix.
       Note that the camera will be moved relative to `transform` by `offset`.
       Moreover, the camera will look at the point defined by `lookDirection`,
       which is relative to `transform` as well.
   */
    void update(const osg::Matrix &transform, const osg::Vec3 &offset, const osg::Vec3 &lookAt, const osg::Vec3 &baseUp = { 0.0, 0.0, 1.0 });

private:
    osg::ref_ptr<osg::Image> m_image;
    bool m_addedSkyNode = false;

    int m_viewPortSize;
    double m_fovy;
    double m_aspectRatio;
    double m_zNear;
    double m_zFar;

    bool m_enableDebugCamera;
    osg::ref_ptr<DebugCamera> m_debugCamera;

    void configureCamera();
    void configureDebugCamera();
    void configureImage();

    void addChildNode(osg::Node *node);
    void addSkyNode(const char *skyNodeName);
};

#endif // COVER_PLUGIN_SPLOTCHAVATAR_RENDERTOTEXTURECAMERA_H
