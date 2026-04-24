#ifndef COVER_PLUGIN_GHOSTAVATAR_TEXTURE_RenderToTextureCamera_H
#define COVER_PLUGIN_GHOSTAVATAR_TEXTURE_RenderToTextureCamera_H

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
    RenderToTextureCamera(bool renderAvatar = false, bool enableDefaultCamera = false);
    RenderToTextureCamera(osg::Vec3 forwardDirection, osg::Vec3 upDirection, bool renderAvatar = false, bool enableDefaultCamera = false);
    RenderToTextureCamera(osg::Vec3 forwardDirection, osg::Vec3 upDirection, int viewPortSize, double fovy, double aspectRatio, double zNear, double zFar, bool renderAvatar = false, bool enableDebugCamera = false);
    virtual ~RenderToTextureCamera() override;

    /*
        Makes camera render the scene and adds it to the scene graph.
    */
    void initialize();
    void deinitialize();

    osg::ref_ptr<osg::Image> getImage() const;

    /*
        Returns a copy of `m_image`.
    */
    osg::ref_ptr<osg::Image> getScreenshot() const;

    /*
        Calls `getScreenshot` and converts the screenshot to a texture.
    */
    osg::ref_ptr<osg::Texture2D> getScreenshotAsTexture() const;

    /*
       Sets the camera's pose directly from a given node transform.
       The camera position is taken from the transform translation and looks along
       the transformed local forward axis.
   */
    void update(const osg::Matrix &transform);

    osg::Vec3 getForwardDirection();
    osg::Vec3 getUpDirection();

    void setForwardDirection(osg::Vec3 direction);
    void setUpDirection(osg::Vec3 direction);

private:
    osg::ref_ptr<osg::Image> m_image;
    bool m_addedSkyNode = false;

    int m_viewPortSize;
    double m_fovy;
    double m_aspectRatio;
    double m_zNear;
    double m_zFar;

    osg::Vec3 m_forwardDirection;
    osg::Vec3 m_upDirection;

    bool m_renderAvatar;

    bool m_enableDebugCamera;
    osg::ref_ptr<DebugCamera> m_debugCamera;
    bool m_isInitialized = false;

    void configureCamera();
    void configureDebugCamera();
    void configureImage();

    void setZFarToClippingPlane(float scale = 1.0);
    void updateCamera(const osg::Matrix &transform);

    void addChildNode(osg::Node *node);
    void addSkyNode(const char *skyNodeName);
};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_RenderToTextureCamera_H
