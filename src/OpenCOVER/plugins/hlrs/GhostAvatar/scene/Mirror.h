#ifndef COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H
#define COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Quat>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include "../texture/RenderToTextureCamera.h"

class Mirror
{
public:
    Mirror(const osg::Vec3 &position, float sizeX, float sizeZ, const osg::Quat& rotation = {0, 0, 0, 1});
    Mirror(const Mirror &) = delete;
    Mirror &operator=(const Mirror &) = delete;
    Mirror(Mirror &&other) noexcept;
    Mirror &operator=(Mirror &&other) noexcept;
    ~Mirror();

    void setReflectedNode(osg::Node *node);
    void updateView();

private:
    osg::Vec3 m_position;
    osg::Quat m_rotation;
    float m_sizeX, m_sizeY, m_sizeZ;

    osg::ref_ptr<osg::MatrixTransform> m_mirrorTransform;
    osg::ref_ptr<RenderToTextureCamera> m_rttCamera;
    osg::ref_ptr<osg::Node> m_reflectedNode;

    osg::ref_ptr<osg::Geometry> createMirror() const;
    void addMirrorToTransform() const;
    osg::Vec3 getMirrorCenter() const;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H