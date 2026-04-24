#ifndef COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H
#define COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include "../texture/RenderToTextureCamera.h"

class Mirror
{
public:
    Mirror(const osg::Vec3 &position, float sizeX, float sizeZ);
    ~Mirror();

    void updateView();

private:

    osg::Vec3 m_position;
    float m_sizeX, m_sizeY, m_sizeZ;

    osg::ref_ptr<osg::MatrixTransform> m_mirrorTransform;
    osg::ref_ptr<RenderToTextureCamera> m_rttCamera;

    osg::ref_ptr<osg::Box> createMirror() const;
    void addMirrorToTransform() const;
    osg::Vec3 getMirrorCenter() const;
};

#endif // #ifndef COVER_PLUGIN_GHOSTAVATAR_Scene_Mirror_H