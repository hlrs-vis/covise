#ifndef COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H
#define COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H

// TODO: make header guards match

#include <string>
#include <vector>

#include <osg/Node>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include <cover/coVRPlugin.h>

#include "RenderToTextureCamera.h"

class SplotchAvatar : public opencover::coVRPlugin
{
public:
    SplotchAvatar();
    SplotchAvatar(RenderToTextureCamera rttCamera);

    void initialize();
    virtual void preFrame() override;

private:
    std::string n_nodeName = "AvatarTrans";
    osg::ref_ptr<osg::Node> m_node;

    osg::Vec3 m_previousPosition;
    float m_distanceThreshold = 100;

    bool enoughDistanceCovered(const osg::Vec3 &currentPosition);
    osg::Matrix getNodeTransform(osg::Node *node) const;
    void updateShaderUniforms();

    // -- Splotch shader --
    std::string m_shaderName = "TerroirAvatarStripes";
    std::vector<osg::Vec3> m_splotchPositions;
    float m_splotchRadius = 0.1f;
    int m_textureSlot = 0;
    int m_nrTextureSlots = 4;

    void applyShaderToNode(const std::string &shaderName);
    bool isNearExistingSplotch(const osg::Vec3 &splotch) const;
    void updateSplotches(const osg::Vec3 &currentPosition);
    void updateSplotchPositions(const osg::Vec3 &splotch);

    // -- Render to Texture Camera --
    RenderToTextureCamera m_rttCamera;

    // moves the camera slightly in front of the avatar so it won't be covered by the mesh
    osg::Vec3 m_cameraOffset = osg::Vec3(50.0, 0.0, 0.0);
    // makes the camera look ahead of the avatar
    osg::Vec3 m_cameraLookAt = osg::Vec3(20.0, 0.0, 0.0);
};
#endif // COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H