#ifndef COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H
#define COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H

// TODO: make header guards match

#include <vector>
#include <osg/Node>
#include <osg/Vec3>

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Owner.h>

#include "RandomFloatGenerator.h"
#include "RenderToTextureCamera.h"

class SplotchAvatar: public opencover::coVRPlugin, public opencover::ui::Owner {
public:
    SplotchAvatar();

    virtual void preFrame() override;

private:
    //  -- Avatar --
    std::string m_avatarNodeName = "AvatarTrans";
    osg::Node *m_avatarNode = nullptr;

    osg::Matrix getNodeTransform(osg::Node *node) const;

    // -- Splotch shader --
    std::string m_splotchShaderName = "TerroirAvatarStripes";
    std::vector<osg::Vec3> m_splotchPositions;
    float m_splotchRadius = 0.1f;
    RandomFloatGenerator m_randomFloat;
    osg::Vec3 m_referencePosition;
    float m_distanceForSplotches = 100;
    int m_textureSlot = 0;
    int m_nrTextureSlots = 4;

    bool isNearExistingSplotch(const osg::Vec3 &splotch) const;
    void updateSplotchShaderUniforms();

    // -- Render to Texture Camera --
    RenderToTextureCamera *m_rttCamera;

    // moves the camera slightly in front of the avatar so it won't be covered by the mesh
    osg::Vec3 m_offsetRelativeToAvatar = osg::Vec3(50.0, 0.0, 0.0);
    // makes the camera look ahead of the avatar
    osg::Vec3 m_lookAtRelativeToAvatar = osg::Vec3(20.0, 0.0, 0.0);

    bool m_addedSkyNode = false;

};
#endif // COVER_PLUGIN_SPLOTCH_COLOR_PLUGIN_H