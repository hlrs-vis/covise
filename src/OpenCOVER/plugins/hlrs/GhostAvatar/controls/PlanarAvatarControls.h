#ifndef COVER_PLUGIN_GHOSTAVATAR_CONTROLS_PlanarAvatarControls_H
#define COVER_PLUGIN_GHOSTAVATAR_CONTROLS_PlanarAvatarControls_H

#include <osg/Quat>

#include "GhostAvatarControls.h"

class PlanarAvatarControls: public GhostAvatarControls
{
public:
    PlanarAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName);

    osg::Vec3 getEyeOffset() const override;
    
    void updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix) override;

private:
    bool flipAvatar(const osg::Matrix &headMatrix) const;
    osg::Quat getFlipRotation(const osg::Matrix &headMatrix) const;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_CONTROLS_PlanarAvatarControls_H