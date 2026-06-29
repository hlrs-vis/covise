#ifndef COVER_PLUGIN_GHOSTAVATAR_CONTROLS_TestAvatarControls_H
#define COVER_PLUGIN_GHOSTAVATAR_CONTROLS_TestAvatarControls_H

#include "GhostAvatarControls.h"

class TestAvatarControls : public GhostAvatarControls
{
public:
    TestAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName);

    osg::Vec3 getEyeOffset() const override;

    void updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix) override;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_CONTROLS_TestAvatarControls_H