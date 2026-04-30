#ifndef COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControlsFactory_H
#define COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControlsFactory_H

#include <memory>
#include <string>

#include "GhostAvatarControls.h"

class GhostAvatarControlsFactory
{
public:
    static std::unique_ptr<GhostAvatarControls> getAvatarByName(const std::string &avatarType);
};

#endif // COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControlsFactory_H