#include <functional>
#include <iostream>
#include <map>

#include "GhostAvatarControlsFactory.h"
#include "PlanarAvatarControls.h"
#include "TestAvatarControls.h"

namespace
{
using AvatarFactory = std::function<std::unique_ptr<GhostAvatarControls>()>;

const std::map<std::string, AvatarFactory> &createGhostAvatarControlsFactory()
{
    static const std::map<std::string, AvatarFactory> factory = {
        { "ghost", []
            {
                return std::make_unique<TestAvatarControls>("/data/STARTS-ECHO/Avatars/shaderTests/ghost_cave_minimal_fix.fbx", "RightArm", "");
            } },
        { "planar", []
            {
                return std::make_unique<PlanarAvatarControls>("/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6_fix.fbx", "Arm", "Head");
            } },
    };

    return factory;
}
}

std::unique_ptr<GhostAvatarControls> GhostAvatarControlsFactory::getAvatarByName(const std::string &avatarType)
{
    const auto &controlsFactory = createGhostAvatarControlsFactory();
    auto selected = controlsFactory.find(avatarType);
    if (selected == controlsFactory.end())
    {
        std::cerr << "Unknown avatar type: " << avatarType << ". Defaulting to planar avatar.\n";
        selected = controlsFactory.find("planar");
    }

    return selected->second();
}