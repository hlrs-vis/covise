#include <filesystem>
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
    static const auto modelsDir = std::filesystem::path(__FILE__).parent_path() / "models";
    static const std::map<std::string, AvatarFactory> factory = {
        { "ghost", []
            {
                auto fbxPath = (modelsDir / "ghost_avatar.fbx").string();
                return std::make_unique<TestAvatarControls>(fbxPath, "RightArm", "");
            } },
        { "planar", []
            {
                auto fbxPath = (modelsDir / "PLANEE3.fbx").string();
                return std::make_unique<PlanarAvatarControls>(fbxPath, "Arm", "Head");
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