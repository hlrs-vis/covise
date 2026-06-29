#include <functional>
#include <iostream>
#include <map>

#include "SplotchTerroirTexture.h"
#include "StripesTerroirTexture.h"
#include "TerroirTextureFactory.h"

namespace
{
using TextureCreator = std::function<std::unique_ptr<TerroirTexture>()>;

const std::map<std::string, TextureCreator> &createTerroirTextureFactory(float distanceThreshold)
{
    static const std::map<std::string, TextureCreator> factory = {
        { "splotches", [distanceThreshold]
            { return std::make_unique<SplotchTerroirTexture>(distanceThreshold); } },
        { "stripes", [distanceThreshold]
            { return std::make_unique<StripesTerroirTexture>(distanceThreshold); } }
    };

    return factory;
}
}

std::unique_ptr<TerroirTexture> TerroirTextureFactory::getTextureByName(const std::string &textureType, float distanceThreshold)
{
    const auto &textureFactory = createTerroirTextureFactory(distanceThreshold);
    auto selected = textureFactory.find(textureType);
    if (selected == textureFactory.end())
    {
        std::cerr << "Unknown texture type: " << textureType << ". Defaulting to splotches texture.\n";
        selected = textureFactory.find("splotches");
    }

    return selected->second();
}