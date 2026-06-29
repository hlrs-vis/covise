#ifndef COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTextureFactory_H
#define COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTextureFactory_H

#include <memory>
#include <string>

#include "TerroirTexture.h"

class TerroirTextureFactory
{
public:
    static std::unique_ptr<TerroirTexture> getTextureByName(const std::string &textureType, float distanceThreshold);
};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTextureFactory_H