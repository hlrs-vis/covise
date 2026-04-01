#ifndef COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H
#define COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H

#include <vector>

#include "TerroirTexture.h"

class SplotchTerroirTexture: public TerroirTexture
{
public:
    SplotchTerroirTexture();

    void updateTexture() override;

private:
    std::vector<osg::Vec3> m_splotchPositions;
    int m_textureSlot = 0;
    int m_nrTextureSlots = 4;

    void onEnoughDistanceCovered() override;
    void updateSplotchPositions(const osg::Vec3 &splotch);
    void updateShaderUniforms();

};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H