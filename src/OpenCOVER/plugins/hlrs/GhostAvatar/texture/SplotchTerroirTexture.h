#ifndef COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H
#define COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H

#include <vector>

#include <osg/Vec3>

#include "TerroirTexture.h"

class SplotchTerroirTexture: public TerroirTexture
{
public:
    SplotchTerroirTexture(float distanceThreshold);

private:
    std::vector<osg::Vec3> m_splotchPositions;

    void onEnoughDistanceCovered() override;
    void updateShader() override;

    osg::Vec3 generateRandomSplotch(int textureSlot);

    void updateShaderUniforms();
    void updateSplotchPositions(const osg::Vec3 &splotch);

};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_SplotchTerroirTexture_H