#include <iostream>
#include <random>

#include <osg/Uniform>

#include <cover/coVRMSController.h>

#include "SplotchTerroirTexture.h"

SplotchTerroirTexture::SplotchTerroirTexture(float distanceThreshold)
    : TerroirTexture("TerroirTextureSplotches", distanceThreshold)
{
    /*
        Since in the CAVE multiple OpenCover instances are running (which would each
        generate their own seeds, resulting in mismatching splotches) we need the master
        to create a seed and send it to the slaves.
    */
    masterGeneratesAndSharesSeed();
}

void SplotchTerroirTexture::onEnoughDistanceCovered()
{
    auto splotch = generateRandomSplotch(getCurrentTextureSlot());
    updateSplotchPositions(splotch);

    TerroirTexture::onEnoughDistanceCovered();
}

void SplotchTerroirTexture::updateShader()
{
    updateShaderUniforms();
}

void SplotchTerroirTexture::masterGeneratesAndSharesSeed()
{
    if (opencover::coVRMSController::instance()->isMaster())
    {
        std::mt19937 engine { std::random_device { }() };
        m_seed = std::uniform_int_distribution<unsigned int> { 0, 1000 }(engine);
        opencover::coVRMSController::instance()->syncData(&m_seed, sizeof(m_seed));
    }
    else
    {
        opencover::coVRMSController::instance()->syncData(&m_seed, sizeof(m_seed));
    }
}

float generateRandomFloat(unsigned int seed, float start, float end)
{
    static std::mt19937 engine { seed };
    return std::uniform_real_distribution<float> { start, end }(engine);
}

osg::Vec3 SplotchTerroirTexture::generateRandomSplotch(int textureSlot)
{
    float texCoordX = generateRandomFloat(m_seed, 0.1f, 0.9f);
    float texCoordY = generateRandomFloat(m_seed, 0.1f, 0.9f);

    return { texCoordX, texCoordY, 1.0f * textureSlot };
}

void SplotchTerroirTexture::updateShaderUniforms()
{
    if (!m_splotchPositions.empty())
    {
        osg::ref_ptr<osg::Uniform> splotchPositionsUniform = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "splotchPositions", m_splotchPositions.size());
        for (size_t i = 0; i < m_splotchPositions.size(); ++i)
            splotchPositionsUniform->setElement(i, m_splotchPositions[i]);
        getNode()->getOrCreateStateSet()->addUniform(splotchPositionsUniform.get(), osg::StateAttribute::ON);
    }
    osg::ref_ptr<osg::Uniform> numSplotchesUniform = new osg::Uniform("numSplotches", int(m_splotchPositions.size()));

    getNode()->getOrCreateStateSet()->addUniform(numSplotchesUniform.get(), osg::StateAttribute::ON);
}

void SplotchTerroirTexture::updateSplotchPositions(const osg::Vec3 &splotch)
{
    if (m_splotchPositions.size() < getNumberOfTextureSlots())
        m_splotchPositions.push_back(splotch);
    else
        m_splotchPositions[getCurrentTextureSlot()] = splotch;
}