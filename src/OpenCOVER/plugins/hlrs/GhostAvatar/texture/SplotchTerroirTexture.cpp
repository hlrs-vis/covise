#include <random>

#include <osg/Texture2D>
#include <osg/Uniform>

#include "SplotchTerroirTexture.h"

SplotchTerroirTexture::SplotchTerroirTexture()
    : TerroirTexture("TerroirTextureSplotches")
{
}

void SplotchTerroirTexture::updateTexture()
{
    TerroirTexture::updateTexture();

    if (!m_node)
        return;

    updateShaderUniforms();
}

float generateRandomFloat(float start, float end)
{
    static thread_local std::mt19937 engine { std::random_device { }() };
    return std::uniform_real_distribution<float> { start, end }(engine);
}

osg::Vec3 generateRandomSplotch(int textureSlot)
{
    float texCoordX = generateRandomFloat(0.1f, 0.9f);
    float texCoordY = generateRandomFloat(0.1f, 0.9f);

    return { texCoordX, texCoordY, 1.0f * textureSlot };
}

void SplotchTerroirTexture::onEnoughDistanceCovered()
{
    auto splotch = generateRandomSplotch(m_textureSlot);
    updateSplotchPositions(splotch);

    TerroirTexture::onEnoughDistanceCovered();
}

void SplotchTerroirTexture::updateSplotchPositions(const osg::Vec3 &splotch)
{
    if (m_splotchPositions.size() <= m_nrTextureSlots)
        m_splotchPositions.push_back(splotch);
    else
        m_splotchPositions[m_textureSlot] = splotch;
}

void SplotchTerroirTexture::updateShaderUniforms()
{
    if (!m_splotchPositions.empty())
    {
        osg::ref_ptr<osg::Uniform> splotchPositionsUniform = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "splotchPositions", m_splotchPositions.size());
        for (size_t i = 0; i < m_splotchPositions.size(); ++i)
            splotchPositionsUniform->setElement(i, m_splotchPositions[i]);
        m_node->getOrCreateStateSet()->addUniform(splotchPositionsUniform.get(), osg::StateAttribute::ON);
    }
    osg::ref_ptr<osg::Uniform> numSplotchesUniform = new osg::Uniform("numSplotches", int(m_splotchPositions.size()));

    m_node->getOrCreateStateSet()->addUniform(numSplotchesUniform.get(), osg::StateAttribute::ON);
}