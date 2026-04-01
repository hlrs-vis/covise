#include <iostream>

#include <osg/MatrixTransform>

#include <cover/coVRShader.h>

#include "TerroirTexture.h"

using namespace opencover;

TerroirTexture::TerroirTexture(const std::string &shaderName)
    : TerroirTexture(shaderName, RenderToTextureCamera(true))
{
}

TerroirTexture::TerroirTexture(const std::string &shaderName, RenderToTextureCamera rttCamera)
    : m_shaderName(shaderName)
    , m_rttCamera(rttCamera)
{
}

void TerroirTexture::applyTexture(osg::Node *node)
{
    m_node = node;
    if (!m_node)
    {
        std::cerr << "ERROR: Node is invalid, cannot apply shader to it!" << std::endl;
        return;
    }
    applyShaderToNode(m_shaderName);

    m_currentPosition = getNodeTransform(m_node).getTrans();
    m_previousPosition = m_currentPosition;

    m_rttCamera.initialize();
    m_rttCamera.update(getNodeTransform(m_node), m_cameraOffset, m_cameraLookAt);
}

void TerroirTexture::onEnoughDistanceCovered()
{
    if (auto cameraScreenshot = m_rttCamera.getScreenshotAsTexture())
        recursivelyAddTextureToSlot(m_node, m_textureSlot, cameraScreenshot);

    m_previousPosition = m_currentPosition;
    m_textureSlot = (m_textureSlot + 1) % m_nrTextureSlots;
}

void TerroirTexture::updateTexture()
{
    if (!m_node)
        return;

    auto nodeTransform = getNodeTransform(m_node);
    m_currentPosition = nodeTransform.getTrans();

    m_rttCamera.update(nodeTransform, m_cameraOffset, m_cameraLookAt);

    if (enoughDistanceCovered())
        onEnoughDistanceCovered();
}

void TerroirTexture::recursivelyAddTextureToSlot(osg::Node *node, int texId, osg::Texture *texture)
{

    auto stateSet = node->getOrCreateStateSet();
    auto textureUniformName = ("texture" + std::to_string(texId + 1)).c_str();

    stateSet->setTextureAttributeAndModes(texId + 1, texture, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
    stateSet->addUniform(new osg::Uniform(textureUniformName, texId + 1));

    if (auto group = node->asGroup())
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
            recursivelyAddTextureToSlot(group->getChild(i), texId, texture);
    }
}

void TerroirTexture::applyShaderToNode(const std::string &shaderName)
{
    if (auto shader = coVRShaderList::instance()->get(shaderName.c_str()))
        shader->apply(m_node);
}

bool TerroirTexture::enoughDistanceCovered()
{
    auto distanceCovered = (m_currentPosition - m_previousPosition).length();
    return distanceCovered >= m_distanceThreshold;
}

osg::Matrix TerroirTexture::getNodeTransform(osg::Node *node) const
{
    if (!node)
        return osg::Matrix::identity();

    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(node);
    if (mt)
        return mt->getMatrix();

    return osg::computeLocalToWorld(node->getParentalNodePaths()[0]);
}