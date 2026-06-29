#include <cassert>
#include <iostream>
#include <utility>

#include <osg/MatrixTransform>

#include <cover/coVRPluginSupport.h> // for Isect::NoMirror
#include <cover/coVRShader.h>

#include "TerroirTexture.h"

using namespace opencover;

namespace
{
void clearNodeMaskBitRecursively(osg::Node *node, unsigned int bit)
{
    if (!node)
        return;

    node->setNodeMask(node->getNodeMask() & ~bit);

    if (auto group = node->asGroup())
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
            clearNodeMaskBitRecursively(group->getChild(i), bit);
    }
}
}

TerroirTexture::TerroirTexture(const std::string &shaderName, float distanceThreshold)
    : TerroirTexture(shaderName, new RenderToTextureCamera(), distanceThreshold)
{
}

TerroirTexture::TerroirTexture(const std::string &shaderName, osg::ref_ptr<RenderToTextureCamera> rttCamera, float distanceThreshold)
    : m_shaderName(shaderName)
    , m_rttCamera(std::move(rttCamera))
    , m_distanceThreshold(distanceThreshold)
{
    assert(distanceThreshold > 0);
}

TerroirTexture::~TerroirTexture()
{
    if (m_rttCamera)
        m_rttCamera->deinitialize();
}

void TerroirTexture::applyTexture(osg::Node *node)
{
    assert(node);

    m_node = node;
    // make sure the node itself is not rendered by the camera
    clearNodeMaskBitRecursively(m_node, Isect::NoMirror);
    applyShaderToNode(m_shaderName);

    m_currentPosition = getNodeTransform(m_node).getTrans();
    setReferencePosition(m_currentPosition);

    if (m_rttCamera)
    {
        m_rttCamera->initialize();
        m_rttCamera->update(getNodeTransform(m_node));
    }
}

void TerroirTexture::updateTexture(const osg::Vec3& offset)
{
    if (!m_node)
    {
        std::cerr << "TerroirTexture::updateTexture: Node is invalid, can't update texture!\n"
                  << "Have you called TerroirTexture::applyTexture before?\n";
        return;
    }

    auto nodeTransform = getNodeTransform(m_node);
    m_currentPosition = nodeTransform.getTrans();

    if (m_rttCamera)
        m_rttCamera->update(nodeTransform * osg::Matrix::translate(offset));

    if (enoughDistanceCovered())
        onEnoughDistanceCovered();

    updateShader();
}

void TerroirTexture::setCameraForwardDir(osg::Vec3 direction)
{
    if (m_rttCamera)
        m_rttCamera->setForwardDirection(direction);
}

void TerroirTexture::setCameraUpDir(osg::Vec3 direction)
{
    if (m_rttCamera)
        m_rttCamera->setUpDirection(direction);
}

void TerroirTexture::onEnoughDistanceCovered()
{
    if (m_rttCamera)
    {
        if (auto cameraScreenshot = m_rttCamera->getScreenshotAsTexture())
            recursivelyAddTextureToSlot(m_node, m_textureSlot, cameraScreenshot);
    }

    setReferencePosition(m_currentPosition);
    m_textureSlot = (m_textureSlot + 1) % m_nrTextureSlots;
}

void TerroirTexture::updateShader()
{
}

osg::ref_ptr<osg::Node> TerroirTexture::getNode()
{
    return m_node;
}

std::string TerroirTexture::getShaderName()
{
    return m_shaderName;
}

osg::Vec3 TerroirTexture::getCurrentPosition()
{
    return m_currentPosition;
}

osg::Vec3 TerroirTexture::getReferencePosition()
{
    return m_referencePosition;
}

void TerroirTexture::setReferencePosition(osg::Vec3 position)
{
    m_referencePosition = position;
}

float TerroirTexture::getDistanceThreshold()
{
    return m_distanceThreshold;
}

void TerroirTexture::setDistanceThreshold(float threshold)
{
    assert(threshold > 0);
    m_distanceThreshold = threshold;
}

int TerroirTexture::getCurrentTextureSlot()
{
    return m_textureSlot;
}

int TerroirTexture::getNumberOfTextureSlots()
{
    return m_nrTextureSlots;
}

void TerroirTexture::applyShaderToNode(const std::string &shaderName)
{
    if (auto shader = coVRShaderList::instance()->get(shaderName.c_str()))
    {
        shader->apply(m_node);
    }
    else
    {
        std::cerr << "TerroirTexture::applyShaderToNode: Could not find shader with the" << "  name " << shaderName << "!" << std::endl;
    }
}

void TerroirTexture::recursivelyAddTextureToSlot(osg::Node *node, int texId, osg::Texture *texture)
{
    assert(node);
    assert(texture);
    assert(texId >= 0 && texId < m_nrTextureSlots);

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

bool TerroirTexture::enoughDistanceCovered()
{
    auto distanceCovered = (m_currentPosition - m_referencePosition).length();
    return distanceCovered >= m_distanceThreshold;
}

osg::Matrix TerroirTexture::getNodeTransform(osg::Node *node) const
{
    assert(node);

    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(node);
    if (mt)
        return mt->getMatrix();

    return osg::computeLocalToWorld(node->getParentalNodePaths()[0]);
}