#include <cover/coVRShader.h>
#include <cover/VRSceneGraph.h>

#include <osg/Texture2D>
#include <osg/Uniform>

#include <random>

#include "SplotchTexture.h"

using namespace opencover;

SplotchTexture::SplotchTexture(const std::string &nodeName)
    : SplotchTexture(nodeName, RenderToTextureCamera(true))
{
}

SplotchTexture::SplotchTexture(const std::string &nodeName, RenderToTextureCamera rttCamera)
    : m_nodeName(nodeName)
    , m_rttCamera(rttCamera)
{
}

void SplotchTexture::initialize()
{
    m_node = VRSceneGraph::instance()->findFirstNode<osg::Node>(m_nodeName.c_str());
    if (!m_node)
        return;

    m_previousPosition = getNodeTransform(m_node).getTrans();

    applyShaderToNode(m_shaderName);

    m_rttCamera.initialize();
    m_rttCamera.update(getNodeTransform(m_node), m_cameraOffset, m_cameraLookAt);
}

void SplotchTexture::update()
{
    // initialize avatar node and transform
    // Note: Has to be done here (instead of init()), because the avatar model is
    //       loaded in the GhostAvatar plugin's update function (first frame).
    if (!m_node)
    {
        initialize();
        return;
    }

    auto nodeTransform = getNodeTransform(m_node);

    m_rttCamera.update(nodeTransform, m_cameraOffset, m_cameraLookAt);

    updateSplotches(nodeTransform.getTrans());
    updateShaderUniforms();
}

bool SplotchTexture::enoughDistanceCovered(const osg::Vec3 &position)
{
    auto distanceCovered = (position - m_previousPosition).length();
    return distanceCovered >= m_distanceThreshold;
}

osg::Matrix SplotchTexture::getNodeTransform(osg::Node *node) const
{
    if (!node)
        return osg::Matrix::identity();

    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(node);
    if (mt)
        return mt->getMatrix();

    return osg::computeLocalToWorld(node->getParentalNodePaths()[0]);
}

void SplotchTexture::updateShaderUniforms()
{
    if (!m_splotchPositions.empty())
    {
        osg::ref_ptr<osg::Uniform> splotchPositionsUniform = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "splotchPositions", m_splotchPositions.size());
        for (size_t i = 0; i < m_splotchPositions.size(); ++i)
            splotchPositionsUniform->setElement(i, m_splotchPositions[i]);
        m_node->getOrCreateStateSet()->addUniform(splotchPositionsUniform.get(), osg::StateAttribute::ON);
    }
    osg::ref_ptr<osg::Uniform> numSplotchesUniform = new osg::Uniform("numSplotches", int(m_splotchPositions.size()));
    osg::ref_ptr<osg::Uniform> splotchRadiusUniform = new osg::Uniform("splotchRadius", m_splotchRadius);

    m_node->getOrCreateStateSet()->addUniform(numSplotchesUniform.get(), osg::StateAttribute::ON);
    m_node->getOrCreateStateSet()->addUniform(splotchRadiusUniform.get(), osg::StateAttribute::ON);
}

void SplotchTexture::applyShaderToNode(const std::string &shaderName)
{
    if (auto shader = coVRShaderList::instance()->get(shaderName.c_str()))
        shader->apply(m_node);
}

bool SplotchTexture::isNearExistingSplotch(const osg::Vec3 &splotch) const
{
    for (const auto &otherSplotch : m_splotchPositions)
        if ((otherSplotch.x() - splotch.x()) * (otherSplotch.x() - splotch.x()) + (otherSplotch.y() - splotch.y()) * (otherSplotch.y() - splotch.y()) < m_splotchRadius * m_splotchRadius)
            return true;
    return false;
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

void addToTextureSlot(osg::Node *node, int texId, osg::Texture *texture)
{

    auto stateSet = node->getOrCreateStateSet();
    auto textureUniformName = ("texture" + std::to_string(texId + 1)).c_str();

    stateSet->setTextureAttributeAndModes(texId + 1, texture, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
    stateSet->addUniform(new osg::Uniform(textureUniformName, texId + 1));

    if (auto group = node->asGroup())
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
            addToTextureSlot(group->getChild(i), texId, texture);
    }
}

void SplotchTexture::updateSplotches(const osg::Vec3 &currentPosition)
{
    if (enoughDistanceCovered(currentPosition))
    {
        auto splotch = generateRandomSplotch(m_textureSlot);
        if (!isNearExistingSplotch(splotch))
        {
            updateSplotchPositions(splotch);

            // update texture with current view
            if (auto cameraScreenshot = m_rttCamera.getScreenshotAsTexture())
                addToTextureSlot(m_node, m_textureSlot, cameraScreenshot);

            m_previousPosition = currentPosition;
            m_textureSlot = (m_textureSlot + 1) % m_nrTextureSlots;
        }
    }
}

void SplotchTexture::updateSplotchPositions(const osg::Vec3 &splotch)
{
    if (m_splotchPositions.size() <= m_nrTextureSlots)
        m_splotchPositions.push_back(splotch);
    else
        m_splotchPositions[m_textureSlot] = splotch;
}
