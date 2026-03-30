#include <array>
#include <iostream>
#include <random>
#include <string>

#include <cover/VRSceneGraph.h>
#include <cover/coVRShader.h>

#include <osg/Math>
#include <osg/Texture2D>

#include "SplotchAvatar.h"

using namespace covise;
using namespace opencover;
using namespace ui;

SplotchAvatar::SplotchAvatar()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , Owner(COVER_PLUGIN_NAME, cover->ui)
{
    // create RTT camera with default camera settings
    m_rttCamera = new RenderToTextureCamera(true);
}

void addImageToTextureSlot(osg::Node *node, int texId, osg::Image *image)
{
    auto texture = new osg::Texture2D(image);

    // make sure there is no undefined content at the texture's edges
    texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);

    osg::StateSet *ss = node->getOrCreateStateSet();
    ss->setTextureAttributeAndModes(texId + 1, texture, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

    auto textureUniformName = ("texture" + std::to_string(texId + 1)).c_str();
    ;
    ss->addUniform(new osg::Uniform(textureUniformName, texId + 1));

    osg::Group *group = node->asGroup();
    if (group)
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
            addImageToTextureSlot(group->getChild(i), texId, image);
    }
}

void SplotchAvatar::preFrame()
{
    // initialize avatar node and transform
    // Note: Has to be done here (instead of init()), because the avatar model is
    //       loaded in the GhostAvatar plugin's update function (first frame).
    if (!m_avatarNode)
    {
        m_avatarNode = VRSceneGraph::instance()->findFirstNode<osg::Node>(m_avatarNodeName.c_str());
        if (!m_avatarNode)
            return;

        m_referencePosition = getNodeTransform(m_avatarNode).getTrans();

        if (auto shader = coVRShaderList::instance()->get(m_splotchShaderName.c_str()))
            shader->apply(m_avatarNode);

        m_rttCamera->initialize();
        m_rttCamera->updateCameraPosition(getNodeTransform(m_avatarNode), m_offsetRelativeToAvatar, m_lookAtRelativeToAvatar);
    }

    auto avatarTransform = getNodeTransform(m_avatarNode);

    // Since the GeoData plugin (which adds the sky node to the scene) can also be loaded after OpenCover
    // has finished loading, we have to add it to the camera here in preFrame as soon as it becomes available.
    if (!m_addedSkyNode)
    {
        // The node containing the skyspheres from the GeoData plugin is not part of OBJECTS_ROOT, but a child
        // node of the main scene. This is why we have to pass the scene explictly to the findFirstNode method
        // here (otherwise the node is not found and thus also not rendered by the camera).
        if (auto skyNode = VRSceneGraph::instance()->findFirstNode<osg::Node>("sky", false,
                VRSceneGraph::instance()->getScene()))
        {
            m_rttCamera->addChildNode(skyNode);
            m_addedSkyNode = true;
        }
    }
    m_rttCamera->updateCameraPosition(avatarTransform, m_offsetRelativeToAvatar, m_lookAtRelativeToAvatar);

    // add splotches based on the distance the avatar has covered
    osg::Vec3 avatarPos = avatarTransform.getTrans();
    if (auto distanceCovered = (avatarPos - m_referencePosition).length(); distanceCovered >= m_distanceForSplotches)
    {
        float texCoordX = m_randomFloat.generate(0.1f, 0.9f);
        float texCoordY = m_randomFloat.generate(0.1f, 0.9f);

        osg::Vec3 splotch(texCoordX, texCoordY, m_textureSlot);
        if (!isNearExistingSplotch(splotch))
        {
            if (m_splotchPositions.size() <= m_nrTextureSlots)
                m_splotchPositions.push_back(splotch);
            else
                m_splotchPositions[m_textureSlot] = splotch;
            std::cout << "Added splotch at (" << splotch.x() << ", " << splotch.y() << ") of type " << splotch.z()
                      << std::endl;

            // update texture with current view
            if (auto cameraScreenshot = m_rttCamera->getScreenshot())
                addImageToTextureSlot(m_avatarNode, m_textureSlot, cameraScreenshot);

            m_referencePosition = avatarPos;
            m_textureSlot = (m_textureSlot + 1) % m_nrTextureSlots;
        }
    }

    updateSplotchShaderUniforms();
}

osg::Matrix SplotchAvatar::getNodeTransform(osg::Node *node) const
{
    if (!node)
        return osg::Matrix::identity();

    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(node);
    if (mt)
        return mt->getMatrix();

    return osg::computeLocalToWorld(node->getParentalNodePaths()[0]);
}

bool SplotchAvatar::isNearExistingSplotch(const osg::Vec3 &splotch) const
{
    for (const auto &otherSplotch : m_splotchPositions)
        if ((otherSplotch.x() - splotch.x()) * (otherSplotch.x() - splotch.x()) + (otherSplotch.y() - splotch.y()) * (otherSplotch.y() - splotch.y()) < m_splotchRadius * m_splotchRadius)
            return true;
    return false;
}

void SplotchAvatar::updateSplotchShaderUniforms()
{
    if (!m_splotchPositions.empty())
    {
        osg::ref_ptr<osg::Uniform> splotchPositionsUniform = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "splotchPositions", m_splotchPositions.size());
        for (size_t i = 0; i < m_splotchPositions.size(); ++i)
            splotchPositionsUniform->setElement(i, m_splotchPositions[i]);
        m_avatarNode->getOrCreateStateSet()->addUniform(splotchPositionsUniform.get(), osg::StateAttribute::ON);
    }
    osg::ref_ptr<osg::Uniform> numSplotchesUniform = new osg::Uniform("numSplotches", int(m_splotchPositions.size()));
    osg::ref_ptr<osg::Uniform> splotchRadiusUniform = new osg::Uniform("splotchRadius", m_splotchRadius);

    m_avatarNode->getOrCreateStateSet()->addUniform(numSplotchesUniform.get(), osg::StateAttribute::ON);
    m_avatarNode->getOrCreateStateSet()->addUniform(splotchRadiusUniform.get(), osg::StateAttribute::ON);
}

COVERPLUGIN(SplotchAvatar);
