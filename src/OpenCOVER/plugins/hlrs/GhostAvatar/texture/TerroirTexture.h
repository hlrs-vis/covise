#ifndef COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTexture_H
#define COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTexture_H

#include <string>

#include <osg/Matrix>
#include <osg/Node>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include "RenderToTextureCamera.h"

class TerroirTexture
{
public:
    TerroirTexture(const std::string &shaderName);
    TerroirTexture(const std::string &shaderName, RenderToTextureCamera rttCamera);

    void applyTexture(osg::Node *node);
    virtual void updateTexture();

protected:
    osg::ref_ptr<osg::Node> m_node;
    std::string m_shaderName;

    osg::Vec3 m_currentPosition;
    osg::Vec3 m_previousPosition;
    float m_distanceThreshold = 100;

    int getCurrentTextureSlot();
    int getNumberOfTextureSlots();

    void applyShaderToNode(const std::string &shaderName);
    void recursivelyAddTextureToSlot(osg::Node *node, int texId, osg::Texture *texture);

    bool enoughDistanceCovered();
    virtual void onEnoughDistanceCovered();
    osg::Matrix getNodeTransform(osg::Node *node) const;

    // -- Render to Texture Camera --
    RenderToTextureCamera m_rttCamera;

    // moves the camera slightly in front of the avatar so it won't be covered by the mesh
    osg::Vec3 m_cameraOffset = osg::Vec3(50.0, 0.0, 0.0);
    // makes the camera look ahead of the avatar
    osg::Vec3 m_cameraLookAt = osg::Vec3(20.0, 0.0, 0.0);

private:
    int m_textureSlot = 0;
    int m_nrTextureSlots = 4;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTexture_H