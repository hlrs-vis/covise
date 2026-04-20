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
    TerroirTexture(const std::string &shaderName, float distanceThreshold);
    TerroirTexture(const std::string &shaderName, osg::ref_ptr<RenderToTextureCamera> rttCamera, float distanceThreshold);
    virtual ~TerroirTexture();

    void applyTexture(osg::Node *node);
    void updateTexture(const osg::Vec3 &offset = { 0, 0, 0 });

    void setCameraForwardDir(osg::Vec3 direction);
    void setCameraUpDir(osg::Vec3 direction);

protected:
    virtual void onEnoughDistanceCovered();
    virtual void updateShader();

    osg::ref_ptr<osg::Node> getNode();

    std::string getShaderName();

    osg::Vec3 getCurrentPosition();

    osg::Vec3 getReferencePosition();
    void setReferencePosition(osg::Vec3 position);

    float getDistanceThreshold();
    void setDistanceThreshold(float threshold);

    int getCurrentTextureSlot();
    int getNumberOfTextureSlots();

    void applyShaderToNode(const std::string &shaderName);
    void recursivelyAddTextureToSlot(osg::Node *node, int texId, osg::Texture *texture);

    bool enoughDistanceCovered();
    osg::Matrix getNodeTransform(osg::Node *node) const;

    // -- Render to Texture Camera --
    osg::ref_ptr<RenderToTextureCamera> m_rttCamera;

private:
    osg::ref_ptr<osg::Node> m_node;
    std::string m_shaderName;

    osg::Vec3 m_currentPosition;
    osg::Vec3 m_referencePosition;

    float m_distanceThreshold = 100;

    int m_textureSlot = 0;
    int m_nrTextureSlots = 4;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_TEXTURE_TerroirTexture_H