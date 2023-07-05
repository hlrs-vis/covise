#ifndef ANIMATED_AVATAR_PLUGIN_CAL3D_PRIVODER_H
#define ANIMATED_AVATAR_PLUGIN_CAL3D_PRIVODER_H

#include "ModelProvider.h"

#include <osgCal/Model>

class Cal3dProvider : public ModelProvider
{
public:
    virtual osg::Node *loadModel(const std::string &filename) override;
    virtual void m_playAnimation(Animation animation, float weight, float delay) override;

private:
osg::ref_ptr<osgCal::Model> m_model;
};



#endif // ANIMATED_AVATAR_PLUGIN_CAL3D_PROVIDER_H
