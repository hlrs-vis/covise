#ifndef ANIMATED_AVATAR_PLUGIN_MODEL_PROVIDER_H
#define ANIMATED_AVATAR_PLUGIN_MODEL_PROVIDER_H

#include <string>
#include <osg/Node>

class ModelProvider
{
public:
    enum Animation
    {
        Idle, Walk, Run, WalkBack, RunBack, TurnLeft, TurRight, Wave, LAST
    };

    virtual ~ModelProvider() = default;
    virtual osg::Node *loadModel(const std::string &filename) = 0;
    void playAnimation(Animation animation, float weight, float delay)
    {
        m_playAnimation(animation, weight, delay);
        m_currentAnimation = animation;
    }
    Animation currentAnimation() const{
        return m_currentAnimation;
    }
protected:
    virtual void m_playAnimation(Animation animation, float weight, float delay) = 0;
private:
    Animation m_currentAnimation = Idle;

};

#endif // ANIMATED_AVATAR_PLUGIN_MODEL_PROVIDER_H