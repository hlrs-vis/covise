#ifndef TOOLMACHINE_TOOLCHANGER_UTILITY_H
#define TOOLMACHINE_TOOLCHANGER_UTILITY_H

#include <osg/NodeVisitor>
#include <osg/Node>
#include <osgAnimation/BasicAnimationManager>
#include <osg/ref_ptr>

#include <string>

struct AnimationManagerFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::BasicAnimationManager> m_am;
    AnimationManagerFinder();
    void apply(osg::Node& node) override;
};

struct TransformFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osg::MatrixTransform> trans;
    TransformFinder(const std::string &name);
    void apply(osg::Node& node) override;
private:
    std::string m_name;
};

osg::MatrixTransform *findTransform(osg::Node *node, const std::string &name);


#endif // TOOLMACHINE_TOOLCHANGER_UTILITY_H