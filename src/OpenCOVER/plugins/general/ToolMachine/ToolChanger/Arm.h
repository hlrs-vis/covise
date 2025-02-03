#ifndef COVER_PLUGIN_TOOL_MASCHIE_ARM_H
#define COVER_PLUGIN_TOOL_MASCHIE_ARM_H

#include "Utility.h"
#include "Tool.h"

#include <osg/Node>
#include <osg/ref_ptr>
#include <osg/MatrixTransform>
#include <osgAnimation/StackedTransform>
#include <osgAnimation/StackedScaleElement>
#include <osgAnimation/BasicAnimationManager>


#include <array>

constexpr int numArms = 60;

const float lengthScale = 500;
const float pathSideLength = 4 * lengthScale;
const float pathRadius = lengthScale / osg::PI;
const float acceletation = 2; //relative to max speed -> number of seconds until max speed
const float pathLength = 2 * pathSideLength + 2 * pathRadius * osg::PI;
const float pathDistance = pathLength / numArms;
const float relativeStreigthLength = 2 * pathSideLength / pathLength;

const float fPI = static_cast<float>(osg::PI);
// const std::array<float, 4> relativeParts = {pathSideLength/ pathLength, pathRadius * fPI / pathLength, pathSideLength/ pathLength, pathRadius * fPI / pathLength};
const std::array<float, 4> parts = {pathRadius * fPI, pathSideLength, pathRadius * fPI, pathSideLength};
const std::array<float, 4> partPathDistance = {pathDistance, pathDistance, pathDistance, pathDistance};

class Arm{
public:
    Arm(osg::Node *model, osg::Group* parent, int id);
    void position(float offset);
    void play();
    float getDistance() const;
    int id() const;
    bool isPlaying() const;;
    void update();
    ToolModel *tool();
    void giveTool(ToolModel::ptr &&tool);
    ToolModel::ptr takeTool();

private:
    osg::ref_ptr<osg::Node> m_model;
    osg::ref_ptr<osg::MatrixTransform> m_transform;
    int m_id = -1;
    ToolModel::ptr m_tool;
    osg::Group *m_toolParent;
    AnimationManagerFinder m_animation;
    float m_distance = 0;
    const float m_intPlayTime = 100;
    float m_playtime = 0;

};

osg::MatrixTransform* createSphere(const osg::Vec3& pos, float radius);


#endif // COVER_PLUGIN_TOOL_MASCHIE_ARM_H
