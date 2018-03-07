#include "ReferencePosition.h"


ReferencePosition::ReferencePosition():
    road(NULL)
{
}

ReferencePosition::~ReferencePosition()
{}

void ReferencePosition::initFromLane(std::string init_roadId, int init_laneId, double init_s, RoadSystem* system)
{
    roadId = init_roadId;
    laneId = init_laneId;
    s = init_s;

    road = system->getRoad(roadId);

    LS = road->getLaneSection(s);
    Vector2D laneCenter = LS->getLaneCenter(laneId, s);

    t = laneCenter[0];
    Transform vtrans = road->getRoadTransform(s, laneCenter[0]);
    xyz = osg::Vec3(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());

    roadLength = road->getLength();
}

void ReferencePosition::moveForward(float dt,float speed)
{
    float step = dt*speed;
    s = step+s;

    LaneSection* newLS = road->getLaneSection(s);

    if (newLS != LS)
    {
        laneId =  road->searchLane(s,t);
        LS = newLS;
    }

    Transform vtrans = road->getRoadTransform(s,t);
    xyz = osg::Vec3(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());

}

osg::Vec3 ReferencePosition::getPosition()
{
    return xyz;
}
