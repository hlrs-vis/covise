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

    LaneSection* LS = road->getLaneSection(s);
    Vector2D laneCenter = LS->getLaneCenter(laneId, s);

    t = laneCenter[0];
    Transform vtrans = road->getRoadTransform(s, laneCenter[0]);
    xyz = osg::Vec3(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());
}
