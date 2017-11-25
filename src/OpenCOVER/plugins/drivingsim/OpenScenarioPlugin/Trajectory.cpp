#include "Trajectory.h"

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
	name = oscTrajectory::name.getValue();
}
void Trajectory::initialize(vector<osg::Vec3> vec_temp)
{
	polylineVertices = vec_temp;
}