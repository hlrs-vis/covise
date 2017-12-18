#include "Trajectory.h"

using namespace std;

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
}
void Trajectory::initialize(vector<osg::Vec3> vec_temp)
{
	polylineVertices = vec_temp;
}
