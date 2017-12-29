#include "Trajectory.h"

using namespace std;

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
	name = oscTrajectory::name.getValue();
}
void Trajectory::initialize(vector<osg::Vec3> vec_temp, std::string mode_temp)
{
	polylineVertices = vec_temp;
    mode = mode_temp;
}
