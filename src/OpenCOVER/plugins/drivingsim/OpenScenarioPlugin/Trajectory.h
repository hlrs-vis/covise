#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscTrajectory.h>

class Trajectory : public OpenScenario::oscTrajectory
{

private:
	

public:
    std::vector<osg::Vec3> polylineVertices;
    std::vector<bool> isRelVertice;
    Trajectory();
	~Trajectory();
	virtual void finishedParsing();
    void initialize(std::vector<osg::Vec3> vec_temp, std::vector<bool> isRelVertice_temp);
};

#endif // TRAJECTORY_H
