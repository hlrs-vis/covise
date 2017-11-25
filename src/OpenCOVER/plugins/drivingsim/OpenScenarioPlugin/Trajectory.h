#ifndef TRAJECTORY_H
#define TRAJECTORY_H

using namespace std;
#include<iostream>
#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscTrajectory.h>

class Trajectory : public OpenScenario::oscTrajectory
{

private:
	

public:
	string name;
	vector<osg::Vec3> polylineVertices;
	Trajectory();
	~Trajectory();
	virtual void finishedParsing();
	void initialize(vector<osg::Vec3> vec_temp);
};

#endif // TRAJECTORY_H
