#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscTrajectory.h>
#include <Entity.h>


class Trajectory : public OpenScenario::oscTrajectory
{

private:
	

public:
    std::vector<osg::Vec3> polylineVertices;
    std::vector<bool> isRelVertice;
    Trajectory();
	~Trajectory();
	virtual void finishedParsing();
    void initialize(int verticesCounter);
    osg::Vec3 getAbsolute(Entity* currentEntity);
    float getReference(int visitedVertices);
    double t0;
    double t1;
    double dt;
    int verticesCounter;

};

#endif // TRAJECTORY_H
