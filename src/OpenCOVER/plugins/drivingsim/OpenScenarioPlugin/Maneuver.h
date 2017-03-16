#ifndef MANEUVER_H
#define MANEUVER_H

using namespace std;
#include<iostream>
#include<string>
#include <vector>
#include <list>
#include <algorithm>
#include <osg/Vec3>


class Maneuver {

 private:
	string name;
	int numberOfExecutions;
	int currentExecution;


 public:
	float totalDistance;
	osg::Vec3 norm_direction_vec;
	//float step_distance;
	int visitedVertices;
	int verticesCounter;
	osg::Vec3 newPosition;
	bool maneuverCondition;
	bool arriveAtVertex;
	//vector<float> targetEntityPosition;
	vector<vector<float>> polylineVertices;
    Maneuver(string name);
	~Maneuver();
    osg::Vec3 &followTrajectory(osg::Vec3 currentPos, vector<float> targetPosition, float speed);
	string getName();
	bool getManeuverCondition();
	void setManeuverCondition();
	//void setTargetEntityPosition(vector<float> position);
	void setPolylineVertices(float x, float y, float z);
	//vector<float> followTrajectory(float speed, vector<float> currentPosition);


};

#endif // MANEUVER_H
