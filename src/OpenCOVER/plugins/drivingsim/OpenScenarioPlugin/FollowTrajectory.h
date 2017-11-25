#ifndef FOLLOW_TRAJECTORY_H
#define FOLLOW_TRAJECTORY_H

using namespace std;
#include<iostream>
#include<string>
#include <OpenScenario/schema/oscFollowTrajectory.h>

class FollowTrajectory : public OpenScenario::oscFollowTrajectory
{

private:
string name;

public:
	FollowTrajectory();
	~FollowTrajectory();
	virtual void finishedParsing();
};

#endif // FOLLOW_TRAJECTORY_H
