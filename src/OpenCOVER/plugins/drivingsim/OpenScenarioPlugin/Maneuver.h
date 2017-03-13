#ifndef MANEUVER_H
#define MANEUVER_H

using namespace std;
#include<iostream>
#include<string>
#include <vector>
#include <algorithm>


class Maneuver {

 private:
	string name;
	int numberOfExecutions;
	int currentExecution;


 public:
	float totalDistance;
	bool maneuverCondition;
	vector<float> targetEntityPosition;
    Maneuver(string name);
	~Maneuver();
    vector<float> calculateNewEntityPosition(vector<float> currentPosition, float speed);
	string getName();
	bool getManeuverCondition();
	void setManeuverCondition();
	void setTargetEntityPosition(vector<float> position);


};

#endif // MANEUVER_H
