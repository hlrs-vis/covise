#ifndef ACT_H
#define ACT_H

#include "Entity.h"
#include "Maneuver.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>
#include <OpenScenario/schema/oscAct.h>
class Sequence;
class Condition;

class Act : public OpenScenario::oscAct
{

 public:
	string name;
	list<Entity*> activeEntityList;
	list<Maneuver*> maneuverList;
    list<::Sequence*> sequenceList;
	int numberOfExecutions;

	//conditions
	bool actCondition;
	bool actFinished;
	string startConditionType;
	float startTime;
	string endConditionType;
	float endTime;

	Act();
	~Act();
	virtual void finishedParsing();
    void initialize(::Sequence *sequence_temp);
	string getName();
	int getNumberOfExecutions();

    list<Condition*> startConditionList;
    list<Condition*> endConditionList;

};

#endif // ACT_H
