#ifndef ACT_H
#define ACT_H

#include "Entity.h"
#include "Maneuver.h"
#include "StoryElement.h"
using namespace std;
#include<iostream>
#include<string>
#include <list>
#include <OpenScenario/schema/oscAct.h>
class Sequence;
class Condition;

class Act : public OpenScenario::oscAct, public StoryElement
{

 public:
	string name;
	list<Entity*> activeEntityList;
    list<::Sequence*> sequenceList;

	Act();
	~Act();
	virtual void finishedParsing();
    void initialize(::Sequence *sequence_temp);
	string getName();

    std::list<Condition*> startConditionList;
    std::list<Condition*> endConditionList;
    void addEndCondition(Condition* condition);
    void addStartCondition(Condition* condition);

};

#endif // ACT_H
