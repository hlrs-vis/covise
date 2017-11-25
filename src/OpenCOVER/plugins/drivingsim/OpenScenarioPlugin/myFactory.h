#ifndef MY_FACTORY_H
#define MY_FACTORY_H

#include <OpenScenario/oscFactory.h>
#include <OpenScenario/oscObjectBase.h>
#include "FollowTrajectory.h"
#include <string>

using namespace std;

class myFactory : public OpenScenario::oscFactory<OpenScenario::oscObjectBase, std::string>
{
public:
	myFactory();
	~myFactory();
	virtual OpenScenario::oscObjectBase *create(const std::string &name);
};

#endif // MY_FACTORY_H
