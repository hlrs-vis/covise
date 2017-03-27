#ifndef MY_FACTORY_H
#define MY_FACTORY_H

#include <DrivingSim/OpenScenario/oscFactory.h>
#include <DrivingSim/OpenScenario/oscObjectBase.h>
#include <FollowTrajectory.h>
#include <string.h>

using namespace std;

class myFactory : public OpenScenario::oscFactory<OpenScenario::oscObjectBase, std::string>
{
public:
	myFactory();
	~myFactory();
	virtual OpenScenario::oscObjectBase *create(const std::string &name);
};

#endif // MY_FACTORY_H