#ifndef SOURCE_H
#define SOURCE_H


using namespace std;
#include<iostream>
#include<string>
#include <list>
#include <OpenScenario/schema/oscSource.h>

class Source : public OpenScenario::oscSource
{

 private:
	

 public:
	 Source();
	~Source();

	void finishedParsing();

};

#endif // SOURCE_H
