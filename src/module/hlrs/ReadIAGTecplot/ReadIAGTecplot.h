#ifndef READIAGTECPLOT_H
#define READIAGTECPLOT_H

#include "reader/coReader.h"
#include "reader/ReaderControl.h"
#include <api/coModule.h>
#include <api/coFileBrowserParam.h>
#include <api/coIntScalarParam.h>

class ReadIAGTecplot : public covise::coReader {

private:
    
    int width;
    int height;
    int numSteps;
    int numTasks;

	// lists for Choice Labels
	vector<string> vectChoices;
	vector<string> scalChoices;

    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);
	std::string fileName;

    covise::coIntScalarParam *p_firstStep, *p_lastStep;

public:
	ReadIAGTecplot(int argc, char *argv[]);
	virtual void param(const char *paramName, bool inMapLoading);
};
#endif
