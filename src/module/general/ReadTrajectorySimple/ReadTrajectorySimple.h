
#ifndef _READ_OBJ_SIMPLE_H
#define _READ_OBJ_SIMPLE_H

#include <api/coModule.h>
#include <vector>
using namespace covise;

//The base class for all module programming is coModule. 
//An application is created by deriving a class of coModule. The constructor of the derived class creates the module layout with input ports, output ports and parameters. 
//Virtual functions are overloaded to implement the module's reactions on events.
class ReadTrajectorySimple : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();

    bool openFile();
    void readFile();

    //  member data
    const char *filename; // obj file name
    FILE *fp;

    coOutputPort *linePort; 
    coFileBrowserParam *objFileParam;

public:
    ReadTrajectorySimple(int argc, char *argv[]);
    virtual ~ReadTrajectorySimple();
};

#endif
