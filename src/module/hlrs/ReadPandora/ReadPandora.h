#ifndef READPANDORA_H
#define READPANDORA_H

#include <api/coModule.h>
#include <api/coFileBrowserParam.h>
#include <hdf5.h>
#include <hdf5_hl.h>

class ReadPandora : public covise::coModule {

private:
    covise::coFileBrowserParam *filename;
    covise::coOutputPort *dataOut, *meshOut;
    
    int width;
    int height;
    int numSteps;
    int numTasks;

    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

public:
    ReadPandora(int argc, char *argv[]);
    covise::coOutputPort *grid;
};
#endif
