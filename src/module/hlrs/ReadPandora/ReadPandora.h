#ifndef READHDF5_H
#define READHDF5_H

#include <api/coModule.h>
#include <api/coFileBrowserParam.h>

class ReadHDF5 : public covise::coModule {

private:
    covise::coFileBrowserParam *filename;
    covise::coOutputPort *uOut, *pointsOut;

    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

public:
    ReadHDF5(int argc, char *argv[]);
    covise::coOutputPort *grid;
};
#endif
