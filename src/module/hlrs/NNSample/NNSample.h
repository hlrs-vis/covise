#ifndef NNSAMPLE_H
#define NNSAMPLE_H


/****************************************************
 *
 *      Nearest Neighbor Sample
 *
 *
 ****************************************************/

#include <api/coSimpleModule.h>
#include <vector>


using namespace covise;


class NNSample: public coSimpleModule
{
private:
    virtual int compute(const char *port);
   // virtual void quit();
    virtual void param(const char *name, bool inMapLoading);

    int nearestNeighborIDX(float x, float y, float z);

    coInputPort *Points_In_Port, *Data_In_Port, *Reference_Grid_In_Port, *Unigrid_In_Port;
    coOutputPort *Grid_Out_Port, *Data_Out_Port;

    coDoPoints *pointList;
    float *p_x, *p_y, *p_z;
    int numPoints;

public:
    NNSample(int argc, char *argv[]);
    ~NNSample();

};

#endif
