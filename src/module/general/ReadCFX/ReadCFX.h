/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_CFX_H
#define _READ_CFX_H
/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Read Ansys CFX                                            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 *\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/coRestraint.h>
#include <do/coDoPoints.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include "../Transform/Matrix.h"

#define GRID 1
#define VECTOR 2
#define SCALAR 3
#define REGION 4
#define REGSCAL 5
#define BOUNDARY 6
#define PARTICLES 7

#define WALL 1
#define INLET 2
#define OUTLET 3
#define CFX_INTERFACE 4
#define SYMMETRY 5
#define OPENING 6

class CFX : public coSimpleModule
{
    COMODULE
private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *, bool);

    coDoSet *normalizeSet(int *numTimesteps, coDistributedObject **list, const coObjInfo &objInfo);

    coRestraint selRegion;
    coRestraint selBoundary;
    coOutputPort *p_outPort1; // mesh
    coOutputPort *p_outPort2; // scalar data
    coOutputPort *p_outPort3; // vector data
    coOutputPort *p_outPort4; // region mesh
    coOutputPort *p_outPort5; // region scalar data
    coOutputPort *p_outPort6; // boundary
    coOutputPort *p_outPort7; // boundary scalar data
    coOutputPort *p_outPort8; // boundary vector data
    coOutputPort *p_outPort9; // particle
    coOutputPort *p_outPort10; // particle scalar data
    coOutputPort *p_outPort11; // particle vector data

    coFileBrowserParam *p_result;

    coChoiceParam *p_zone;

    coChoiceParam *p_scalar;
    coChoiceParam *p_vector;

    coFloatParam *p_scalarFactor;
    coFloatParam *p_scalarDelta;

    coChoiceParam *p_boundScalar;
    coChoiceParam *p_boundVector;

    coChoiceParam *p_particleType;
    coChoiceParam *p_particleScalar;
    coChoiceParam *p_particleVector;

    coStringParam *p_region;
    coStringParam *p_boundary;

    coBooleanParam *p_readGrid;
    coBooleanParam *p_readRegions;
    coBooleanParam *p_readBoundaries;

    coIntScalarParam *p_maxTimeSteps;
    coIntScalarParam *p_skipParticles;
    coIntScalarParam *p_skipTimeSteps;

    coIntScalarParam *p_timesteps;
    coIntScalarParam *p_firststep;

    coBooleanParam *p_readGridForAllTimeSteps;
    coIntScalarParam *p_timeDependentZone;
    coFloatParam *p_initialRotation;
    coFloatVectorParam *p_rotAxis;
    coFloatVectorParam *p_rotVertex;

    coBooleanParam *p_relabs;
    coIntScalarParam *p_relabs_zone;
    coFloatParam *p_rotAngle;
    coFloatParam *p_rotSpeed;

    coBooleanParam *p_rotVectors;

    int t;

    char **VectChoiceVal, **ScalChoiceVal, **RegionChoiceVal, **ZoneChoiceVal;
    char **pVectChoiceVal, **pScalChoiceVal;
    int *VectIndex, *ScalIndex;
    const char *resultfileName;
    char **ParticleTypeNames;
    int nscalars, nvectors, nregions, nzones, nparticleTracks, nparticleTypes;
    int npscalars, npvectors, numTracks;
    bool *boundary_only;

    int *boundary_type_list; // type of boundary (INLET, OUTLET, WALL, CFX_INTERFACE or SYMMETRY)
    int *boundary_zone_list; // zone that contains boundary
    int nboundaries_total;

    int counts[cfxCNT_SIZE];
    int currenttimestep;

    void setParticleVarChoice(int type);

    int readTimeStep(char *Name, char *Name2, char *Name3, int timestep, int var, int zone, float *omega, float stepduration, coDistributedObject **timeStepToRead, coDistributedObject **timeStepScalToRead);
    int readZone(char *Name, char *Name2, char *Name3, int timesteps, int var, int zone, float *omega, float stepduration, coDistributedObject **zoneToRead, coDistributedObject **zoneStepScalToRead, coDistributedObject **zoneVectToRead);

    int readGrid(const char *gridName, int nelems, int nconn, int nnodes, int step, int zone, float omega, float stepduration, coDoUnstructuredGrid **gridObj);
    int readScalar(const char *scalarName, int n_values, int scal, coDoFloat **scalarObj);
    int readVector(const char *vectorName, int n_values, int vect, int zone, float *omega, coDoVec3 **vectorObj);
    int readRegion(const char *reg_gridName, const char *reg_vectorName, const char *reg_scalarName, int regnum, int scal, coDoPolygons **gridObj, coDoFloat **scalarObj);
    int readBoundary(const char *bound_gridName, const char *bound_scalarName, const char *bound_vectorName, int boundnrglob, int zone, int scal, int vect, coDoPolygons **gridObj, coDoFloat **scalObj, coDoVec3 **vectObj);
    int readParticles(const char *particle_gridName, const char *particle_scalarName, const char *particle_vectorName, int scal, int vect, coDoPoints **gridObj, coDoFloat **scalObj, coDoVec3 **vectObj);

    void get_choice_fields(int zone);

    string _dir;
    string _theFile;
    bool inMapLoading;

    // hack: set this flag if param called from compute
    bool computeRunning;

    // 'user level' decides how many variables are shown: 0=all 1=default, 2=more
    int usrLevel_;

    enum AxisChoice
    {
        X = 0x00,
        Y = 0x01,
        Z = 0x02
    };
    coChoiceParam *p_relabsAxis;
    char *s_rotaxis[3];

    enum DirectionChoice
    {
        None = 0x00,
        Abs2Rel = 0x01,
        Rel2Abs = 0x02
    };
    coChoiceParam *p_relabsDirection;
    char *s_direction[3];
    float **xcoords;
    float **ycoords;
    float **zcoords;
    int *psVarIndex;
    int *pvVarIndex;

    float scalarDelta;
    float scalarFactor;

public:
    CFX(int argc, char **argv);
    virtual ~CFX();
};
#endif
