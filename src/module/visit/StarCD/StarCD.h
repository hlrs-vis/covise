/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STARCD_H
#define STARCD_H

// 30.09.99

#include <api/coSimLib.h>
using namespace covise;

class StarMesh;
/**
 * Coupling of Star-CD and Covise - both parallel and sequential
 *
 */

class StarCD : public coSimLib
{
public:
    // set port numbers: never re-use Nework files from different settings
    // Maximum number of Regions
    enum
    {
        MAX_REGIONS = 16
    };

    // Maximum number of SCALARS
    enum
    {
        MAX_SCALARS = 10
    };

    // Number of user data fields
    enum
    {
        MAX_UDATA = 10
    };

    // Number of data output ports
    enum
    {
        NUM_OUT_DATA = 6
    };

    typedef int32_t int32;

private:
    // SHUTDOWN means: sim is running but should shutdown ASAP
    enum SimStatus
    {
        NOT_STARTED, // Currently no simulation connected
        WAIT_BOCO, // Simulation waits for new BC
        RUNNING, // Simulation runs, will send EXEC when data readi
        SHUTDOWN, // = RUNNING, but shut down Sim after next data send
        RESTART // = RUNNING, but shut down+restart Sim after next data send
    } d_simState;

    // fortran flag block: transmitted to simulation once at start-up
    struct
    {
        int32 iconum; // number of regions
        int32 icoreg[MAX_REGIONS]; // region number:  0 if not set
        int32 icovel[MAX_REGIONS]; // = 0 if velocity     not set, =1  if set
        int32 icot[MAX_REGIONS]; // = 0 if temperature  not set, =1  if set
        int32 icoden[MAX_REGIONS]; // = 0 if density      not set, =1  if set
        int32 icotur[MAX_REGIONS]; // = 0 if turbulence   not set, =1  k/eps, =-1 Int/len
        int32 icop[MAX_REGIONS]; // = 0 if pressure     not set, =1  if set
        int32 icosca[MAX_REGIONS][MAX_SCALARS]; // 0 for none  # for used scalar
        int32 icousr[MAX_REGIONS][MAX_UDATA]; // = 0 if USRx not used, =1  if used
    } flags;

    struct
    {
        int32 icostp; // how many steps to go now
        int32 icoout[NUM_OUT_DATA]; // which field to send
        float covvel[MAX_REGIONS][3]; // u,v,w
        float covt[MAX_REGIONS];
        float covden[MAX_REGIONS];
        float covtu1[MAX_REGIONS]; // k    or  Int
        float covtu2[MAX_REGIONS]; // eps  or  Len
        float covp[MAX_REGIONS];
        float covsca[MAX_REGIONS][MAX_SCALARS];
        float covusr[MAX_REGIONS][MAX_UDATA];
    } para;

    // the size inside the common block which we should transfer
    static const int ISETUP, IPARAM;

    // the common parameters
    coFileBrowserParam *p_setup;
    coChoiceParam *p_region;
    coIntScalarParam *p_steps;
    coBooleanParam *p_freeRun;
    coBooleanParam *p_runSim;
    coBooleanParam *p_quitSim;
    coIntScalarParam *p_numProc;

    // labels of p_region
    char *choices[MAX_REGIONS];
    int numRegions;

    // The parameters for every inlet
    coFloatVectorParam *p_v[MAX_REGIONS];
    coFloatSliderParam *p_vmag[MAX_REGIONS];
    coFloatSliderParam *p_t[MAX_REGIONS];
    coFloatSliderParam *p_den[MAX_REGIONS];
    coFloatSliderParam *p_p[MAX_REGIONS];

    // only one set of p_k,p_eps or p_tin,p_len is used, depending on p_tur
    coFloatSliderParam *p_k[MAX_REGIONS];
    coFloatSliderParam *p_eps[MAX_REGIONS];
    coFloatSliderParam *p_tin[MAX_REGIONS];
    coFloatSliderParam *p_len[MAX_REGIONS];
    coFloatVectorParam *p_euler[MAX_REGIONS];

    //coChoiceParam      *p_scalSw[MAX_REGIONS];
    coFloatSliderParam *p_scal[MAX_REGIONS][MAX_SCALARS];
    //coChoiceParam      *p_userSw[MAX_REGIONS];
    coFloatSliderParam *p_user[MAX_REGIONS][MAX_SCALARS];
    coChoiceParam *p_value[NUM_OUT_DATA];

    // Input port: can receive Case info from here (e.g. in VISiT pipeline)
    coInputPort *p_configObj;
    coInputPort *p_commObj;

    // output ports
    coOutputPort *p_mesh, *p_data[NUM_OUT_DATA], *p_residual, *p_feedback;

    // currently used setup file name and setup object name
    char *d_setupFileName;
    char *d_setupObjName;
    char *d_commObjName;

    // StarCD run set-up
    char *d_user; // user name for simulation run
    char *d_host; // host name for simulation run
    char *d_compDir; // directory for simulation run
    char *d_case; // case name for simulation run
    char *d_script; // script to create geometry
    char *d_commArg; // if a COMM object arrives, it disposes its arg here
    char *d_meshDir; // Directory for .mdl file in local space
    char *d_creator; // program that created our grid, give to SCRIPT
    char *d_usr[10]; // user parameters 0..9

    // our Mesh handling class
    StarMesh *d_mesh;

    // Flag: 0 if configuration used first time, 1 otherwise
    int d_useOldConfig;

    // ------- internally used functions

    // called by c'tor
    void createParam();
    void createOutPorts();
    void createRegionParam();

    // called by compute()

    int mustReconfigure(); // check whether we must re-config becauise of
    // new input objects

    bool mustIgnoreExec(); // check whether we must NOT execute because  we
    // were stared with NULL object at connected port

    // allocate buffer and write in region settings
    char *activeRegionSettings();

    int doSetup(const coDistributedObject *setupObj,
                const coDistributedObject *commObj,
                int useModified);
    int setupCase(char *&filebuffer, const char *&line);
    int setupRegions(char *&filebuffer, const char *&line);
    void addParaAttrib();
    int startupSimulation(const coDistributedObject *setupObj,
                          const coDistributedObject *commObj,
                          const char *settingsString);
    int startupParallel(int numNodes);
    int sendPortValues();
    void addFeedbackInfo(coDistributedObject **feedPatches);
    void createFeedbackObjects();

    // called by param() and compute()
    int stopSimulation();

    static coModule *module;

public:
    /// Default constructor
    StarCD(int argc, char *argv[]);

    /// Destructor : virtual in case we derive objects
    virtual ~StarCD();

    // the compute callback
    virtual int compute(const char *);

    // the parameter callback
    virtual void param(const char *, bool inMapLoading);

    // the postInst callback
    virtual void postInst();

    // called after COMM_QUIT command from simulation
    virtual int endIteration();

    static coModule *getModule()
    {
        return module;
    }
};
#endif
