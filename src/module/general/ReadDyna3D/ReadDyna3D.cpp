/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Dyna3D data         	                  **
 **              (Attention: Thick shell elements omitted !)               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
 ** Revision R. Beller 08.97 & 02.99                                       **
\**************************************************************************/

//#include <appl/ApplInterface.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include "ReadDyna3D.h"
#include <util/coviseCompat.h>

#define tauio_1 tauio_

//-------------------------------------------------------------------------
static const char *colornames[10] = {
    "yellow", "red", "blue", "green",
    "violet", "chocolat", "linen", "white",
    "crimson", "LightGoldenrod"
};
/*
// GLOBAL VARIABLES
char InfoBuf[1000];

int NumIDs;
int nodalDataType;
int elementDataType;
int component;
int ExistingStates = 0;
int MinState = 1;
int MaxState = 1;
int State    = 1;

int IDLISTE[MAXID];
int *numcoo, *numcon, *numelem;

int *maxSolid  = NULL;
int *minSolid  = NULL;
int *maxTShell = NULL;
int *minTShell = NULL;
int *maxShell  = NULL;
int *minShell  = NULL;
int *maxBeam   = NULL;
int *minBeam   = NULL;

int *delElem = NULL;
int *delCon  = NULL;

Element** solidTab  = NULL;
Element** tshellTab = NULL;
Element** shellTab  = NULL;
Element** beamTab   = NULL;

// storing coordinate positions for all time steps
int **coordLookup = NULL;

// element and connection list for time steps (considering the deletion table)
int **My_elemList = NULL;
int **conList  = NULL;

int *DelTab = NULL;  // node/element deletion table

coDoUnstructuredGrid*	  grid_out    = NULL;
coDoVec3* Vertex_out  = NULL;
coDoFloat* Scalar_out = NULL;

coDoUnstructuredGrid**  grids_out        = NULL;
coDoVec3** Vertexs_out  = NULL;
coDoFloat** Scalars_out = NULL;

coDoSet* grid_set_out,    *grid_timesteps;
coDoSet*	Vertex_set_out,  *Vertex_timesteps;
coDoSet*	Scalar_set_out, *Scalar_timesteps;

coDoSet*	grid_sets_out[MAXTIMESTEPS];
coDoSet*	Vertex_sets_out[MAXTIMESTEPS];
coDoSet*	Scalar_sets_out[MAXTIMESTEPS];
*/

/* kern.f -- translated by f2c (version 19950110).*/

/*
struct {
  int itrecin, nrin, nrzin, ifilin, itrecout, nrout, nrzout, ifilout, irl, iwpr;
  float tau[512];
} tauio_;

int infile;
int numcoord;
int *NodeIds=NULL;
int *SolidNodes=NULL;
int *SolidMatIds=NULL;
int *BeamNodes=NULL;
int *BeamMatIds=NULL;
int *ShellNodes=NULL;
int *ShellMatIds=NULL;
int *TShellNodes=NULL;
int *TShellMatIds=NULL;
int *SolidElemNumbers=NULL;
int *BeamElemNumbers=NULL;
int *ShellElemNumbers=NULL;
int *TShellElemNumbers=NULL;
int *Materials=NULL;

// Common Block Declarations
// global scope !!! attention to side-effects !!!
float *Coord      = NULL;           // Coordinates
float *DisCo      = NULL;           // displaced coordinates
float *NodeData   = NULL;
float *SolidData    = NULL;
float *TShellData = NULL;
float *ShellData  = NULL;
float *BeamData   = NULL;

char CTauin[300];
char CTrace, CEndin, COpenin;
float TimestepTime    = 0.0;
int NumSolidElements  = 0;
int NumBeamElements   = 0;
int NumShellElements  = 0;
int NumTShellElements = 0;
int NumMaterials=0;
int NumDim=0;
int NumGlobVar=0;
int NumWords=0;
int ITFlag=0;
int IUFlag=0;
int IVFlag=0;
int IAFlag=0;
int MDLOpt=0;
int IStrn=0;
int NumSolidMat=0;
int NumSolidVar=0;
int NumAddSolidVar=0;
int NumBeamMat=0;
int NumBeamVar=0;
int NumAddBeamVar=0;
int NumShellMat=0;
int NumShellVar=0;
int NumAddShellVar=0;
int NumTShellMat=0;
int NumTShellVar=0;
int NumShellInterpol=0;
int NumInterpol=0;
int NodeAndElementNumbering=0;
int SomeFlags[5];
int NumBRS=0;
int NodeSort=0;
int NumberOfWords=0;
int NumNodalPoints=0;
int IZControll=0;
int IZElements=0;
int IZArb=0;
int IZStat0=0;

// GLOBAL STATIC VARIABLES
// Table of "constant" values
static int c__1 = 1;
static int c__0 = 0;
static int c__13 = 13;
static int c__8 = 8;
static int c__5 = 5;
static int c__4 = 4;
static int c__10 = 10;
static int c__16 = 16;
*/

//
//
//..........................................................................
//

// Constructor
ReadDyna3D::ReadDyna3D(int argc, char *argv[])
    : coModule(argc, argv, "Read LS-DYNA3D ptf files")
{

    const char *nodalDataType_label[4] = {
        "None", "Displacements", "Velocities",
        "Accelerations"
    };
    const char *elementDataType_label[4] = { "None", "StressTensor", "PlasticStrain", "Thickness" };
    const char *component_label[8] = {
        "Sx", "Sy", "Sz", "Txy", "Tyz", "Txz",
        "Pressure", "Von Mises"
    };
    const char *format_label[2] = { "cadfem", "original" };

    const char *byteswap_label[3] = { "off", "on", "auto" };

    // Output ports
    p_grid = addOutputPort("out_grid", "UnstructuredGrid", "grid");
    p_data_1 = addOutputPort("data_1", "Vec3", "output nodal vector data");
    p_data_2 = addOutputPort("data_2", "Float", "output nodal scalar data");

    // Parameters
    p_data_path = addFileBrowserParam("data_path", "Geometry file path");
    p_nodalDataType = addChoiceParam("nodalDataType", "Nodal results data to be read");
    p_elementDataType = addChoiceParam("elementDataType", "Element results data to be read");
    p_component = addChoiceParam("component", "stress tensor component");
    p_Selection = addStringParam("Selection", "Number selection for parts");
    // p_State = addIntSliderParam("State","Timestep");
    p_Step = addInt32VectorParam("Min_Num_State", "Time step");
    p_format = addChoiceParam("format", "Format of LS-DYNA3D ptf-File");

    p_byteswap = addChoiceParam("byteswap", "Perform Byteswapping");
    p_byteswap->setValue(3, byteswap_label, 2);
    byteswapFlag = BYTESWAP_OFF;

    p_only_geometry = addBooleanParam("OnlyGeometry", "Reading only Geometry? yes|no");

    // Default setting
    // p_data_path->setValue("/common/kaiper/d3plot","*");
    p_data_path->setValue("/data/general/examples/Dyna3d/aplot", "*");
    p_nodalDataType->setValue(4, nodalDataType_label, 0);
    p_elementDataType->setValue(4, elementDataType_label, 0);
    p_component->setValue(8, component_label, 7);
    p_Selection->setValue("0-9999999");
    // p_State->setValue(1,1,1);
    // p_Step->setValue(1,1,1);
    long init_Step[2] = { 1, 0 };
    p_Step->setValue(2, init_Step);
    p_format->setValue(2, format_label, 1);
    p_only_geometry->setValue(0);
}

void ReadDyna3D::postInst()
{

    p_nodalDataType->show();
    p_elementDataType->show();
    p_Step->show();

    ExistingStates = 0;
    MinState = 1;
    MaxState = 1;
    NumState = 0;
    State = 1;

    maxSolid = 0;
    minSolid = 0;
    maxTShell = 0;
    minTShell = 0;
    maxShell = 0;
    minShell = 0;
    maxBeam = 0;
    minBeam = 0;

    delElem = delCon = 0;
    solidTab = tshellTab = shellTab = beamTab = 0;

    coordLookup = 0;

    My_elemList = conList = 0;
    DelTab = 0;

    grid_out = 0;
    Vertex_out = 0;
    Scalar_out = 0;

    grids_out = 0;
    Vertexs_out = 0;
    Scalars_out = 0;

    NodeIds = SolidNodes = SolidMatIds = BeamNodes = BeamMatIds = 0;
    ShellNodes = ShellMatIds = TShellNodes = TShellMatIds = 0;
    SolidElemNumbers = BeamElemNumbers = ShellElemNumbers = 0;
    TShellElemNumbers = Materials = 0;

    Coord = DisCo = NodeData = SolidData = TShellData = ShellData = 0;
    BeamData = 0;

    TimestepTime = 0.0;
    version = 0.0;
    NumSolidElements = NumBeamElements = NumShellElements = 0;
    NumTShellElements = NumMaterials = NumDim = NumGlobVar = NumWords = 0;
    ITFlag = IUFlag = IVFlag = IAFlag = IDTDTFlag = NumTemperature = MDLOpt = IStrn = 0;
    NumSolidMat = 0;
    NumSolidVar = 0;
    NumAddSolidVar = 0;
    NumBeamMat = 0;
    NumBeamVar = 0;
    NumAddBeamVar = 0;
    NumShellMat = 0;
    NumShellVar = 0;
    NumAddShellVar = 0;
    NumTShellMat = 0;
    NumTShellVar = 0;
    NumShellInterpol = 0;
    NumInterpol = 0;
    NodeAndElementNumbering = 0;
    NumBRS = 0;
    NodeSort = 0;
    NumberOfWords = 0;
    NumNodalPoints = 0;
    IZControll = 0;
    IZElements = 0;
    IZArb = 0;
    IZStat0 = 0;
    NumFluidMat = 0;
    NCFDV1 = 0;
    NCFDV2 = 0;
    NADAPT = 0;

    c__1 = 1;
    c__0 = 0;
    c__13 = 13;
    c__8 = 8;
    c__5 = 5;
    c__4 = 4;
    c__10 = 10;
    c__16 = 16;
    numcon = NULL;
    numelem = NULL;
    numcoo = NULL;
}

int ReadDyna3D::compute(const char *)
{
    //
    // ...... do work here ........
    //

    numcon = NULL;
    numelem = NULL;
    numcoo = NULL;

    ExistingStates = 0;
    MinState = 1;
    MaxState = 1;
    NumState = 0;
    State = 1;

    maxSolid = 0;
    minSolid = 0;
    maxTShell = 0;
    minTShell = 0;
    maxShell = 0;
    minShell = 0;
    maxBeam = 0;
    minBeam = 0;

    delElem = delCon = 0;
    solidTab = tshellTab = shellTab = beamTab = 0;

    coordLookup = 0;

    My_elemList = conList = 0;
    DelTab = 0;

    grid_out = 0;
    Vertex_out = 0;
    Scalar_out = 0;

    grids_out = 0;
    Vertexs_out = 0;
    Scalars_out = 0;

    NodeIds = SolidNodes = SolidMatIds = BeamNodes = BeamMatIds = 0;
    ShellNodes = ShellMatIds = TShellNodes = TShellMatIds = 0;
    SolidElemNumbers = BeamElemNumbers = ShellElemNumbers = 0;
    TShellElemNumbers = Materials = 0;

    Coord = DisCo = NodeData = SolidData = TShellData = ShellData = 0;
    BeamData = 0;

    TimestepTime = 0.0;
    NumSolidElements = NumBeamElements = NumShellElements = 0;
    NumTShellElements = NumMaterials = NumDim = NumGlobVar = NumWords = 0;
    ITFlag = IUFlag = IVFlag = IAFlag = IDTDTFlag = NumTemperature = MDLOpt = IStrn = 0;
    NumSolidMat = 0;
    NumSolidVar = 0;
    NumAddSolidVar = 0;
    NumBeamMat = 0;
    NumBeamVar = 0;
    NumAddBeamVar = 0;
    NumShellMat = 0;
    NumShellVar = 0;
    NumAddShellVar = 0;
    NumTShellMat = 0;
    NumTShellVar = 0;
    NumShellInterpol = 0;
    NumInterpol = 0;
    NodeAndElementNumbering = 0;
    NumBRS = 0;
    NodeSort = 0;
    NumberOfWords = 0;
    NumNodalPoints = 0;
    IZControll = 0;
    IZElements = 0;
    IZArb = 0;
    IZStat0 = 0;
    NumFluidMat = 0;
    NCFDV1 = 0;
    NCFDV2 = 0;
    NADAPT = 0;

    c__1 = 1;
    c__0 = 0;
    c__13 = 13;
    c__8 = 8;
    c__5 = 5;
    c__4 = 4;
    c__10 = 10;
    c__16 = 16;

    int i;
    coRestraint sel;

    //=======================================================================
    // get input parameters and data object name

    // Get file path
    // Covise::get_browser_param("data_path", &data_Path);
    data_Path = p_data_path->getValue();
    // Get nodal output data type
    // Covise::get_choice_param("nodalDataType",&nodalDataType);
    nodalDataType = p_nodalDataType->getValue();
    // Get element output data type
    // Covise::get_choice_param("elementDataType",&elementDataType);
    elementDataType = p_elementDataType->getValue();
    // Get component of stress tensor
    // Covise::get_choice_param("component",&component);
    component = p_component->getValue();

    byteswapFlag = p_byteswap->getValue();
    if (byteswapFlag == BYTESWAP_OFF)
    {
        cerr << "ReadDyna3D::readDyna3D info: byteswap off" << endl;
    }
    else if (byteswapFlag == BYTESWAP_ON)
    {
        cerr << "ReadDyna3D::readDyna3D info: byteswap on" << endl;
    }
    else if (byteswapFlag == BYTESWAP_AUTO)
    {
        cerr << "ReadDyna3D::readDyna3D info: byteswap auto" << endl;
    }
    else
    {
        cerr << "ReadDyna3D::readDyna3D info: byteswap unknown error" << endl;
    }

    // Get the selection
    const char *selection;
    // Covise::get_string_param("Selection",&selection);
    selection = p_Selection->getValue();
    sel.add(selection);

    // Initilize list of part IDs
    for (i = 0; i < MAXID; i++)
        IDLISTE[i] = 0;

    // Find selected parts per IDs
    int numParts = 0;
    for (i = 0; i < MAXID; i++)
    {
        if ((IDLISTE[i] = sel(i + 1)) > 0) // Notice: offset for ID (= material number - 1)
        {
            numParts++;
        }
    }
    if (numParts == 0)
    {
        sendError("No parts with selected IDs");
        return STOP_PIPELINE;
    }

    // Get timesteps
    // Covise::get_slider_param("State", &MinState, &MaxState, &State);
    //p_Step->getValue(MinState,MaxState,State);
    MinState = p_Step->getValue(0);
    if (MinState <= 0)
    {
        sendWarning("Minimum time step must be positive!!");
        p_Step->setValue(0, 1);
        MinState = 1;
    }
    NumState = p_Step->getValue(1);
    MaxState = MinState + NumState;
    if (MinState > MaxState)
    {
        int temp_swap;
        temp_swap = MaxState;
        MaxState = MinState;
        MinState = temp_swap;
    }
    NumState = MaxState - MinState;
    State = MaxState;
    // if(MinState>State || MaxState<State) State=MaxState;

    // p_Step->setValue(MinState,MaxState,State);
    p_Step->setValue(0, MinState);
    p_Step->setValue(1, MaxState - MinState);

    // Get geometry key
    // Covise::get_boolean_param("only geometry",&key);
    key = p_only_geometry->getValue();

    // Get output object names
    // Grid = Covise::get_object_name("grid");
    Grid = p_grid->getObjName();

    if (nodalDataType > 0)
    {
        // displacements
        // velocities
        // accelerations
        // Vertex = Covise::get_object_name("data_1");
        Vertex = p_data_1->getObjName();
    }
    if (elementDataType > 0)
    {
        // stress tensor
        // plastic strain
        // Scalar = Covise::get_object_name("data_2");
        Scalar = p_data_2->getObjName();
    }

    //=======================================================================

    readDyna3D();

    return CONTINUE_PIPELINE;
}

int nrelem; // number of rigid body shell elements
int nmmat; // number of materials in the database
int *matTyp = NULL;
int *MaterialTables = NULL;

int ReadDyna3D::readDyna3D()
{

    /* Local variables */
    char buf[256];
    enum // format of cadfem
    {
        GERMAN = 0,
        US // format used by ARUP
    };
    int format;
    static int istate;
    int i;

    for (i = 0; i < MAXTIMESTEPS; i++)
    {
        grid_sets_out[i] = NULL;
        Vertex_sets_out[i] = NULL;
        Scalar_sets_out[i] = NULL;
    }
    NodeIds = NULL;
    SolidNodes = NULL;
    SolidMatIds = NULL;
    BeamNodes = NULL;
    BeamMatIds = NULL;
    ShellNodes = NULL;
    ShellMatIds = NULL;
    TShellNodes = NULL;
    TShellMatIds = NULL;
    SolidElemNumbers = NULL;
    BeamElemNumbers = NULL;
    ShellElemNumbers = NULL;
    TShellElemNumbers = NULL;
    Materials = NULL;
    Vertex_timesteps = NULL;
    Scalar_timesteps = NULL;

    /* initialize TAURUS access */
    taurusinit_();

    // Covise::get_choice_param("format", &format);
    format = p_format->getValue();

    switch (format)
    {

    case GERMAN:
        strncpy(ciform, "cadfem  ", 9L);
        cerr << "ReadDyna3D::readDyna3D info: selecting format " << ciform
             << endl;
        break;

    case US:
        strncpy(ciform, "original", 9L);
        cerr << "ReadDyna3D::readDyna3D info: selecting format " << ciform
             << endl;
        break;

    default:
        sprintf(buf, "ERROR: Incorrect file format selected");
        sendError("%s", buf);
        cerr << "ReadDyna3D::readDyna3D " << buf << endl;
        break;
    };

    /* read TAURUS control data */
    rdtaucntrl_(ciform);

    /* read TAURUS node coordinates */
    rdtaucoor_();

    /* read TAURUS elements */
    rdtauelem_(ciform);

    /* read user node and element numbers */
    rdtaunum_(ciform);
    /* TAURUS geometry length (number of words) */
    taugeoml_();
    /* TAURUS state length (number of words) */
    taustatl_();

    // initialize
    ExistingStates = 0;

    createElementLUT();
    createGeometryList();

    // Only geometry?
    if (!key)
    {

        /******************/
        /* Read Timesteps */
        /******************/
        for (istate = 1; istate < MAXTIMESTEPS; ++istate)
        {
            int flag;
            if (CTrace == 'Y')
            {
                fprintf(stderr, "* test existence of state: %6d\n", istate);
            }

            // only reading when lying in timestep interval
            if ((istate >= MinState) && (istate <= State))
            {
                flag = rdstate_(&istate);
                //	  cerr << "ReadDyna3D::readDyna3D() istate : " << istate << " flag: " << flag <<  endl;
                if (flag == -1)
                    return (-1);

                if (CEndin == 'N')
                {
                    ExistingStates++;
                    fprintf(stderr, "* read state: %2d   time: %12.6f\n", istate, TimestepTime);
                }
            }
            else
            {
                // open next file (!! Assume: number of files and time steps correspond !!)
                otaurusr_();
                if (CEndin == 'N')
                    ExistingStates++;
                else
                    break;
            }
        }
        // mark end of set array
        grid_sets_out[(State - MinState) + 1] = NULL;
        Vertex_sets_out[(State - MinState) + 1] = NULL;
        Scalar_sets_out[(State - MinState) + 1] = NULL;

        grid_timesteps = new coDoSet(Grid, (coDistributedObject **)(void *)grid_sets_out);
        sprintf(buf, "%d %d", MinState, State);
        grid_timesteps->addAttribute("TIMESTEP", buf);
        p_grid->setCurrentObject(grid_timesteps);

        if (nodalDataType > 0)
        {
            Vertex_timesteps = new coDoSet(Vertex, (coDistributedObject **)(void *)Vertex_sets_out);
            Vertex_timesteps->addAttribute("TIMESTEP", buf);
            p_data_1->setCurrentObject(Vertex_timesteps);
        }

        if (elementDataType > 0)
        {
            Scalar_timesteps = new coDoSet(Scalar, (coDistributedObject **)(void *)Scalar_sets_out);
            Scalar_timesteps->addAttribute("TIMESTEP", buf);
            p_data_2->setCurrentObject(Scalar_timesteps);
        }

        // update timestep slider
        if (MaxState > (ExistingStates + 1))
        {
            MaxState = ExistingStates + 1;
            // Covise::update_slider_param("State",1,State,MaxState);
            // p_State->setValue(1,State,MaxState);
            // p_Step->setValue(MinState,State,MaxState);
            p_Step->setValue(0, MinState);
            p_Step->setValue(1, State - MinState);
            MaxState = State;
            NumState = State - MinState;
        }
        if (State > MaxState)
        {
            State = MaxState;
            // Covise::update_slider_param("State",1,State,MaxState);
            // p_State->setValue(1,State,MaxState);
            // p_Step->setValue(MinState,State,MaxState);
            p_Step->setValue(0, MinState);
            p_Step->setValue(1, State - MinState);
            MaxState = State - MinState;
        }

        // free
        for (i = 0; i < State - MinState + 1; i++)
        {
            delete grid_sets_out[i];
            delete Vertex_sets_out[i];
            delete Scalar_sets_out[i];

            grid_sets_out[i] = NULL;
            Vertex_sets_out[i] = NULL;
            Scalar_sets_out[i] = NULL;
        }
    }
    else
    {
        createGeometry();
        grid_sets_out[0] = grid_set_out;
        grid_timesteps = new coDoSet(Grid, (coDistributedObject **)(void *)grid_sets_out);

        // free
        delete grid_sets_out[0];
        grid_sets_out[0] = NULL;
    }

    //free
    //
    for (i = 0; i < NumIDs; i++)
    {
        delete[] My_elemList[i];
        delete[] conList[i];
        delete[] coordLookup[i];
    }
    delete[] My_elemList;
    delete[] conList;
    delete[] coordLookup;

    delete[] numcon;
    delete[] numelem;
    delete[] numcoo;

    for (i = 0; i < NumSolidElements; i++)
        delete solidTab[i];
    delete[] solidTab;
    for (i = 0; i < NumTShellElements; i++)
        delete tshellTab[i];
    delete[] tshellTab;
    for (i = 0; i < NumShellElements; i++)
        delete shellTab[i];
    delete[] shellTab;
    for (i = 0; i < NumBeamElements; i++)
        delete beamTab[i];
    delete[] beamTab;

    delete[] maxSolid;
    delete[] minSolid;
    delete[] maxTShell;
    delete[] minTShell;
    delete[] maxShell;
    delete[] minShell;
    delete[] maxBeam;
    delete[] minBeam;

    delete[] delElem;
    delete[] delCon;

    //  delete grid_timesteps;
    //  delete Vertex_timesteps;
    //  delete Scalar_timesteps;

    delete[] Coord;
    delete[] NodeIds;
    delete[] SolidNodes;
    delete[] SolidMatIds;
    delete[] BeamNodes;
    delete[] BeamMatIds;
    delete[] ShellNodes;
    delete[] ShellMatIds;
    delete[] TShellNodes;
    delete[] TShellMatIds;
    delete[] Materials;

    delete[] SolidElemNumbers;
    delete[] BeamElemNumbers;
    delete[] ShellElemNumbers;
    delete[] TShellElemNumbers;

    delete[] matTyp;
    delete[] MaterialTables;
    matTyp = NULL;
    MaterialTables = NULL;

    return (0);

} /* readDyna3D() */

/* Subroutine */
int ReadDyna3D::taurusinit_()
{
    /* Local variables */

    /*------------------------------------------------------------------------------*/
    /*     initialize TAURUS access */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    /* ITermr = 5;
      ITermw = 6;
      ISesi = 15;
      ISeso = 16;
      IErr = 17;
      IMatdb = 18; */
    COpenin = 'N';
    CEndin = 'N';
    //CTrace = 'Y';
    CTrace = 'N';

    tauio_1.irl = 512;
    /* g77 IRL     = 128 */
    /* iris IRL     = 512 */

    tauio_1.iwpr = tauio_1.irl;
    /* g77 IWPR    = IRL * 4 */
    /* iris IWPR    = IRL */

    return 0;
} /* taurusinit_ */

/* Subroutine */
int ReadDyna3D::rdtaucntrl_(char *ciform)
{
    /* Local variables */
    static int izioshl1, izioshl2, izioshl3, izioshl4, idum[20], iznumelb,
        iznumelh, izmaxint, iznumels, iznumelt;
    static int izndim, iziaflg, iznmatb, izneiph, izitflg, iziuflg,
        izivflg, iznglbv, iznmath, iznvarb, iznarbs, iznvarh, izneips,
        iznmats, iznmatt, iznvars, iznvart;

    /*------------------------------------------------------------------------------*/
    /*     read TAURUS control block */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    int iznver = 15;
    rdrecr_(&version, &iznver, &c__1);

    // Dumb auto detection. Checks if the version number of LSDYNA
    // has a sensible value. CAN FAIL!

    if ((byteswapFlag == BYTESWAP_AUTO) && ((version < 1) || (version > 10000)))
    {

        cerr << "ReadDyna3D::rdtaucntrl_ info: autobyteswap: "
             << "asuming byteswap on (" << version << ")"
             << endl;

        byteswapFlag = BYTESWAP_ON;
        unsigned int *buffer = (unsigned int *)(void *)(tauio_1.tau);
        byteswap(buffer, tauio_1.taulength);

        rdrecr_(&version, &iznver, &c__1);
    }
    else if (byteswapFlag == BYTESWAP_AUTO)
    {
        cerr << "ReadDyna3D::rdtaucntrl_ info: autobyteswap: asuming byteswap off"
             << endl;
        byteswapFlag = BYTESWAP_OFF;
    }

    /* number of dimensions */
    izndim = 16;
    rdreci_(&NumDim, &izndim, &c__1, ciform);
    if (NumDim == 4)
    {
        NumDim = 3;
    }

    /* number of nodal points */
    NumNodalPoints = 17;
    rdreci_(&numcoord, &NumNodalPoints, &c__1, ciform);
    /* number of global variable */
    iznglbv = 19;
    rdreci_(&NumGlobVar, &iznglbv, &c__1, ciform);
    /* flags */
    /*   flag for temp./node varb */
    izitflg = 20;
    rdreci_(&ITFlag, &izitflg, &c__1, ciform);
    /*   flag for geometry */
    iziuflg = 21;
    rdreci_(&IUFlag, &iziuflg, &c__1, ciform);
    /*   flag for velocities */
    izivflg = 22;
    rdreci_(&IVFlag, &izivflg, &c__1, ciform);
    /*   flag for accelerations */
    iziaflg = 23;
    rdreci_(&IAFlag, &iziaflg, &c__1, ciform);

    /* solid elements */
    /*   number of solid elements */
    iznumelh = 24;
    rdreci_(&NumSolidElements, &iznumelh, &c__1, ciform);
    /*   number of solid element materials */
    iznmath = 25;
    rdreci_(&NumSolidMat, &iznmath, &c__1, ciform);

    /*   number of variables in database for each brick element */
    iznvarh = 28;
    rdreci_(&NumSolidVar, &iznvarh, &c__1, ciform);
    //cerr << "NVARH: " << NumSolidVar << endl;

    /* beam elements */
    /*   number of beam elements */
    iznumelb = 29;
    rdreci_(&NumBeamElements, &iznumelb, &c__1, ciform);
    /*   number of beam element materials */
    iznmatb = 30;
    rdreci_(&NumBeamMat, &iznmatb, &c__1, ciform);

    /*   number of variables in database for each 1D element */
    iznvarb = 31;
    rdreci_(&NumBeamVar, &iznvarb, &c__1, ciform);
    //cerr << "NVARB: " << NumBeamVar << endl;

    /* shell elements */
    /*   number of shell elements */
    iznumels = 32;
    rdreci_(&NumShellElements, &iznumels, &c__1, ciform);
    /*   number of shell element materials */
    iznmats = 33;
    rdreci_(&NumShellMat, &iznmats, &c__1, ciform);

    /*   number of variables in database for each 4 node shells */
    iznvars = 34;
    rdreci_(&NumShellVar, &iznvars, &c__1, ciform);
    //cerr << "NVARS: " << NumShellVar << endl;
    /*   number of additional variables per brick element */
    izneiph = 35;
    rdreci_(&NumAddSolidVar, &izneiph, &c__1, ciform);
    /*   number of additional variables at shell integration points */
    izneips = 36;
    rdreci_(&NumAddShellVar, &izneips, &c__1, ciform);
    /*   number of integration points dumped for each shell */
    izmaxint = 37;
    rdreci_(&NumShellInterpol, &izmaxint, &c__1, ciform);
    //    cerr << "MAXINT: " << NumShellInterpol << endl;
    if (NumShellInterpol >= 0)
    {
        MDLOpt = 0;
#ifdef _AIRBUS
        NumShellInterpol = 0;
#endif
        NumInterpol = NumShellInterpol;
    }
    else
    {
        if (NumShellInterpol < 0)
        {
            MDLOpt = 1;
            NumInterpol = abs(NumShellInterpol);
        }
        if (NumShellInterpol < -10000)
        {
            MDLOpt = 2;
            NumInterpol = abs(NumShellInterpol) - 10000;
        }
    }

    /* node and element numbering */
    /*     NARBS = 10 + NUMNP + NUMELH + NUMELS + NUMELT */
    iznarbs = 40;
    rdreci_(&NodeAndElementNumbering, &iznarbs, &c__1, ciform);

    //For Dyna3D we have an additional offset
    int offset = 0;
#ifdef _AIRBUS
    if (version == 0.0)
    {
        offset = 6;
    }
#endif

    /* thick shell elements */
    /*   number of thick shell elements */
    iznumelt = 41 + offset;
    rdreci_(&NumTShellElements, &iznumelt, &c__1, ciform);
    /*   number of thick shell thick shell materials */
    iznmatt = 42 + offset;
    rdreci_(&NumTShellMat, &iznmatt, &c__1, ciform);
    /*   number of variables in database for each thick shell element */
    iznvart = 43 + offset;
    rdreci_(&NumTShellVar, &iznvart, &c__1, ciform);
    //cerr << "NVART: " << NumTShellVar << endl;
    /*   Symmetric stress tensor written flag (1000 = yes) */
    izioshl1 = 44 + offset;
    rdreci_(SomeFlags, &izioshl1, &c__1, ciform);
    /*   Plastic strain written flag          (1000 = yes) */
    izioshl2 = 45 + offset;
    rdreci_(&SomeFlags[1], &izioshl2, &c__1, ciform);
    /*   Shell force resultants written flag  (1000 = yes) */
    izioshl3 = 46 + offset;
    rdreci_(&SomeFlags[2], &izioshl3, &c__1, ciform);
    /*   Shell thickness, energy + 2 others   (1000 = yes) */
    izioshl4 = 47 + offset;
    rdreci_(&SomeFlags[3], &izioshl4, &c__1, ciform);
    if (SomeFlags[0] > 0)
    {
        SomeFlags[0] = 1;
    }
    if (SomeFlags[1] > 0)
    {
        SomeFlags[1] = 1;
    }
    if (SomeFlags[2] > 0)
    {
        SomeFlags[2] = 1;
    }
    if (SomeFlags[3] > 0)
    {
        SomeFlags[3] = 1;
    }
    int position;
    position = 48 + offset;
    rdreci_(&NumFluidMat, &position, &c__1, ciform);
    position = 49 + offset;
    rdreci_(&NCFDV1, &position, &c__1, ciform);
    position = 50 + offset;
    rdreci_(&NCFDV2, &position, &c__1, ciform);
    position = 51 + offset;
    rdreci_(&NADAPT, &position, &c__1, ciform);
    position = 57 + offset;
    rdreci_(&IDTDTFlag, &position, &c__1, ciform);

    /*
   cerr << "IOSHL(1): " << SomeFlags[0] << endl;
   cerr << "IOSHL(2): " << SomeFlags[1] << endl;
   cerr << "IOSHL(3): " << SomeFlags[2] << endl;
   cerr << "IOSHL(4): " << SomeFlags[3] << endl;
   */

    ctrlLen_ = 64;
    if (NumDim == 5)
    {
        ctrlLen_ += 2;
        int off_nrlelem = 65;
        int off_nmmat = off_nrlelem + 1;
        rdreci_(&nrelem, &off_nrlelem, &c__1, ciform);
        rdreci_(&nmmat, &off_nmmat, &c__1, ciform);
        ctrlLen_ += nmmat;
        matTyp = new int[nmmat];
        int off_listMatTyp = off_nmmat + 1;
        rdreci_(matTyp, &off_listMatTyp, &nmmat, ciform);
    }

    if (NumInterpol > 3)
        IStrn = abs(NumShellVar - NumInterpol * (NumAddShellVar + 7) - 12) / 12;
    else
        IStrn = abs(NumShellVar - 3 * (NumAddShellVar + 7) - 12) / 12;

#ifdef _AIRBUS

    if (NumShellElements > 0)
    {
        IStrn = 1;
    }
#endif

    // Calculate the number of temperature/flux/mass-scaling values per node based on ITFlag and IDTDTFlag.
    NumTemperature = ITFlag % 10 + IDTDTFlag;
    if (ITFlag % 10 == 2)
        NumTemperature += 2;
    else if (ITFlag % 10 == 3)
        NumTemperature += 3;
    if (ITFlag >= 10)
        ++NumTemperature;

    /* calculate pointers */
    IZControll = 1;
    NumNodalPoints = IZControll + ctrlLen_;
    IZElements = NumNodalPoints + numcoord * 3;
    IZArb = IZElements + (NumSolidElements + NumTShellElements) * 9 + NumBeamElements * 6 + NumShellElements * 5;
    NumBRS = 0;
    NumMaterials = 0;
    if (NodeAndElementNumbering > 0)
    {
        rdreci_(&NodeSort, &IZArb, &c__1, ciform);
        if (NodeSort >= 0)
        {
            IZStat0 = IZArb + 10 + numcoord + NumSolidElements + NumBeamElements + NumShellElements + NumTShellElements;
        }
        else
        {
            rdreci_(idum, &c__0, &c__13, ciform);
            rdreci_(&NumBRS, &c__0, &c__1, ciform);
            rdreci_(&NumMaterials, &c__0, &c__1, ciform);
            IZStat0 = IZArb + 16 + numcoord + NumSolidElements + NumBeamElements + NumShellElements + NumTShellElements + NumMaterials * 3;
            // @@@ take a look at material tables
            /*
                     int matLength = NumMaterials * 3;
                     MaterialTables = new int[matLength];
                     int IZ_MatTables = IZStat0 - matLength;
                     rdreci_(MaterialTables,&IZ_MatTables,&matLength,ciform);
                     delete [] MaterialTables;
         */
        }
    }
    else
    {
        IZStat0 = IZArb;
    }

    /* write control data to output file */
    // not implemented

    // Info into mapeditor
    sendInfo("File '%s': %i nodal points, %i solid elements, %i thick shell elements, %i shell elements, %i beam elements",
             data_Path, numcoord, NumSolidElements, NumTShellElements, NumShellElements, NumBeamElements);

    fprintf(stderr, "* LSDYNA version                   %g\n", version);
    fprintf(stderr, "* number of dimensions             %d\n", NumDim);
    fprintf(stderr, "* number of nodes                  %d\n", numcoord);
    fprintf(stderr, "* NARBS                            %d\n", NodeAndElementNumbering);
    fprintf(stderr, "* number of solids                 %d\n", NumSolidElements);
    fprintf(stderr, "* number of 2 node beams           %d\n", NumBeamElements);
    fprintf(stderr, "* number of 4 node shells          %d\n", NumShellElements);
    fprintf(stderr, "* number of thick shell elements   %d\n", NumTShellElements);
    fprintf(stderr, "* variables (solids)               %d\n", NumSolidVar);
    fprintf(stderr, "* variables (node beams)           %d\n", NumBeamVar);
    fprintf(stderr, "* variables (4 node shells)        %d\n", NumShellVar);
    fprintf(stderr, "* variables (thick shell elements) %d\n", NumTShellVar);
    fprintf(stderr, "* NGLBV                            %d\n", NumGlobVar);
    fprintf(stderr, "* ITFLG                            %d\n", ITFlag);
    fprintf(stderr, "* IUFLG                            %d\n", IUFlag);
    fprintf(stderr, "* IVFLG                            %d\n", IVFlag);
    fprintf(stderr, "* IAFLG                            %d\n", IAFlag);
    fprintf(stderr, "* NEIPH                            %d\n", NumAddSolidVar);
    fprintf(stderr, "* NEIPS                            %d\n", NumAddShellVar);
    fprintf(stderr, "* ISTRN                            %d\n", IStrn);
    fprintf(stderr, "* IOSHL(1)                         %d\n", SomeFlags[0]);
    fprintf(stderr, "* IOSHL(2)                         %d\n", SomeFlags[1]);
    fprintf(stderr, "* IOSHL(3)                         %d\n", SomeFlags[2]);
    fprintf(stderr, "* IOSHL(4)                         %d\n", SomeFlags[3]);
    fprintf(stderr, "* MAXINT                           %d\n", NumShellInterpol);
    fprintf(stderr, "* NSORT                            %d\n", NodeSort);
    fprintf(stderr, "* NUMBRS                           %d\n", NumBRS);
    fprintf(stderr, "* NMMAT                            %d\n", NumMaterials);
    fprintf(stderr, "* IZCNTRL                          %d\n", IZControll);
    fprintf(stderr, "* IZNUMNP                          %d\n", NumNodalPoints);
    fprintf(stderr, "* IZELEM                           %d\n", IZElements);
    fprintf(stderr, "* IZARB                            %d\n", IZArb);
    fprintf(stderr, "* MDLOPT                           %d\n", MDLOpt);
    fprintf(stderr, "* IDTDTFLG                         %d\n", IDTDTFlag);
    fprintf(stderr, "* NUMRBE                           %d\n", nrelem);
    fprintf(stderr, "* NUMMAT                           %d\n", nmmat);

    return 0;

} /* rdtaucntrl_ */

/* Subroutine */
int ReadDyna3D::taugeoml_()
{
    /*------------------------------------------------------------------------------*/
    /*     TAURUS geometry length (number of words) */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    NumberOfWords = 0;
    /*  Control block and title */
    NumberOfWords += ctrlLen_;
    /* Nodal coordinates */
    NumberOfWords += numcoord * 3;
    /* Solid topology */
    NumberOfWords += NumSolidElements * 9;
    /* Thick shell topology */
    NumberOfWords += NumTShellElements * 9;
    /* Beam topology */
    NumberOfWords += NumBeamElements * 6;
    /* Shell topology */
    NumberOfWords += NumShellElements * 5;
    /* Arbitary numbering tables */
    NumberOfWords += NodeAndElementNumbering;

    if (CTrace == 'Y')
    {
        fprintf(stderr, "* TAURUS geometry length NWORDG: %d\n", NumberOfWords);
    }

    return 0;
} /* taugeoml_ */

/* Subroutine */
int ReadDyna3D::taustatl_()
{
    /* Local variables */
    static int nwordbe, nwordhe, nwordse, nwordte;

    /*------------------------------------------------------------------------------*/
    /*     TAURUS state length (number of words) */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    NumWords = 0;

    if (CTrace == 'Y')
    {
        fprintf(stderr, "* INTPNTS = %d\n", NumInterpol);
    }
    ++NumWords;
    NumWords += NumGlobVar;
    NumWords += numcoord * NumTemperature;
    NumWords += numcoord * 3 * IUFlag;
    NumWords += numcoord * 3 * IVFlag;
    NumWords += numcoord * 3 * IAFlag;
    /* solid elements */
    nwordhe = NumSolidElements * NumSolidVar;
    NumWords += nwordhe;
    /* thick shell elements */
    // new evaluation of NumTShellVar - why? -> uncomment 11.12.97
    if (IStrn < 1)
    {
        //NumTShellVar = (SomeFlags[0] * 6 + SomeFlags[1] + NumAddShellVar) * NumInterpol + IStrn * 12;
        nwordte = NumTShellElements * NumTShellVar;
    }
    else
    {
        //NumTShellVar = (SomeFlags[0] * 6 + SomeFlags[1] + NumAddShellVar) * NumInterpol + 1 * 12;
        nwordte = NumTShellElements * NumTShellVar;
    }
    NumWords += nwordte;
    /* beam elements */
    nwordbe = NumBeamElements * NumBeamVar;
    NumWords += nwordbe;
    /* shell elements */
    nwordse = (NumShellElements - nrelem) * NumShellVar;
    NumWords += nwordse;

    if (NumShellInterpol < 0)
    {
        if (MDLOpt == 1)
        {
            NumWords += numcoord;
        }
        if (MDLOpt == 2)
        {
            NumWords = NumWords + NumSolidElements + NumTShellElements + NumShellElements + NumBeamElements;
        }
    }

    if (CTrace == 'Y')
    {
        fprintf(stderr, "* words for SolidE   elements: %d\n", nwordhe);
        fprintf(stderr, "* words for TSHELL elements: %d\n", nwordte);
        fprintf(stderr, "* words for BEAM   elements: %d\n", nwordbe);
        fprintf(stderr, "* words for SHELL  elements: %d\n", nwordse);
        fprintf(stderr, "* TAURUS state length:           %d\n", NumWords);
    }

    return 0;
} /* taustatl_ */

/* Subroutine */
int ReadDyna3D::rdtaucoor_()
{

    /* Local variables */
    int nxcoor;

    /*------------------------------------------------------------------------------*/
    /*     read TAURUS node coordinates */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    if (CTrace == 'Y')
    {
        fprintf(stderr, "* read node coordinates\n");
    }

    nxcoor = numcoord * 3;
    Coord = new float[nxcoor];

#ifndef _AIRBUS
    rdrecr_(Coord, &NumNodalPoints, &nxcoor);
#else
    if (version != 0.0)
    {
        rdrecr_(Coord, &NumNodalPoints, &nxcoor);
    }
    else
    {
        NumNodalPoints += 3;

        //rdrecr_(Coord, &NumNodalPoints, &nxcoor);
        int ic = 78;

        rdrecr_(Coord, &NumNodalPoints, &ic);

        float dummy;
        int i;
        int cr = 79;

        ic = 612;

        while (cr < numcoord)
        {
            tauio_1.nrin++;
            rdrecr_(Coord + cr * 3, &NumNodalPoints, &ic);
            cr += 204;
        }
    }
#endif

    return 0;
} /* rdtaucoor_ */

/* Subroutine */
int ReadDyna3D::rdtauelem_(char *ciform)
{

    /* Local variables */
    int i;

    /*------------------------------------------------------------------------------*/
    /*     read TAURUS elements */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    if (CTrace == 'Y')
    {
        fprintf(stderr, "* read elements\n");
    }

    /* place pointer to address of 1st element */
    placpnt_(&IZElements);
    /* read solid elements */
    if (NumSolidElements > 0)
    {
        SolidNodes = new int[NumSolidElements * 8];
        SolidMatIds = new int[NumSolidElements];
        for (i = 0; i < NumSolidElements; i++)
        {
            rdreci_(SolidNodes + (i << 3), &c__0, &c__8, ciform);
            rdreci_(SolidMatIds + i, &c__0, &c__1, ciform);
        }
    }
    /* read thick shell elements */
    if (NumTShellElements > 0)
    {
        TShellNodes = new int[NumTShellElements * 8];
        TShellMatIds = new int[NumTShellElements];
        for (i = 0; i < NumTShellElements; i++)
        {
            rdreci_(TShellNodes + (i << 3), &c__0, &c__8, ciform);
            rdreci_(TShellMatIds + i, &c__0, &c__1, ciform);
        }
    }
    /* read beam elements */
    if (NumBeamElements > 0)
    {
        BeamNodes = new int[NumBeamElements * 5];
        BeamMatIds = new int[NumBeamElements];
        for (i = 0; i < NumBeamElements; i++)
        {
            rdreci_(BeamNodes + (i * 5), &c__0, &c__5, ciform);
            rdreci_(BeamMatIds + i, &c__0, &c__1, ciform);
        }
    }
    /* read shell elements */
    if (NumShellElements > 0)
    {
        ShellNodes = new int[NumShellElements * 4];
        ShellMatIds = new int[NumShellElements];
        for (i = 0; i < NumShellElements; i++)
        {
            rdreci_(ShellNodes + (i << 2), &c__0, &c__4, ciform);
            rdreci_(ShellMatIds + i, &c__0, &c__1, ciform);
        }
    }

    // why uncomment?
    /*    rtau_.nel = NumSolidElements + NumBeamElements + NumShellElements + NumTShellElements;*/

    return 0;
} /* rdtauelem_ */

/* Subroutine */
int ReadDyna3D::rdtaunum_(char *ciform)
{
    /* Local variables */
    static float rdum[20];
    /*static int i,j,ie;*/

    /*------------------------------------------------------------------------------*/
    /*     read user node and element numbers */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    if (NodeAndElementNumbering > 0)
    {
        if (CTrace == 'Y')
        {
            fprintf(stderr, "* read user ids\n");
        }
        /* place pointer to address */
        placpnt_(&IZArb);
        if (NumSolidElements > 0)
            SolidElemNumbers = new int[NumSolidElements];
        if (NumBeamElements > 0)
            BeamElemNumbers = new int[NumBeamElements];
        if (NumShellElements > 0)
            ShellElemNumbers = new int[NumShellElements];
        if (NumTShellElements > 0)
            TShellElemNumbers = new int[NumTShellElements];
        NodeIds = new int[numcoord];
        if (NodeSort > 0)
        {
            rdrecr_(rdum, &c__0, &c__10);
            rdreci_(NodeIds, &c__0, &numcoord, ciform);
            rdreci_(SolidElemNumbers, &c__0, &NumSolidElements, ciform);
            rdreci_(BeamElemNumbers, &c__0, &NumBeamElements, ciform);
            rdreci_(ShellElemNumbers, &c__0, &NumShellElements, ciform);
            rdreci_(TShellElemNumbers, &c__0, &NumTShellElements, ciform);
        }
        else
        {
            Materials = new int[NumMaterials];
            rdrecr_(rdum, &c__0, &c__16);
            rdreci_(NodeIds, &c__0, &numcoord, ciform);
            rdreci_(SolidElemNumbers, &c__0, &NumSolidElements, ciform);
            rdreci_(BeamElemNumbers, &c__0, &NumBeamElements, ciform);
            rdreci_(ShellElemNumbers, &c__0, &NumShellElements, ciform);
            rdreci_(TShellElemNumbers, &c__0, &NumTShellElements, ciform);
            rdreci_(Materials, &c__0, &NumMaterials, ciform);
        }

        /*ie = 0;*/
        /* renumber solid element node ids */
        /*	if (NumSolidElements > 0) {
             i__1 = NumSolidElements;
             for (i = 1; i <= i__1; ++i) {
            ++ie;
            for (j = 1; j <= 8; ++j) {
                work_1.lnodes[j + (ie << 3) - 9] = NodeIds[
                   work_1.lnodes[j + (ie << 3) - 9] - 1];
            }
            if (NodeSort < 0) {
                work_1.iemat[ie - 1] = Materials[work_1.iemat[ie - 1] -
                   1];
      }
      }
      } */
        /* renumber thick shell element node ids */
        /*	if (NumTShellElements > 0) {
             i__1 = NumTShellElements;
             for (i = 1; i <= i__1; ++i) {
            ++ie;
            for (j = 1; j <= 8; ++j) {
                work_1.lnodes[j + (ie << 3) - 9] = NodeIds[
                   work_1.lnodes[j + (ie << 3) - 9] - 1];
            }
            if (NodeSort < 0) {
                work_1.iemat[ie - 1] = Materials[work_1.iemat[ie - 1] -
                   1];
      }
      }
      } */
        /* renumber beam element node ids */
        /*	if (NumBeamElements > 0) {
             i__1 = NumBeamElements;
             for (i = 1; i <= i__1; ++i) {
            ++ie;
            for (j = 1; j <= 3; ++j) {
                work_1.lnodes[j + (ie << 3) - 9] = NodeIds[
                   work_1.lnodes[j + (ie << 3) - 9] - 1];
            }
            if (NodeSort < 0) {
                work_1.iemat[ie - 1] = Materials[work_1.iemat[ie - 1] -
                   1];
      }
      }
      } */
        /* renumber shell element node ids */
        /*	if (NumShellElements > 0) {
             i__1 = NumShellElements;
             for (i = 1; i <= i__1; ++i) {
            ++ie;
            for (j = 1; j <= 4; ++j) {
                work_1.lnodes[j + (ie << 3) - 9] = NodeIds[
                   work_1.lnodes[j + (ie << 3) - 9] - 1];
            }
            if (NodeSort < 0) {
                work_1.iemat[ie - 1] = Materials[work_1.iemat[ie - 1] -
                   1];
      }
      }
      } */
    }

    return 0;
} /* rdtaunum_ */

/* Subroutine */
int ReadDyna3D::rdstate_(int *istate)
{
    /* Local variables */
    static int n;
    static int iad;
    static int iaddress;
    static float val;

    /*------------------------------------------------------------------------------*/
    /*     read state */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4 TAU(128) */
    /* iris4 TAU(512) */

    if (CEndin == 'N')
    {
        /* shift state pointer of opened data base file */
        if (*istate == 1)
        {
            tauio_1.nrzin += NumberOfWords;
        }
        else
        {
            tauio_1.nrzin += NumWords;
        }

        /* test if the last record of the state exists. If file is too short,  */
        /* try to open next file. Max file id is 99. When file id 99 i reached */
        /* CENDIN --> Y */
        rdrecr_(&val, &NumWords, &c__1);

        //      cerr << "ReadDyna3D::rdstate_(..) CEndin: " << CEndin << endl;

        if (CEndin == 'N')
        {
            /*         time address = 1 */
            rdrecr_(&TimestepTime, &c__1, &c__1);

            /*         NGLBV (global variables for this state) */
            /*         6 Model data : */
            /*             total kinetic energy */
            /*             internal energy */
            /*             total energy */
            /*             vx vy vz */
            /*         6 data blocks for each material */
            /*             ie */
            /*             ke */
            /*             vx vy vz */
            /*             m */
            /*         for all stonewalls */
            /*             Fn */
            /*         if IAFLG = 1: accelerations */
            /*         if IVFLG = 1: velocities */
            /*         matn: vx vy vz */
            /*         mass of each material */
            /*         internal energy by material */
            /*         kinetic energy by material */
            /*         ? */
            /*         vx vy vz rigid body velocities for each material */
            /*         mass of each material */

            iad = NumGlobVar + 2; // ! skipping current analysis time + global variables
            n = numcoord * 3;

            if (IUFlag == 1)
            {
                DisCo = new float[numcoord * 3];
                rdrecr_(DisCo, &iad, &n);
            }
            else
            {
                sendError("Dyna3D Plotfile doesn't contain nodal coordinate data.");
                return (-1);
            }

            // nodal data stored as followed (at least I think it is, I found no information about this):
            // ( posX posY posZ ) ( temp1 ( temp2 temp3 ) ( tempChange ) ( flux1 flux2 flux3 )) ( massScaling ) ( velX velY velZ ) ( accelX accelY accelZ )
            //       IUFlag      |                              ITFlag and IDTDT                               |     IVFlag       |         IAFlag
            switch (nodalDataType)
            {
            case 0:
                break;

            case 2:
                if (IVFlag == 1)
                {
                    NodeData = new float[numcoord * 3];
                    iaddress = iad + n * IUFlag + numcoord * NumTemperature;
                    rdrecr_(NodeData, &iaddress, &n);
                }
                else
                {
                    sendError("Dyna3D Plotfile doesn't contain nodal velocity data.");
                    return (-1);
                }
                break;

            case 3:
                if (IAFlag == 1)
                {
                    NodeData = new float[numcoord * 3];
                    iaddress = iad + n * IUFlag + numcoord * NumTemperature + n * IVFlag;
                    rdrecr_(NodeData, &iaddress, &n);
                }
                else
                {
                    sendError("Dyna3D Plotfile doesn't contain nodal acceleration data.");
                    return (-1);
                }
                break;
            }

            if (elementDataType > 0)
            {
                int numResults = NumSolidVar * NumSolidElements + NumBeamVar * NumBeamElements
                                 + NumShellVar * (NumShellElements - nrelem) + NumTShellVar * NumTShellElements;
                if (numResults != 0)
                {
                    int nodeLength = numcoord * NumTemperature + n * IUFlag + n * IVFlag + n * IAFlag;

                    /* solid elements */
                    int solidLength = NumSolidElements * NumSolidVar;
                    SolidData = new float[solidLength];
                    iaddress = 2 + NumGlobVar + nodeLength;
                    rdrecr_(SolidData, &iaddress, &solidLength);

                    /* thick shell elements */
                    int tshellLength;
                    tshellLength = NumTShellElements * NumTShellVar;
                    TShellData = new float[tshellLength];
                    iaddress = 2 + NumGlobVar + nodeLength + solidLength;
                    rdrecr_(TShellData, &iaddress, &tshellLength);

                    /* beam elements */
                    int beamLength = NumBeamElements * NumBeamVar;
                    BeamData = new float[beamLength];
                    iaddress = 2 + NumGlobVar + nodeLength + solidLength + tshellLength;
                    rdrecr_(BeamData, &iaddress, &beamLength);

                    /* shell elements */
                    int shellLength = (NumShellElements - nrelem) * NumShellVar;
                    ShellData = new float[shellLength];
                    iaddress = 2 + NumGlobVar + nodeLength + solidLength + tshellLength + beamLength;
                    rdrecr_(ShellData, &iaddress, &shellLength);
                }
                else
                {
                    sendError("Dyna3D Plotfile doesn't contain element results data.");
                    return (-1);
                }
            }

            // lists of deleted nodes/elements ?
            if (NumShellInterpol < 0)
            {
                // read element deletion table
                if (MDLOpt == 1) // old VEC_DYN
                {
                    // not yet supported
                }
                if (MDLOpt == 2) // LS_930
                {
                    int NumElements = NumSolidElements + NumTShellElements + NumShellElements
                                      + NumBeamElements;
                    DelTab = new int[NumElements];
                    // total length of node data
                    int nodeLength = numcoord * NumTemperature + n * IUFlag + n * IVFlag + n * IAFlag;
                    // total number of element results
                    int numResults = NumSolidVar * NumSolidElements + NumBeamVar * NumBeamElements
                                     + NumShellVar * (NumShellElements - nrelem) + NumTShellVar * NumTShellElements;
                    iaddress = 2 + NumGlobVar + nodeLength + numResults;
                    float *tmpTab = new float[NumElements];
                    rdrecr_(tmpTab, &iaddress, &NumElements);
                    for (int i = 0; i < NumElements; ++i)
                    {
                        DelTab[i] = (int)(tmpTab[i]);
                    }
                    delete[] tmpTab;
                    // Notice: The Flags of the deletion table are written in the order:
                    //                Solids, Thick shells, Thin shells, Beams
                }
            }

            if ((*istate >= MinState) && (*istate <= State))
            {
                // This line has been commented out and modified because in case that
                // the files contain more than 1 time step, it may produce wrong
                // informations.
                // sprintf(buf, "Timestep =  %2d \t with time = %6.3f \n", *istate, TimestepTime);
                sendInfo("Timestep with time = %6.3f \n", TimestepTime);

                if (NumShellInterpol < 0)
                {
                    // consider visibility (using the actual deletion table)
                    visibility();
                }
                createStateObjects(*istate);

                grid_sets_out[(*istate) - MinState] = grid_set_out;
                Vertex_sets_out[(*istate) - MinState] = Vertex_set_out;
                Scalar_sets_out[(*istate) - MinState] = Scalar_set_out;
            }

            // free
            delete[] DisCo;
            delete[] NodeData;
            delete[] SolidData;
            delete[] TShellData;
            delete[] BeamData;
            delete[] ShellData;
            delete[] DelTab;
        } // end of if (CEndin == 'N')

    } // end of if (CEndin == 'N') ???

    return 0;
} /* rdstate_ */

/*==============================================================================*/
/* Subroutine */
int ReadDyna3D::rdrecr_(float *val, int *istart, int *n)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int i, irdst, iz;

    /*------------------------------------------------------------------------------*/
    /*     read taurus record for floats */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    /* Parameter adjustments */
    --val;

    /* Function Body */
    if (*n > 0)
    {
        /* open TAURUS database for reading (control data and geometry) */
        if (COpenin == 'N' && CEndin == 'N')
        {
            otaurusr_();
        }
        /* read n values starting at current position */
        i__1 = *n;
        for (i = 1; i <= i__1; ++i)
        {
            /*         get record address */
            grecaddr_(&i, istart, &iz, &irdst);
            if (irdst == 0)
            {
                /*           required word found in this file */
                val[i] = tauio_1.tau[iz - 1];
            }
            else
            {
                /*           required word not found in this file. */
                /*           search and open next file. */
                otaurusr_();
                if (CEndin == 'N')
                {
                    if (*istart > 0)
                    {
                        tauio_1.nrin = tauio_1.nrzin + *istart - 1;
                    }
                    /*             get record address */
                    grecaddr_(&i, istart, &iz, &irdst);
                    if (irdst == 0) // !@@@
                    {
                        /*               required word found in this file */
                        val[i] = tauio_1.tau[iz - 1];
                    }
                    else
                    {
                        /*               error ! */
                        sendError("*E* Opened file is too short! (RDRECI)");
                    }
                }
                else
                {
                    val[i] = (float)0.;
                    return 0;
                }
            }
        }
    }

    return 0;
} /* rdrecr_ */

/* Subroutine */
int ReadDyna3D::rdreci_(int *ival, int *istart, int *n, char *ciform)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int i, irdst, iz;

    /*------------------------------------------------------------------------------*/
    /*     read taurus record for ints */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    /* Parameter adjustments */
    --ival;

    /* Function Body */
    if (*n > 0)
    {
        /* open TAURUS database for reading (control data and geometry) */
        if (COpenin == 'N' && CEndin == 'N')
        {
            otaurusr_();
        }
        /*       read n values starting at current position */
        i__1 = *n;
        for (i = 1; i <= i__1; ++i)
        {
            /*         get record address */
            grecaddr_(&i, istart, &iz, &irdst);
            if (irdst == 0)
            {
                /*           required word found in this file */
                if (strncmp(ciform, "cadfem  ", 8L) == 0)
                {
                    ival[i] = (int)(tauio_1.tau[iz - 1]);
                }
                else if (strncmp(ciform, "original", 8L) == 0)
                {
                    // ???
                    ival[i] = ((int *)(void *)(tauio_1.tau))[iz - 1];
                }
                else
                {
                    cerr << "ReadDyna3D::rdreci_ info: Unknown format '"
                         << ciform << "' selected"
                         << endl;
                }
            }
            else
            {
                /*           required word not found in this file. */
                /*           search and open next file. */
                otaurusr_();
                if (CEndin == 'N')
                {
                    if (*istart > 0)
                    {
                        tauio_1.nrin = tauio_1.nrzin + *istart - 1;
                    }
                    /*             get record address */
                    grecaddr_(&i, istart, &iz, &irdst);
                    if (irdst == 0)
                    {
                        /*               required word found in this file */
                        if (strncmp(ciform, "cadfem  ", 8L) == 0)
                        {
                            ival[i] = (int)(tauio_1.tau[iz - 1]);
                        }
                        else if (strncmp(ciform, "original", 8L) == 0)
                        {
                            ival[i] = ((int *)(void *)tauio_1.tau)[iz - 1];
                        }
                        else
                        {
                            cerr << "ReadDyna3D::rdreci_ info: Unknown format '"
                                 << ciform << "' selected"
                                 << endl;
                        }
                    }
                    else
                    {
                        /*               error ! */
                        if (CTrace == 'Y')
                        {
                            sendError("*E* Opened file is too short! (RDRECI)");
                        }
                    }
                }
                else
                {
                    ival[i] = 0;
                    return 0;
                }
            }
        }
    }

    return 0;
} /* rdreci_ */

/* Subroutine */
int ReadDyna3D::grecaddr_(int *i, int *istart, int *iz, int *irdst)
{
    /* Local variables */
    static int itrecn;

    /*------------------------------------------------------------------------------*/
    /*     get record address for reading */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    *irdst = 0;
    if (*istart > 0)
    {
        tauio_1.nrin = tauio_1.nrzin + *istart + *i - 1;
    }
    else
    {
        ++tauio_1.nrin;
    }
    if (*i == 1 && CTrace == 'Y')
    {
        fprintf(stderr, "* NRIN = %d\n", tauio_1.nrin);
    }
    if (*i > 0)
    {
        itrecn = (tauio_1.nrin - 1) / tauio_1.irl + 1;
        *iz = tauio_1.nrin - (itrecn - 1) * tauio_1.irl;
        if (tauio_1.itrecin != itrecn)
        {
            tauio_1.itrecin = itrecn;
            if (lseek(infile, (itrecn - 1) * sizeof(tauio_1.tau), SEEK_SET) >= 0)
            {
                *irdst = read(infile, (char *)&tauio_1.tau[0], sizeof(tauio_1.tau))
                         - sizeof(tauio_1.tau);

                tauio_1.taulength = *irdst + sizeof(tauio_1.tau) / sizeof(float);
                if (byteswapFlag == BYTESWAP_ON)
                {
                    unsigned int *buffer = (unsigned int *)(void *)(tauio_1.tau);
                    byteswap(buffer, tauio_1.taulength);
                }
            }
            else
            {
                *irdst = -1;
            }
        }
    }

    return 0;
} /* grecaddr_ */

/* Subroutine */
int ReadDyna3D::placpnt_(int *istart)
{
    /* Local variables */

    /*------------------------------------------------------------------------------*/
    /*     place pointer to address ISTART */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    if (*istart > 0)
    {
        tauio_1.nrin = tauio_1.nrzin + *istart - 1;
    }

    return 0;
} /* placpnt_ */

/* Subroutine */
int ReadDyna3D::otaurusr_()
{
    /* Local variables */
    static char ctaun[2000];

    /*------------------------------------------------------------------------------*/
    /*     open a TAURUS database for reading */
    /*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----*/

    /* g77  INTEGER ITAU(128) */
    /* iris INTEGER ITAU(512) */

    /* g77 4                TAU(128) */
    /* iris4                TAU(512) */

    if (COpenin == 'N')
    {
        tauio_1.ifilin = 0;
        tauio_1.nrin = 0;
        tauio_1.nrzin = 0;
        tauio_1.itrecin = 0;
        tauio_1.adaptation = 0;
        tauio_1.adapt[0] = '\0';
        tauio_1.taulength = 0;
        COpenin = 'Y';
        fprintf(stderr, " TAURUS input file : %s\n", data_Path);
        strcpy(CTauin, data_Path);
// infile=Covise::open(CTauin,O_RDONLY);
#ifdef WIN32
        infile = open(CTauin, O_RDONLY | O_BINARY);
#else
        infile = open(CTauin, O_RDONLY);
#endif
        if (infile < 0)
            fprintf(stderr, "could not open %s\n", CTauin);
    }
    else
    {
        if (infile > 0)
            close(infile);

        int numtries = 0;
        while ((tauio_1.ifilin < MAXTIMESTEPS) && (numtries < 10))
        {
            numtries++;
            ++tauio_1.ifilin;
            tauio_1.nrin = 0;
            tauio_1.nrzin = 0;
            tauio_1.itrecin = 0;
            sprintf(ctaun, "%s%s%02d", CTauin, tauio_1.adapt, tauio_1.ifilin);
// infile=Covise::open(ctaun,O_RDONLY);
#ifndef _AIRBUS
#ifdef WIN32
            infile = open(ctaun, O_RDONLY | O_BINARY);
#else
            infile = open(ctaun, O_RDONLY);
#endif
            if (infile > 0)
            {
                fprintf(stderr, "opened %s\n", ctaun);
                return (0);
            }
#else
            if ((version == 0.0 && tauio_1.ifilin % 2 == 1) || version != 0.0)
            {
#ifdef WIN32
                infile = open(ctaun, O_RDONLY | O_BINARY);
#else
                infile = open(ctaun, O_RDONLY);
#endif
                if (infile > 0)
                {
                    fprintf(stderr, "opened %s\n", ctaun);
                    return (0);
                }
            }
#endif
        }
        int num1 = (tauio_1.adaptation) / 26;
        int num2 = (tauio_1.adaptation) - 26 * num1;
        tauio_1.adapt[0] = 'a' + num1;
        tauio_1.adapt[1] = 'a' + num2;
        tauio_1.adapt[2] = '\0';
        tauio_1.adaptation++;
        tauio_1.ifilin = 0;
        tauio_1.nrin = 0;
        tauio_1.nrzin = 0;
        tauio_1.itrecin = 0;
        sprintf(ctaun, "%s%s", CTauin, tauio_1.adapt);
// infile=Covise::open(ctaun,O_RDONLY);
#ifdef WIN32
        infile = open(ctaun, O_RDONLY | O_BINARY);
#else
        infile = open(ctaun, O_RDONLY);
#endif
        if (infile > 0)
        {
            fprintf(stderr, "opened %s\n", ctaun);

            deleteElementLUT();
            /* read TAURUS control data */
            rdtaucntrl_(ciform);

            /* read TAURUS node coordinates */
            rdtaucoor_();
            /* read TAURUS elements */
            rdtauelem_(ciform);

            /* read user node and element numbers */
            rdtaunum_(ciform);
            /* TAURUS geometry length (number of words) */
            taugeoml_();
            /* TAURUS state length (number of words) */
            taustatl_();

            // initialize

            createElementLUT();
            createGeometryList();
            /* shift state pointer of opened data base file */
            tauio_1.nrzin += NumberOfWords;

            return (0);
        }
        CEndin = 'Y';
    }

    return 0;
} /* otaurusr_ */

//---------------------------------------------------------------------

void ReadDyna3D::visibility()
{
    /*                                                        */
    /* decrease the grids in case of invisible nodes/elements */
    /*                                                        */

    int i;
    int elemNo;

    // initialize
    for (i = 0; i < NumIDs; i++)
    {
        delElem[i] = 0;
        delCon[i] = 0;
    }

#ifndef _AIRBUS
    if (MDLOpt == 2) // LS_930
#else
    if (version == 0.0 || MDLOpt == 2)
#endif
    {

        // solids
        for (i = 0; i < NumSolidElements; i++)
        {
            if (DelTab[i] == 0)
            {
                delElem[SolidMatIds[i]]++;
                solidTab[i]->set_visible(INVIS);

                switch (solidTab[i]->get_coType())
                {
                case TYPE_TETRAHEDER:
                    delCon[SolidMatIds[i]] += 4;
                    break;

                case TYPE_PRISM:
                    delCon[SolidMatIds[i]] += 6;
                    break;

                case TYPE_HEXAGON:
                    delCon[SolidMatIds[i]] += 8;
                    break;
                }
            }
        }
        // !!! thick shells omitted

        // thin shells
        for (i = 0; i < NumShellElements; i++)
        {
            elemNo = NumSolidElements + i;
            if (DelTab[elemNo] == 0)
            {
                delElem[ShellMatIds[i]]++;
                shellTab[i]->set_visible(INVIS);

                switch (shellTab[i]->get_coType())
                {

                case TYPE_TRIANGLE:
                    delCon[ShellMatIds[i]] += 3;
                    break;

                case TYPE_QUAD:
                    delCon[ShellMatIds[i]] += 4;
                    break;
                }
            }
        }
        // beams
        for (i = 0; i < NumBeamElements; i++)
        {
            elemNo = NumSolidElements + NumShellElements + i;
            if (DelTab[elemNo] == 0)
            {
                delElem[BeamMatIds[i]]++;
                delCon[BeamMatIds[i]] += 2;
                beamTab[i]->set_visible(INVIS);
            }
        }
    }
    if (MDLOpt == 1) // VEC_DYN
    {
        // not yet supported
    }
}

void ReadDyna3D::createElementLUT()
{
    int i;
    int count;
    int ID;

    // allocate data objects for element lookup tables
    //solids
    solidTab = new Element *[NumSolidElements];
    for (i = 0; i < NumSolidElements; i++)
        solidTab[i] = new Element;
    // thick shells
    tshellTab = new Element *[NumTShellElements];
    for (i = 0; i < NumTShellElements; i++)
        tshellTab[i] = new Element;
    // shells
    shellTab = new Element *[NumShellElements];
    for (i = 0; i < NumShellElements; i++)
        shellTab[i] = new Element;
    // beams
    beamTab = new Element *[NumBeamElements];
    for (i = 0; i < NumBeamElements; i++)
        beamTab[i] = new Element;

    // init material ID counter
    NumIDs = 0;
    // maximum material ID number
    for (i = 0; i < NumSolidElements; i++)
    {
        SolidMatIds[i]--; // !? decrement ?!
        if (SolidMatIds[i] > NumIDs)
            NumIDs = SolidMatIds[i];
    }
    for (i = 0; i < NumTShellElements; i++)
    {
        TShellMatIds[i]--;
        if (TShellMatIds[i] > NumIDs)
            NumIDs = TShellMatIds[i];
    }
    for (i = 0; i < NumShellElements; i++)
    {
        ShellMatIds[i]--;
        if (ShellMatIds[i] > NumIDs)
            NumIDs = ShellMatIds[i];
    }
    for (i = 0; i < NumBeamElements; i++)
    {
        BeamMatIds[i]--;
        if (BeamMatIds[i] > NumIDs)
            NumIDs = BeamMatIds[i];
    }
    NumIDs++;

    // allocate
    numcon = new int[NumIDs];
    numelem = new int[NumIDs];
    numcoo = new int[NumIDs];

    maxSolid = new int[NumIDs];
    minSolid = new int[NumIDs];
    maxTShell = new int[NumIDs];
    minTShell = new int[NumIDs];
    maxShell = new int[NumIDs];
    minShell = new int[NumIDs];
    maxBeam = new int[NumIDs];
    minBeam = new int[NumIDs];

    coordLookup = new int *[NumIDs];

    My_elemList = new int *[NumIDs];
    conList = new int *[NumIDs];

    delElem = new int[NumIDs];
    delCon = new int[NumIDs];

    // initialize
    for (i = 0; i < NumIDs; i++)
    {
        numcon[i] = 0;
        numelem[i] = 0;
        numcoo[i] = 0;

        maxSolid[i] = -1;
        minSolid[i] = NumSolidElements;
        maxTShell[i] = -1;
        minTShell[i] = NumTShellElements;
        maxShell[i] = -1;
        minShell[i] = NumShellElements;
        maxBeam[i] = -1;
        minBeam[i] = NumBeamElements;

        coordLookup[i] = NULL;

        My_elemList[i] = NULL;
        conList[i] = NULL;

        delElem[i] = 0;
        delCon[i] = 0;
    }

    // SETTING TYPE AND NODE NUMBERS
    // solids
    for (i = 0; i < NumSolidElements; i++)
    {

        if ((SolidNodes[i * 8 + 4] == SolidNodes[i * 8 + 5]) && (SolidNodes[i * 8 + 6] == SolidNodes[i * 8 + 7]))
        {
            if (SolidNodes[i * 8 + 5] == SolidNodes[i * 8 + 6])
            {
                numelem[SolidMatIds[i]]++;
                numcon[SolidMatIds[i]] += 4;
                solidTab[i]->set_coType(TYPE_TETRAHEDER);
            }
            else
            {
                numelem[SolidMatIds[i]]++;
                numcon[SolidMatIds[i]] += 6;
                solidTab[i]->set_coType(TYPE_PRISM);
            }
        }
        else
        {
            numelem[SolidMatIds[i]]++;
            numcon[SolidMatIds[i]] += 8;
            solidTab[i]->set_coType(TYPE_HEXAGON);
        }

        /*
      switch ( solidTab[i]->get_coType() ){

        case TYPE_TETRAHEDER:
      for (nn=0; nn<4; nn++)
       solidTab[i]->set_node(nn, SolidNodes[i*8+nn]);
        break;

        case TYPE_PRISM:
      for (nn=0; nn<6; nn++)
       solidTab[i]->set_node(nn, SolidNodes[i*8+nn]);
      break;

      case TYPE_HEXAGON:
      for (nn=0; nn<8; nn++)
      solidTab[i]->set_node(nn, SolidNodes[i*8+nn]);
      break;

      }
      */
    }
    // !!! thick shells omitted
    // thin shells
    for (i = 0; i < NumShellElements; i++)
    {

        if (ShellNodes[i * 4 + 2] == ShellNodes[i * 4 + 3])
        {
            numelem[ShellMatIds[i]]++;
            numcon[ShellMatIds[i]] += 3;
            shellTab[i]->set_coType(TYPE_TRIANGLE);
        }
        else
        {
            numelem[ShellMatIds[i]]++;
            numcon[ShellMatIds[i]] += 4;
            shellTab[i]->set_coType(TYPE_QUAD);
        }

        /*
      switch ( shellTab[i]->get_coType() ){

        case TYPE_TRIANGLE:
      for (nn=0; nn<3; nn++)
       shellTab[i]->set_node(nn, ShellNodes[i*4+nn]);
        break;

        case TYPE_QUAD:
      for (nn=0; nn<4; nn++)
       shellTab[i]->set_node(nn, ShellNodes[i*4+nn]);
      break;

      }
      */
    }
    // beams
    for (i = 0; i < NumBeamElements; i++)
    {
        numelem[BeamMatIds[i]]++;
        numcon[BeamMatIds[i]] += 2;
        beamTab[i]->set_coType(TYPE_BAR);
        /*
      for (nn=0; nn<2; nn++)
        beamTab[i]->set_node(nn, BeamNodes[i*5+nn]);
      */
    }

    // SETTING MATERIAL NUMBER
    for (ID = 0; ID < NumIDs; ID++)
    {
        if (numelem[ID] > 0)
        {
            // solids
            count = -1;
            for (i = 0; i < NumSolidElements; i++)
            {
                if (SolidMatIds[i] == ID)
                {
                    solidTab[i]->set_matNo(ID);
                    maxSolid[ID] = i;
                    if (count == -1)
                    {
                        minSolid[ID] = i;
                        count++;
                    }
                }
            }
            count = -1;
            // thick shells
            for (i = 0; i < NumTShellElements; i++)
            {
                if (TShellMatIds[i] == ID)
                {
                    tshellTab[i]->set_matNo(ID);
                    maxTShell[ID] = i;
                    if (count == -1)
                    {
                        minTShell[ID] = i;
                        count++;
                    }
                }
            }
            count = -1;
            // thin shells
            for (i = 0; i < NumShellElements; i++)
            {
                if (ShellMatIds[i] == ID)
                {
                    shellTab[i]->set_matNo(ID);
                    maxShell[ID] = i;
                    if (count == -1)
                    {
                        minShell[ID] = i;
                        count++;
                    }
                }
            }
            // beams
            count = -1;
            for (i = 0; i < NumBeamElements; i++)
            {
                if (BeamMatIds[i] == ID)
                {
                    beamTab[i]->set_matNo(ID);
                    maxBeam[ID] = i;
                    if (count == -1)
                    {
                        minBeam[ID] = i;
                        count++;
                    }
                }
            }
        }
    }
}

void ReadDyna3D::deleteElementLUT()
{
    int i;
    if (solidTab)
    {
        // free tables of old refinement
        for (i = 0; i < NumSolidElements; i++)
            delete solidTab[i];
    }
    delete[] solidTab;
    if (tshellTab)
    {
        for (i = 0; i < NumTShellElements; i++)
            delete tshellTab[i];
    }
    delete[] tshellTab;
    if (shellTab)
    {
        for (i = 0; i < NumShellElements; i++)
            delete shellTab[i];
    }
    delete[] shellTab;
    if (beamTab)
    {
        for (i = 0; i < NumBeamElements; i++)
            delete beamTab[i];
    }
    delete[] beamTab;
    delete[] numcon;
    delete[] numelem;
    delete[] numcoo;
    delete[] maxSolid;
    delete[] minSolid;
    delete[] maxTShell;
    delete[] minTShell;
    delete[] maxShell;
    delete[] minShell;
    delete[] maxBeam;
    delete[] minBeam;
    if (My_elemList)
    {
        for (i = 0; i < NumIDs; i++)
        {
            delete[] My_elemList[i];
            delete[] conList[i];
            delete[] coordLookup[i];
        }
    }
    delete[] coordLookup;
    delete[] My_elemList;
    delete[] conList;
    delete[] delElem;
    delete[] delCon;
}

void ReadDyna3D::createGeometryList()
{
    int i, j;
    int ID;

    int IDcount = 0; // part ID counter
    int elemNo; // COVISE element list entry
    int coElem; // COVISE element number counter
    int coCon; // COVISE connectivity counter

    for (ID = 0; ID < NumIDs; ID++)
    {
        if (IDLISTE[ID] && numelem[ID] > 0)

        {
            int *lookupTab = new int[numcoord + 1];
            createNodeLUT(ID, lookupTab);

            // allocate element list
            My_elemList[ID] = new int[numelem[ID]];
            // allocate connection list
            conList[ID] = new int[numcon[ID]];

            // initialize element numbering
            elemNo = 0;
            coElem = 0;
            coCon = 0;

#ifndef _AIRBUS
            if (MDLOpt == 2) // LS_930
#else
            if (version == 0.0 || MDLOpt == 2)
#endif
            {

                // solids
                for (i = minSolid[ID]; i <= maxSolid[ID]; i++)
                {
                    if (SolidMatIds[i] == ID)
                    {
                        switch (solidTab[i]->get_coType())
                        {

                        case TYPE_TETRAHEDER:
                            My_elemList[ID][coElem] = elemNo;
                            for (j = 0; j < 4; j++)
                            {
                                conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + j]];
                                coCon++;
                            }
                            elemNo += 4;
                            solidTab[i]->set_coElem(coElem);
                            coElem++;
                            break;

                        case TYPE_PRISM:
                            My_elemList[ID][coElem] = elemNo;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + 4]];
                            coCon++;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + 1]];
                            coCon++;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8]];
                            coCon++;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + 6]];
                            coCon++;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + 2]];
                            coCon++;
                            conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + 3]];
                            coCon++;
                            elemNo += 6;
                            solidTab[i]->set_coElem(coElem);
                            coElem++;
                            break;

                        case TYPE_HEXAGON:
                            My_elemList[ID][coElem] = elemNo;
                            for (j = 0; j < 8; j++)
                            {
                                conList[ID][coCon] = lookupTab[SolidNodes[i * 8 + j]];
                                coCon++;
                            }
                            elemNo += 8;
                            solidTab[i]->set_coElem(coElem);
                            coElem++;
                            break;
                        }
                    }
                }

                // Shells
                for (i = minShell[ID]; i <= maxShell[ID]; i++)
                {
                    if (ShellMatIds[i] == ID)
                    {
                        switch (shellTab[i]->get_coType())
                        {

                        case TYPE_TRIANGLE:
                            My_elemList[ID][coElem] = elemNo;
                            for (j = 0; j < 3; j++)
                            {
                                conList[ID][coCon] = lookupTab[ShellNodes[i * 4 + j]];
                                coCon++;
                            }
                            elemNo += 3;
                            shellTab[i]->set_coElem(coElem);
                            coElem++;
                            break;

                        case TYPE_QUAD:
                            My_elemList[ID][coElem] = elemNo;
                            for (j = 0; j < 4; j++)
                            {
                                conList[ID][coCon] = lookupTab[ShellNodes[i * 4 + j]];
                                coCon++;
                            }
                            elemNo += 4;
                            shellTab[i]->set_coElem(coElem);
                            coElem++;
                            break;
                        }
                    }
                }

                // Beams
                for (i = minBeam[ID]; i <= maxBeam[ID]; i++)
                {
                    if (BeamMatIds[i] == ID)
                    {
                        My_elemList[ID][coElem] = elemNo;
                        for (j = 0; j < 2; j++)
                        {
                            conList[ID][coCon] = lookupTab[BeamNodes[i * 5 + j]];
                            coCon++;
                        }
                        elemNo += 2;
                        beamTab[i]->set_coElem(coElem);
                        coElem++;
                    }
                }

                // nodal data
                // initialize coordinate numbering
                coordLookup[ID] = new int[numcoo[ID]];
                for (i = 1; i <= numcoord; i++)
                {
                    if (lookupTab[i] >= 0)
                    {
                        coordLookup[ID][lookupTab[i]] = i;
                    }
                }

            } // end of if (MDLOpt ...)
            else
            {
                sendError("LS-DYNA Plotfile has wrong version. Only support for version LS-930 and higher.");
                //exit(-1);
            }
            IDcount++;

            //free
            delete[] lookupTab;

        } // end of if ( IDLISTE[ID] && numelem[ID] > 0)
    } // end of for (ID=0 ...)
}

void ReadDyna3D::createGeometry()
{
    char name[256];
    char part_buf[256];

    int i, j;
    int ID;

    int IDcount = 0; // part ID counter
    int conNo; // COVISE connection list entry
    int elemNo; // COVISE element list entry

    // allocate
    grids_out = new coDoUnstructuredGrid *[NumIDs + 1];
    grids_out[NumIDs] = NULL; // mark end of set array

    for (ID = 0; ID < NumIDs; ID++)
    {

        // initialize
        grids_out[ID] = NULL;

        if (IDLISTE[ID] && numelem[ID] > 0)

        {
            int *el, *cl, *tl;
            float *x_c, *y_c, *z_c;

            // mesh
            sprintf(name, "%s_%d_ID%d", Grid, 0, ID);
            grid_out = new coDoUnstructuredGrid(name,
                                                numelem[ID],
                                                numcon[ID],
                                                numcoo[ID],
                                                1);
            grid_out->getAddresses(&el, &cl, &x_c, &y_c, &z_c);
            grid_out->getTypeList(&tl);
            // COLOR attribute
            grid_out->addAttribute("COLOR", colornames[ID % 10]);
            // PART attribute
            sprintf(part_buf, "%d", ID + 1);
            grid_out->addAttribute("PART", part_buf);

            // initialize element numbering
            elemNo = 0;

#ifndef _AIRBUS
            if (MDLOpt == 2) // LS_930
#else
            if (version == 0.0 || MDLOpt == 2)
#endif
            {

                // solids
                for (i = minSolid[ID]; i <= maxSolid[ID]; i++)
                {
                    if (SolidMatIds[i] == ID)
                    {

                        switch (solidTab[i]->get_coType())
                        {

                        case TYPE_TETRAHEDER:
                            *tl++ = TYPE_TETRAHEDER;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 4; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 4;
                            break;

                        case TYPE_PRISM:
                            *tl++ = TYPE_PRISM;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 6; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 6;
                            break;

                        case TYPE_HEXAGON:
                            *tl++ = TYPE_HEXAGON;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 8; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 8;
                            break;
                        }
                    }
                }
                // Shells
                for (i = minShell[ID]; i <= maxShell[ID]; i++)
                {
                    if (ShellMatIds[i] == ID)
                    {

                        switch (shellTab[i]->get_coType())
                        {

                        case TYPE_TRIANGLE:
                            *tl++ = TYPE_TRIANGLE;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][shellTab[i]->get_coElem()];
                            for (j = 0; j < 3; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 3;
                            break;

                        case TYPE_QUAD:
                            *tl++ = TYPE_QUAD;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][shellTab[i]->get_coElem()];
                            for (j = 0; j < 4; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 4;
                            break;
                        }
                    }
                }

                // Beams
                for (i = minBeam[ID]; i <= maxBeam[ID]; i++)
                {
                    if (BeamMatIds[i] == ID)
                    {

                        *tl++ = TYPE_BAR;
                        *el++ = elemNo;
                        conNo = My_elemList[ID][beamTab[i]->get_coElem()];
                        for (j = 0; j < 2; j++)
                        {
                            *cl++ = conList[ID][conNo + j];
                        }
                        elemNo += 2;
                    }
                }

                // nodal data
                for (i = 0; i < numcoo[ID]; i++)
                {
                    x_c[i] = Coord[(coordLookup[ID][i] - 1) * 3];
                    y_c[i] = Coord[(coordLookup[ID][i] - 1) * 3 + 1];
                    z_c[i] = Coord[(coordLookup[ID][i] - 1) * 3 + 2];
                }

            } // end of if (MDLOpt ...)
            else
            {
                sendError("LS-DYNA Plotfile has wrong version. Only support for version LS-930 and higher.");
                exit(-1);
            }
            grids_out[IDcount] = grid_out;
            IDcount++;

        } // end of if ( IDLISTE[ID] && numelem[ID] > 0)
    } // end of for (ID=0 ...)

    // mark end of set array
    grids_out[IDcount] = NULL;

    sprintf(name, "%s_%d", Grid, 0);
    grid_set_out = new coDoSet(name, (coDistributedObject **)grids_out);

    // free
    for (i = 0; i <= IDcount; i++)
    {
        delete grids_out[i];
    }
    delete[] grids_out;
}

void ReadDyna3D::createNodeLUT(int id, int *lut)
{
    int i;
    int nn;

    // init node lookup table
    for (i = 0; i <= numcoord; i++)
        lut[i] = -1;

    // solids
    for (i = minSolid[id]; i <= maxSolid[id]; i++)
    {
        if (SolidMatIds[i] == id)
        {
            for (nn = 0; nn < 8; nn++)
            {
                if (lut[SolidNodes[i * 8 + nn]] < 0)
                {
                    lut[SolidNodes[i * 8 + nn]] = numcoo[id]++;
                }
            }
        }
    }
    // thick shells
    for (i = minTShell[id]; i <= maxTShell[id]; i++)
    {
        if (TShellMatIds[i] == id)
        {
            for (nn = 0; nn < 8; nn++)
            {
                if (lut[TShellNodes[i * 8 + nn]] < 0)
                {
                    lut[TShellNodes[i * 8 + nn]] = numcoo[id]++;
                }
            }
        }
    }
    // shells
    for (i = minShell[id]; i <= maxShell[id]; i++)
    {
        if (ShellMatIds[i] == id)
        {
            for (nn = 0; nn < 4; nn++)
            {
                if (lut[ShellNodes[i * 4 + nn]] < 0)
                {
                    lut[ShellNodes[i * 4 + nn]] = numcoo[id]++;
                }
            }
        }
    }
    // beams
    for (i = minBeam[id]; i <= maxBeam[id]; i++)
    {
        if (BeamMatIds[i] == id)
        {
            for (nn = 0; nn < 2; nn++)
            {
                if (lut[BeamNodes[i * 5 + nn]] < 0)
                {
                    lut[BeamNodes[i * 5 + nn]] = numcoo[id]++;
                }
            }
        }
    }
}

void ReadDyna3D::createStateObjects(int timestep)
{
    char name[256];
    char part_buf[256];
    float druck, vmises;

    int i, j;
    int ID;

    int IDcount = 0; // part ID counter
    int conNo; // COVISE connection list entry
    int elemNo; // COVISE element list entry

    // allocate
    grids_out = new coDoUnstructuredGrid *[NumIDs + 1];
    grids_out[NumIDs] = NULL; // mark end of set array

    Vertexs_out = new coDoVec3 *[NumIDs + 1];
    Vertexs_out[NumIDs] = NULL;
    Scalars_out = new coDoFloat *[NumIDs + 1];
    Scalars_out[NumIDs] = NULL;

    for (ID = 0; ID < NumIDs; ID++)
    {
        // initialize
        grids_out[ID] = NULL;
        Vertexs_out[ID] = NULL;
        Scalars_out[ID] = NULL;

        if (IDLISTE[ID] && numelem[ID] > 0)
        {
            int *el, *cl, *tl;
            float *x_c, *y_c, *z_c;
            float *vx_out = NULL, *vy_out = NULL, *vz_out = NULL;
            float *s_el = NULL;

            // mesh
            sprintf(name, "%s_%d_ID%d", Grid, timestep, ID);
            grid_out = new coDoUnstructuredGrid(name,
                                                numelem[ID] - delElem[ID],
                                                numcon[ID] - delCon[ID],
                                                numcoo[ID],
                                                1);
            grid_out->getAddresses(&el, &cl, &x_c, &y_c, &z_c);
            grid_out->getTypeList(&tl);
            // COLOR attribute
            grid_out->addAttribute("COLOR", colornames[ID % 10]);
            // PART attribute
            sprintf(part_buf, "%d", ID + 1);
            grid_out->addAttribute("PART", part_buf);

            // nodal vector data
            if (nodalDataType > 0)
            {

                sprintf(name, "%s_%d_ID%d", Vertex, timestep, ID);
                Vertex_out = new coDoVec3(name, numcoo[ID]);
                Vertex_out->addAttribute("PART", part_buf);
                Vertex_out->getAddresses(&vx_out, &vy_out, &vz_out);
            }
            else
            {
                Vertex_out = NULL;
            }
            // element scalar data
            if (elementDataType > 0)
            {
                sprintf(name, "%s_%d_ID%d", Scalar, timestep, ID);
                // @@@ This only works if rigid shells are not mixed
                //     in a "part" with other element types!!!!!!
                if (NumDim == 5
                    && minShell[ID] >= 0
                    && minShell[ID] < NumShellElements
                    && ShellMatIds[minShell[ID]] == ID)
                {
                    Scalar_out = new coDoFloat(name, matTyp[ID] != 20 ? numelem[ID] - delElem[ID] : 0);
                }
                else
                {
                    Scalar_out = new coDoFloat(name, numelem[ID] - delElem[ID]);
                }
                Scalar_out->addAttribute("PART", part_buf);
                Scalar_out->getAddress(&s_el);
            }
            else
            {
                Scalar_out = NULL;
            }

            // initialize element numbering
            elemNo = 0;

#ifndef _AIRBUS
            if (MDLOpt == 2) // LS_930
#else
            if (version == 0.0 || MDLOpt == 2)
#endif
            {

                // solids
                for (i = minSolid[ID]; i <= maxSolid[ID]; i++)
                {
                    if (SolidMatIds[i] == ID && solidTab[i]->get_visible() == VIS)
                    {

                        switch (solidTab[i]->get_coType())
                        {

                        case TYPE_TETRAHEDER:
                            *tl++ = TYPE_TETRAHEDER;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 4; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 4;
                            break;

                        case TYPE_PRISM:
                            *tl++ = TYPE_PRISM;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 6; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 6;
                            break;

                        case TYPE_HEXAGON:
                            *tl++ = TYPE_HEXAGON;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][solidTab[i]->get_coElem()];
                            for (j = 0; j < 8; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 8;
                            break;
                        }

                        // element data
                        switch (elementDataType)
                        {

                        case 1: // element stress (tensor component)

                            switch (component)
                            {

                            case 0:
                                *s_el++ = SolidData[i * NumSolidVar];
                                break;

                            case 1:
                                *s_el++ = SolidData[i * NumSolidVar + 1];
                                break;

                            case 2:
                                *s_el++ = SolidData[i * NumSolidVar + 2];
                                break;

                            case 3:
                                *s_el++ = SolidData[i * NumSolidVar + 3];
                                break;

                            case 4:
                                *s_el++ = SolidData[i * NumSolidVar + 4];
                                break;

                            case 5:
                                *s_el++ = SolidData[i * NumSolidVar + 5];
                                break;

                            case 6:
                                druck = SolidData[i * NumSolidVar];
                                druck += SolidData[i * NumSolidVar + 1];
                                druck += SolidData[i * NumSolidVar + 2];
                                druck /= -3.0;
                                *s_el++ = druck;
                                break;
                            case 7:
                                druck = SolidData[i * NumSolidVar];
                                druck += SolidData[i * NumSolidVar + 1];
                                druck += SolidData[i * NumSolidVar + 2];
                                druck /= 3.0;
                                vmises = (SolidData[i * NumSolidVar] - druck) * (SolidData[i * NumSolidVar] - druck);
                                vmises += (SolidData[i * NumSolidVar + 1] - druck) * (SolidData[i * NumSolidVar + 1] - druck);
                                vmises += (SolidData[i * NumSolidVar + 2] - druck) * (SolidData[i * NumSolidVar + 2] - druck);
                                vmises += 2.0f * SolidData[i * NumSolidVar + 3] * SolidData[i * NumSolidVar + 3];
                                vmises += 2.0f * SolidData[i * NumSolidVar + 4] * SolidData[i * NumSolidVar + 4];
                                vmises += 2.0f * SolidData[i * NumSolidVar + 5] * SolidData[i * NumSolidVar + 5];
                                vmises = (float)sqrt(1.5 * vmises);
                                *s_el++ = vmises;
                                break;
                            }

                            break;

                        case 2: // plastic strain
                            *s_el++ = SolidData[i * NumSolidVar + 6 * SomeFlags[0]];
                            break;
                        }
                    }
                }

                // Shells
                int CoviseElement = -1;
                for (i = minShell[ID]; i <= maxShell[ID]; i++)
                {
                    if (ShellMatIds[i] == ID && shellTab[i]->get_visible() == VIS)
                    {
                        ++CoviseElement;
                        switch (shellTab[i]->get_coType())
                        {

                        case TYPE_TRIANGLE:
                            *tl++ = TYPE_TRIANGLE;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][shellTab[i]->get_coElem()];
                            for (j = 0; j < 3; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 3;
                            break;

                        case TYPE_QUAD:
                            *tl++ = TYPE_QUAD;
                            *el++ = elemNo;
                            conNo = My_elemList[ID][shellTab[i]->get_coElem()];
                            for (j = 0; j < 4; j++)
                            {
                                *cl++ = conList[ID][conNo + j];
                            }
                            elemNo += 4;
                            break;
                        }

                        // element data (NEIPS==0 !!!!!!)
                        int jumpIntegPoints = SomeFlags[0] * 6 + SomeFlags[1];

                        if (NumDim == 5
                            && matTyp[ID] == 20)
                        {
                            continue;
                        }
                        switch (elementDataType)
                        {

                        case 1: // element stress (tensor component)
                            float Sav[6];
                            int comp;
                            for (comp = 0; comp < 6; ++comp)
                            {
                                // average stresses over integration points
                                Sav[comp] = 0.0;
                                int intPoint;
                                for (intPoint = 0; intPoint < NumInterpol; ++intPoint)
                                {
                                    Sav[comp] += ShellData[CoviseElement * NumShellVar + comp + jumpIntegPoints * intPoint];
                                }
                                Sav[component] /= NumInterpol;
                            }
                            switch (component)
                            {
                            case 0:
                            case 1:
                            case 2:
                            case 3:
                            case 4:
                            case 5:
                                *s_el++ = Sav[component - 1];
                                break;
                            /*
                                          case 2:
                                                            *s_el++ = ShellData[CoviseElement*NumShellVar+1];
                                                  break;
                                                    case 3:
                                                            *s_el++ = ShellData[CoviseElement*NumShellVar+2];
                                                  break;
                                          case 4:
                                                            *s_el++ = ShellData[CoviseElement*NumShellVar+3];
                                                  break;
                                          case 5:
                              *s_el++ = ShellData[CoviseElement*NumShellVar+4];
                              break;
                              case 6:
                              *s_el++ = ShellData[CoviseElement*NumShellVar+5];
                              break;
                              */
                            case 6:
                                druck = Sav[0];
                                druck += Sav[1];
                                druck += Sav[2];
                                druck /= -3.0;
                                *s_el++ = druck;
                                break;
                            case 7:
                                druck = Sav[0];
                                druck += Sav[1];
                                druck += Sav[2];
                                druck /= 3.0;
                                vmises = (Sav[0] - druck) * (Sav[0] - druck);
                                vmises += (Sav[1] - druck) * (Sav[1] - druck);
                                vmises += (Sav[2] - druck) * (Sav[2] - druck);
                                vmises += 2.0f * Sav[3] * Sav[3];
                                vmises += 2.0f * Sav[4] * Sav[4];
                                vmises += 2.0f * Sav[5] * Sav[5];
                                vmises = (float)sqrt(1.5 * vmises);
                                *s_el++ = vmises;
                                break;
                            }

                            break;

                        case 2: // plastic strain
                            *s_el++ = (ShellData[CoviseElement * NumShellVar + 6 * SomeFlags[0]] + ShellData[CoviseElement * NumShellVar + 6 * SomeFlags[0] + jumpIntegPoints] + ShellData[CoviseElement * NumShellVar + 6 * SomeFlags[0] + jumpIntegPoints * 2]) / 3;
                            break;

                        case 3: // Thickness: NEIPS is 0!!!!!!
                            *s_el++ = ShellData[CoviseElement * NumShellVar + NumInterpol * (6 * SomeFlags[0] + SomeFlags[1]) + 8 * SomeFlags[2]];
                            break;
                        }
                    }
                }

                // Beams
                for (i = minBeam[ID]; i <= maxBeam[ID]; i++)
                {
                    if (BeamMatIds[i] == ID && beamTab[i]->get_visible() == VIS)
                    {

                        *tl++ = TYPE_BAR;
                        *el++ = elemNo;
                        conNo = My_elemList[ID][beamTab[i]->get_coElem()];
                        for (j = 0; j < 2; j++)
                        {
                            *cl++ = conList[ID][conNo + j];
                        }
                        elemNo += 2;

                        // element data
                        switch (elementDataType)
                        {

                        case 1: // element stress (tensor component)
                            // For beams no stress available.
                            *s_el++ = 0.0;
                            break;

                        case 2: // plastic strain
                            // For beams no plastic strain available.
                            *s_el++ = 0.0;
                            ;
                            break;
                        }
                    }
                }

                // nodal data
                for (i = 0; i < numcoo[ID]; i++)
                {
                    x_c[i] = DisCo[(coordLookup[ID][i] - 1) * 3];
                    y_c[i] = DisCo[(coordLookup[ID][i] - 1) * 3 + 1];
                    z_c[i] = DisCo[(coordLookup[ID][i] - 1) * 3 + 2];

                    switch (nodalDataType)
                    {

                    case 0:
                        break;

                    case 1: // displacements
                        vx_out[i] = x_c[i] - Coord[(coordLookup[ID][i] - 1) * 3];
                        vy_out[i] = y_c[i] - Coord[(coordLookup[ID][i] - 1) * 3 + 1];
                        vz_out[i] = z_c[i] - Coord[(coordLookup[ID][i] - 1) * 3 + 2];
                        break;

                    case 2: // nodal velocity
                        vx_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3];
                        vy_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3 + 1];
                        vz_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3 + 2];
                        break;
                    case 3: // nodal acceleration
                        vx_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3];
                        vy_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3 + 1];
                        vz_out[i] = NodeData[(coordLookup[ID][i] - 1) * 3 + 2];
                        break;
                    }
                }

            } // end of if (MDLOpt ...)
            else
            {
                sendError("LS-DYNA Plotfile has wrong version. Only support for version LS-930 and higher.");
                //exit(-1);
            }
            grids_out[IDcount] = grid_out;
            Vertexs_out[IDcount] = Vertex_out;
            Scalars_out[IDcount] = Scalar_out;
            IDcount++;

        } // end of if ( IDLISTE[ID] && numelem[ID] > 0)
    } // end of for (ID=0 ...)

    // mark end of set array
    grids_out[IDcount] = NULL;
    Vertexs_out[IDcount] = NULL;
    Scalars_out[IDcount] = NULL;

    sprintf(name, "%s_%d", Grid, timestep);
    grid_set_out = new coDoSet(name, (coDistributedObject **)grids_out);

    // aw: only creates sets when creating data at the port at all
    if (nodalDataType > 0)
    {
        sprintf(name, "%s_%d", Vertex, timestep);
        Vertex_set_out = new coDoSet(name, (coDistributedObject **)Vertexs_out);
    }
    else
        Vertex_set_out = NULL;

    if (elementDataType > 0)
    {
        sprintf(name, "%s_%d", Scalar, timestep);
        Scalar_set_out = new coDoSet(name, (coDistributedObject **)Scalars_out);
    }
    else
        Scalar_set_out = NULL;

    // free
    for (i = 0; i <= IDcount; i++)
    {
        delete grids_out[i];
        delete Vertexs_out[i];
        delete Scalars_out[i];
    }
    delete[] grids_out;
    delete[] Vertexs_out;
    delete[] Scalars_out;
}

void ReadDyna3D::byteswap(unsigned int *buffer, int length)
{

    for (int ctr = 0; ctr < length; ctr++)
    {
        unsigned int &val = buffer[ctr];
        val = ((val & 0xff000000) >> 24) | ((val & 0x00ff0000) >> 8) | ((val & 0x0000ff00) << 8) | ((val & 0x000000ff) << 24);
    }
}

MODULE_MAIN(IO, ReadDyna3D)
