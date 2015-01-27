/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READDYNA3D_H
#define _READDYNA3D_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Dyna3D data         	                  **
 **                                                                        **
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
/////////////////////////////////////////////////////////
//                I M P O R T A N T
/////////////////////////////////////////////////////////
// NumState keeps the number of time steps to be read
// (apart from the first one).
// MaxState is kept equal to MinState+NumState.
// Now State is also a redundant variable which is kept
// equal to MaxState.
// These redundancies have been introduced in order to facilitate
// that the authors of this code (or perhaps
// someone else who has the courage to accept
// the challenge) may easily recognise its original state and
// eventually improve its functionality and correct
// its original flaws in spite of the
// modifications that have been introduced to make
// the parameter interface (of parameter p_Step,
// the former parameter p_State) simpler.
// These last changes have not contributed perhaps
// to make the code more clear, but also not
// darker as it already was.
//
// The main problem now is how to determine the first time step
// to be read. It works for the first (if MinState==1), but
// the behaviour is not very satisfactory in other cases.
// This depends on how many time steps are contained per file.
// A second improvement would be to read time steps with
// a frequency greater than 1.
/////////////////////////////////////////////////////////

//#include <appl/ApplInterface.h>

#include <api/coModule.h>
using namespace covise;

#include <util/coRestraint.h>
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>

#include "Element.h"

class ReadDyna3D : public coModule
{
public:
    enum // maximum part ID
    {
        MAXID = 200000,
        MAXTIMESTEPS = 1000
    };

    enum
    {
        BYTESWAP_OFF = 0x00,
        BYTESWAP_ON = 0x01,
        BYTESWAP_AUTO = 0x02
    };

private:
    struct
    {
        int itrecin, nrin, nrzin, ifilin, itrecout, nrout, nrzout, ifilout, irl, iwpr, adaptation;
        float tau[512];
        int taulength;
        char adapt[3];

    } tauio_;

    //  member functions
    virtual int compute(const char *port);
    virtual void postInst();
    void byteswap(unsigned int *buffer, int length);

    char ciform[9];
    char InfoBuf[1000];

    int byteswapFlag;

    // Some old global variables have been hidden hier:
    int NumIDs;
    int nodalDataType;
    int elementDataType;
    int component;
    int ExistingStates; // initially = 0;
    int MinState; // initially = 1;
    int MaxState; // initially = 1;
    int NumState; // initially = 1;
    int State; // initially = 1;

    // More "old" global variables
    int IDLISTE[MAXID];
    int *numcoo, *numcon, *numelem;

    // Another block of "old" global variables
    int *maxSolid; // initially = NULL;
    int *minSolid; // initially = NULL;
    int *maxTShell; // initially = NULL;
    int *minTShell; // initially = NULL;
    int *maxShell; // initially = NULL;
    int *minShell; // initially = NULL;
    int *maxBeam; // initially = NULL;
    int *minBeam; // initially = NULL;

    // More and more blocks of "old" global variables to be initialised to 0
    int *delElem; // = NULL;
    int *delCon; //  = NULL;

    Element **solidTab; //  = NULL;
    Element **tshellTab; // = NULL;
    Element **shellTab; //  = NULL;
    Element **beamTab; //   = NULL;

    // storing coordinate positions for all time steps
    int **coordLookup; // = NULL;

    // element and connection list for time steps (considering the deletion table)
    int **My_elemList; // = NULL;
    int **conList; //  = NULL;

    // node/element deletion table
    int *DelTab; // = NULL;

    coDoUnstructuredGrid *grid_out; //    = NULL;
    coDoVec3 *Vertex_out; //  = NULL;
    coDoFloat *Scalar_out; // = NULL;

    coDoUnstructuredGrid **grids_out; //        = NULL;
    coDoVec3 **Vertexs_out; //  = NULL;
    coDoFloat **Scalars_out; // = NULL;

    // Even two additional blocks of "old" global variables
    coDoSet *grid_set_out, *grid_timesteps;
    coDoSet *Vertex_set_out, *Vertex_timesteps;
    coDoSet *Scalar_set_out, *Scalar_timesteps;

    coDoSet *grid_sets_out[MAXTIMESTEPS];
    coDoSet *Vertex_sets_out[MAXTIMESTEPS];
    coDoSet *Scalar_sets_out[MAXTIMESTEPS];

    // Yes, certainly! More "old" global variables!
    int infile;
    int numcoord;
    int *NodeIds; //=NULL;
    int *SolidNodes; //=NULL;
    int *SolidMatIds; //=NULL;
    int *BeamNodes; //=NULL;
    int *BeamMatIds; //=NULL;
    int *ShellNodes; //=NULL;
    int *ShellMatIds; //=NULL;
    int *TShellNodes; //=NULL;
    int *TShellMatIds; //=NULL;
    int *SolidElemNumbers; //=NULL;
    int *BeamElemNumbers; //=NULL;
    int *ShellElemNumbers; //=NULL;
    int *TShellElemNumbers; //=NULL;
    int *Materials; //=NULL;

    /* Common Block Declarations */
    // Coordinates
    float *Coord; //= NULL;
    // displaced coordinates
    float *DisCo; //= NULL;
    float *NodeData; //= NULL;
    float *SolidData; //= NULL;
    float *TShellData; //= NULL;
    float *ShellData; //= NULL;
    float *BeamData; //= NULL;

    // A gigantic block of "old" global variables
    char CTauin[300];
    char CTrace, CEndin, COpenin;
    float TimestepTime; //= 0.0;
    float version; //= 0.0;
    int NumSolidElements; //= 0;
    int NumBeamElements; //= 0;
    int NumShellElements; //= 0;
    int NumTShellElements; //= 0;
    int NumMaterials; //=0;
    int NumDim; //=0;
    int NumGlobVar; //=0;
    int NumWords; //=0;
    int ITFlag; //=0;
    int IUFlag; //=0;
    int IVFlag; //=0;
    int IAFlag; //=0;
    int IDTDTFlag; //=0;
    int NumTemperature; //=0;
    int MDLOpt; //=0;
    int IStrn; //=0;
    int NumSolidMat; //=0;
    int NumSolidVar; //=0;
    int NumAddSolidVar; //=0;
    int NumBeamMat; //=0;
    int NumBeamVar; //=0;
    int NumAddBeamVar; //=0;
    int NumShellMat; //=0;
    int NumShellVar; //=0;
    int NumAddShellVar; //=0;
    int NumTShellMat; //=0;
    int NumTShellVar; //=0;
    int NumShellInterpol; //=0;
    int NumInterpol; //=0;
    int NodeAndElementNumbering; //=0;
    int SomeFlags[5];
    int NumFluidMat;
    int NCFDV1;
    int NCFDV2;
    int NADAPT;
    int NumBRS; //=0;
    int NodeSort; //=0;
    int NumberOfWords; //=0;
    int NumNodalPoints; //=0;
    int IZControll; //=0;
    int IZElements; //=0;
    int IZArb; //=0;
    int IZStat0; //=0;
    int ctrlLen_;

    // A last block
    int c__1; //= 1;
    int c__0; //= 0;
    int c__13; //= 13;
    int c__8; //= 8;
    int c__5; //= 5;
    int c__4; //= 4;
    int c__10; //= 10;
    int c__16; //= 16;

    //  Parameter names
    const char *data_Path;
    int key;

    //  object names
    const char *Grid;
    const char *Scalar;
    const char *Vertex;

    //  Local data

    //  Shared memory data

    //  LS-DYNA3D specific member functions

    int readDyna3D();
    void visibility();

    void createElementLUT();
    void deleteElementLUT();
    void createGeometryList();
    void createGeometry();
    void createNodeLUT(int id, int *lut);

    void createStateObjects(int timestep);

    /* Subroutines */
    int taurusinit_();
    int rdtaucntrl_(char *);
    int taugeoml_();
    int taustatl_();
    int rdtaucoor_();
    int rdtauelem_(char *);
    int rdtaunum_(char *);
    int rdstate_(int *istate);
    int rdrecr_(float *val, int *istart, int *n);
    int rdreci_(int *ival, int *istart, int *n, char *ciform);
    int grecaddr_(int *i, int *istart, int *iz, int *irdst);
    int placpnt_(int *istart);
    int otaurusr_();

    // Ports
    coOutputPort *p_grid;
    coOutputPort *p_data_1;
    coOutputPort *p_data_2;

    // Parameters
    coFileBrowserParam *p_data_path;
    coChoiceParam *p_nodalDataType;
    coChoiceParam *p_elementDataType;
    coChoiceParam *p_component;
    coStringParam *p_Selection;
    // coIntSliderParam *p_State;
    coIntVectorParam *p_Step;
    coChoiceParam *p_format;
    coBooleanParam *p_only_geometry;
    coChoiceParam *p_byteswap;

public:
    /* Constructor */
    ReadDyna3D(int argc, char *argv[]);

    virtual ~ReadDyna3D()
    {
    }
};
#endif // _READDYNA3D_H
