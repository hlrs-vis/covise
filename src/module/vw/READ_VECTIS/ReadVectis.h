/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PATRAN_H
#define _PATRAN_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for PATRAN Neutral and Results Files          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Reiner Beller                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.07.97  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include "VectisFile.h"
#include "ChoiceList.h"
#include <covise/covise_vectis.h>

#define SKIP 1
#define NO_SKIP 0

#define V_PRESSURE 1
#define V_DENSITY 2
#define V_TEMPERATURE 3
#define V_PASSIVE_SCALAR 4
#define V_TURBULENCE_ENERGY 5
#define V_TURBULENCE_DISSIPATION 6
#define V_VELOCITY 7
#define V_PATCH_TEMPERATURE_DIRECT 8
#define V_PATCH_FLUID_TEMP 9
#define V_PATCH_HTC 10
#define V_PATCH_SHEAR 11
#define V_PATCH_PRESSURE 12
#define V_PATCH_DENSITY 13
#define V_PATCH_TEMPERATURE_CELLBASED 14
#define V_PATCH_PASSIVE_SCALAR 15
#define V_PATCH_TURBULENCE_ENERGY 16
#define V_PATCH_TURBULENCE_DISSIPATION 17
#define V_PATCH_VELOCITY 18

class CellPointer
{
public:
    int e;
    int w;
    int s;
    int n;
    int l;
    int h;
};

class VectisData
{
    int init;

public:
    float pref;
    float time;
    float cangle;
    int ndrops;
    // pressure
    off_t o_p;
    float *p;
    // density
    off_t o_den;
    float *den;
    // temperature
    off_t o_t;
    float *t;
    // passive scalar
    off_t o_ps1;
    float *ps1;
    // turbulence energy
    off_t o_te;
    float *te;
    // turbulence dissipation
    off_t o_ed;
    float *ed;
    // velocities
    off_t o_uvel;
    float *uvel;
    off_t o_vvel;
    float *vvel;
    off_t o_wvel;
    float *wvel;
    // patch temperature
    off_t o_tpatch;
    float *tpatch;
    // patch fluid temp
    off_t o_tflpatch;
    float *tflpatch;
    // patch HTC
    off_t o_gpatch;
    float *gpatch;
    // patch shear
    off_t o_taupatch;
    float *taupatch;
    // node coordinates (moving mesh)
    off_t o_noco;

    VectisData()
    {
        init = 1;
        o_p = o_den = o_t = o_ps1 = o_te = o_ed = o_uvel = o_vvel = o_wvel = 0;
        o_tpatch = o_tflpatch = o_gpatch = o_taupatch = 0;
        p = den = t = ps1 = te = ed = uvel = vvel = wvel = 0L;
        tpatch = tflpatch = gpatch = taupatch = 0L;
    };
    ~VectisData()
    {
        delete[] p;
        delete[] den;
        delete[] t;
        delete[] ps1;
        delete[] te;
        delete[] ed;
        delete[] uvel;
        delete[] vvel;
        delete[] wvel;
        delete[] tpatch;
        delete[] tflpatch;
        delete[] gpatch;
        delete[] taupatch;
    };
};

class ReadVectis
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    void paramChange(void *);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);

    //  Local data

    ChoiceList *choicelist;
    VectisFile *vectisfile;
    char *file_name;
    int ident;
    float xmin, xmax, ymin, ymax, zmin, zmax;
    int ncells, nts;
    int icube, jcube, kcube, ni, nj, nk;
    float *xndim, *yndim, *zndim;
    int *iglobe, *jglobe, *kglobe;
    int *ilpack, *itpack;
    char *ils, *ile, *jls, *jle, *kls, *kle;
    char *itypew, *itypee, *itypes, *itypen, *itypel, *itypeh;
    float *voln;
    int ncellu, ntu, ncellv, ntv, ncellw, ntw;
    int iafactor, jafactor, kafactor;
    float *areau, *areav, *areaw;
    int *lwus, *leus, *lsvs, *lnvs, *llws, *lhws;
    CellPointer *global_cell2face;
    CellPointer *global_cell2face_check;
    int *polygon2patch;
    int nbpatch, nbtris, nbound, nnode, nnodref, *ncpactual, *mpatch, *ltype;
    int *nodspp, *lbnod, *nodlist;
    float *pts;
    int nfpadr;
    int *nfpol, *lbfpol, *lfpol;

    int no1, no2, no3;
    VectisData *vdata;

public:
    ReadVectis(int argc, char *argv[]);
    ~ReadVectis();
    inline void run()
    {
        Covise::main_loop();
    }
    int ReadLinkageData();
    int ReadTimeStepData(int skip = NO_SKIP);
    int ReadParallelBlock(int);
    int ReadGlobalMeshDimensions(int);
    int ReadScalarCellInformation(int);
    int ReadVelocityFaceInformation(int);
    int ReadPatchInformation(int);
    int ComputeNeighbourList();
    int ResetChoiceList();
    int UpdateChoiceList();
    int WriteVectisMesh();
    int WriteVectisPatch();
    int WriteVectisData(int, int, int);
    int FindInternalCell(int);
};
#endif // _PATRAN_H
