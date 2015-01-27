/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READSTAR09_H
#define _READSTAR09_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Star-CD Files          	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Andreas Werner                             **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  03.01.96  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <star/File29.h>
#include <star/File09.h>
#include <star/File16.h>
#include <util/ChoiceList.h>
#include <api/coModule.h>
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

class ReadStar : public coModule
{

private:
    enum
    {
        NUMPORTS = 16
    };

    coFileBrowserParam *p_dataFile, *p_meshFile;
    coIntVectorParam *p_fromToStep;
    coIntSliderParam *p_timestep;
    coChoiceParam *p_field[NUMPORTS];
    coChoiceParam *p_cellVert;
    coOutputPort *p_data[NUMPORTS], *p_mesh, *p_type, *p_celltab, *p_cpPoly;

    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    // the 'included' CheckUSG modules: return final #vertices
    int checkUSG(coDoUnstructuredGrid *grid);

    // the 'included' CellToVert module : return new object
    // the grid
    coDistributedObject *cellToVert(coDoUnstructuredGrid *grid,
                                    float *elemData[3], // 1 or 3 comp data
                                    const char *name, // obj name for res
                                    const int *trans); // samm trans

    // the 'included' CellToVert module without transform: return new object
    // the grid
    coDistributedObject *cellToVert(coDoUnstructuredGrid *grid,
                                    float *elemData[3], // 1 or 3 comp data
                                    const char *name); // obj name for res

    void computeCell(float *xcoord, float *ycoord, float *zcoord,
                     int *coordInBox, int numCoordInBox,
                     float bbx1, float bby1, float bbz1,
                     float bbx2, float bby2, float bbz2,
                     int maxCoord,
                     int *replBy, int &numCoordToRemove);

    void computeReplaceLists(int num_coord, int *replBy,
                             int *&src2filt, int *&filt2src);

    int handleChangedDataPath(const char *newpath, int inMapLoading, int ignoreErr);
    void handleChangedMeshPath(const char *newpath, int inMapLoading);

    void parseStartStep(const char *, int *, int *);
    void addBlockInfo(coDistributedObject *obj, int timestep, int num_timesteps);

    //  Local data
    File16 *file16;
    char file16_name[512];
    File09 *file09;
    char file9_name[512];
    File29 *file29;
    StarFile *starfile;
    char buf[500]; // buffer: for immediate use only !!!
    int *choice_to_field;
    char *strcpy_cpp(const char *);
    int fromStep, toStep, byStep;
    //int activeFieldChoice[4];
    ChoiceList *choice;

    //int *cells_used;
    //int num_all_cells;
    //int *all_cells_used;

    // choice selection != STAR field No.
    int field_no[NUMPORTS]; //,choiceSel[NUMPORTS];
    const char *fieldName[NUMPORTS];

    // damn immediate mode parameters suck
    long num_timesteps, timestep;

    // filenames received at map loading : do not immediately read it.
    char *mapLoad9, *mapLoad16;

    // actually read the data in here...
    int statData();
    int transData();

public:
    ReadStar(int argc, char *argv[]);

    virtual ~ReadStar();
};
#endif // _READSTAR_H
