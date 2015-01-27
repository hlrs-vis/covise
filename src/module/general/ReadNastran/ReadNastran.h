/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READNASTRAN_H
#define _READNASTRAN_H
/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description: Read module for Nastran data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Franz Maurer                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  07.04.1998  V1.0                                                **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include "elemList.h"
#include <util/coviseCompat.h>

// a struct for coordinate systems
typedef struct _CoordinateSystem2
{
    int cid; // the coordinate system id
    int type; // [RECTANGULAR, CYLINDRICAL, SPHERICAL]
    double t[3]; // the translation vector
    double r[3][3]; // the rotation matrix
} CoordinateSystem2;

// a struct for grid connection lines
typedef struct _ConnectionLine
{
    int g[2]; // the grid points for the line
} ConnectionLine;

// a simple arrow
typedef struct _VectorArrow
{
    int gid; // the grid point id
    float length; // the length of the arrow
    float n[3]; // the direction of the arrow
} VectorArrow;

// degrees of freedom (for SPC1 cards)
typedef struct _DOF
{
    int gid; // the grid point id
    int dof; // degrees of freedom
} DOF;

class ReadNastran : public coModule
{

public:
    ReadNastran(int argc, char *argv[]);
    ~ReadNastran();

    // some constants
    //   enum {BLOCKSIZE = 24576, BLOCKDESCSIZE = 5};
    enum
    {
        BLOCKSIZE = 2000000000,
        BLOCKDESCSIZE = 5
    };
    // the states
    enum
    {
        START,
        HEADER,
        BLOCKDESCRIPTION,
        DATABLOCK,
        TRAILER,
        END
    };
    // the substates
    enum
    {
        MYIGNORE,
        MYDEFINITION,
        MYRESULT
    };
    // the result blocks, we support
    enum
    {
        OQG1 = 1,
        OUGV1,
        OEF1X,
        OES1X,
        GPDT
    };
    // the records (for the result tables)
    enum
    {
        DESCRIPTION = 1,
        RECORD0,
        RECORD1,
        RECORD2
    };
    // format codes
    enum
    {
        REAL = 1,
        REAL_IMAGINARY = 2,
        MAGNITUDE_PHASE = 3
    };
    // point types
    enum
    {
        GRID_POINT = 1,
        SCALAR_POINT = 2,
        EXTRA_POINT = 3,
        MODAL_POINT = 4
    };
    // element types (for the result records)
    enum
    {
        CTRIA1 = 6,
        CQUAD4 = 33,
        CBAR = 34
    };

private:
    // member functions
    int compute(const char *port);

    /// calculate the transformation matrix
    void calcTransformMatrix(CoordinateSystem2 *cs, double a[3], double b[3], double c[3]);
    /// free all resources for the module
    void cleanup();
    /// free all resources
    void freeResources();
    /// load a NASTRAN output2 file
    bool load(const char *filename);
    /// print out a coordinate sytem
    void printCS(CoordinateSystem2 *cs, const char *str = 0);
    /// process a NASTRAN block
    int processBlock(int numBytes);
    /// read a NASTRAN card
    int readCard(int numBytes);
    /// read a NASTRAN result set
    int readResult(int numBytes);
    /// transform a coordinate into the world coordinate system
    int transformCoordinate(int csid, double pin[3], float *pout);

    FILE *fp;
    // remember the file
    char op2Path[128];
    // common used string buffer
    char buf[128];

    //  Local data
    int runCnt_;

    // try skipping non-zero bytes between two blocks
    // until now only for one data set necessary
    int try_skipping;

    //  Do we have byte_swapped data?
    bool byte_swapped;

    /// swap field of doubles with length no, assume double is already 4Byte-swapped´
    void swap_double(double *d, int no);

    // ----------------------
    // the coordinate systems
    // ----------------------
    // coordinate system ids
    ElementList<int> csID;
    // the coordinate systems
    ElementList<CoordinateSystem2 *> csList;

    // --------
    // the grid
    // --------
    // grid point ids
    ElementList<int> gridID;
    // grid point coordinate system ids
    ElementList<int> gridCID;
    // grid coordinates
    ElementList<float> gridX;
    ElementList<float> gridY;
    ElementList<float> gridZ;
    // type list
    ElementList<int> typeList;
    // the connection list
    ElementList<int> connectionList;
    // the element list
    ElementList<int> elementList;
    // the element ids
    ElementList<int> elementID;
    // the property ids
    ElementList<int> propertyID;

    // ----------
    // the plotel
    // ----------
    ElementList<ConnectionLine *> plotelList;

    // ---------
    // the conm2
    // ---------
    ElementList<int> conm2List;

    // ---------
    // the force
    // ---------
    ElementList<VectorArrow *> forceList;

    // --------
    // the grav
    // --------
    ElementList<VectorArrow *> gravList;

    // ----------
    // the moment
    // ----------
    ElementList<VectorArrow *> momentList;

    // --------
    // the temp (temperature, of course)
    // --------
    ElementList<float> tempList;

    // --------
    // the rbar
    // --------
    ElementList<ConnectionLine *> rbarList;

    // --------
    // the rbe2
    // --------
    ElementList<ConnectionLine *> rbe2List;

    // --------
    // the spc1
    // --------
    ElementList<DOF *> spc1Trans;
    ElementList<DOF *> spc1Rot;

    // -----------------
    // the displacements
    // -----------------
    ElementList<float> dispX;
    ElementList<float> dispY;
    ElementList<float> dispZ;
    int numDisplacements;
    int numDisplacementSets;

    // -------------------
    // the reaction forces
    // -------------------
    ElementList<float> rfX;
    ElementList<float> rfY;
    ElementList<float> rfZ;

    // ------------------
    // the element forces
    // ------------------
    ElementList<float> efX;
    ElementList<float> efY;
    ElementList<float> efZ;

    // ------------------
    // the element stress
    // ------------------
    ElementList<float> stressList;
    long fibreDistance;
    int numStresses;
    int numStressSets;

    // the number of bytes for a char, int, float and double
    int charSize;
    int intSize;
    int floatSize;
    int doubleSize;
    int wordSize;

    // arrays for the block description and the block
    int blockdesc[BLOCKDESCSIZE];
    char *block;

    int state;
    int substate;
    int card[3];
    int cardFinished;

    // handle broken records
    int recPartSize;
    int flBrokenRecord;
    char record[128];
    char *dataPtr;
    int gridElementID;

    // the result section
    int recordNr;
    int resultID;
    int resElementID;
    int nwds;
    int deviceCode;

    // input
    coFileBrowserParam *output2Path;
    coStringParam *plotelColor;
    coStringParam *conm2Color;
    coFloatParam *conm2Scale;
    coStringParam *forceColor;
    coStringParam *gravColor;
    coStringParam *momentColor;
    coStringParam *rbarColor;
    coStringParam *rbe2Color;
    coStringParam *spc1Color;
    coFloatParam *spc1Scale;
    coIntScalarParam *modeParam;
    coIntScalarParam *fibreDistanceParam;
    coBooleanParam *trySkipping;
    coBooleanParam *dispTransient;

    // output
    coOutputPort *meshOut;
    coOutputPort *typeOut;
    coOutputPort *plotelOut;
    coOutputPort *conm2Out;
    coOutputPort *forceOut;
    coOutputPort *momentOut;
    coOutputPort *gravOut;
    coOutputPort *tempOut;
    coOutputPort *rbarOut;
    coOutputPort *rbe2Out;
    coOutputPort *spc1Out;
    coOutputPort *oqg1Out;
    coOutputPort *ougv1Out;
    coOutputPort *oef1Out;
    coOutputPort *oes1Out;
};
#endif // _READNASTRAN_H
