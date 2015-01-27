/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FILE16_H_
#define __FILE16_H_

#include "istreamFTN.h"
#include "StarFile.h"

#include <covise/covise.h>
#include <util/coTypes.h>

/****************************

Element indices:   Cov  =  continuous     , SAMMs split
                   Star =  continuous     , incl. SAMM
                   Pro  =  non-continuous , SAMMs split
*****************************/

namespace covise
{

class STAREXPORT File16 : public StarModelFile
{
private:
    File16(const File16 &);
    File16 &operator=(const File16 &);
    void unblank(char *str, int length);

    // we need this to check whether file has changed
    ino_t d_inode;
    dev_t d_device;

    // if this exists, it gives original index of given new cell
    // index for SAMM conversion
    int *oldOfNewCell;
    int *covToStar;
    int *covToPro;

    // this is a function that dumps a text to somewhere;
    void (*dumper)(const char *);

    // our default: print to stderr
    static void printStderr(const char *text);

public:
    // read File16 from a file
    File16(int fd, void (*dumpFunct)(const char *) = NULL);

    // destructor
    ~File16();

    // check whether constructed File16 object is valid
    int isValid();

    // get the sizes for allocation
    void getMeshSize(int &numCells, int &numConn, int &numVert);

    void getMesh(int *elPtr, int *clPtr, int *tlPtr,
                 float *xPtr, float *yPtr, float *zPtr,
                 int *typPtr);

    void getReducedMesh(
        int *el, int *cl, int *tl,
        int *starToCov,
        float *vx, float *vy, float *vz,
        int *eLenPtr, int *cLenPtr, int *vLenPtr,
        int *typPtr);

    // get the sizes for Region patches
    void getRegionPatchSize(int region, int &numPoly, int &numConn, int &numVert);

    // get the patches for  this region
    void getRegionPatch(int region, int *elPtr, int *clPtr,
                        float *xPtr, float *yPtr, float *zPtr);

    // Access translation table Covise->Prostar
    int getMaxProstarIdx();

    // create the Mapping covise-prostar
    void createMap(int calcSolids);

    // get some info about the data set
    float getScale()
    {
        return scale8;
    }
    int getVersion()
    {
        return jvers;
    }
    int getNumMat()
    {
        return numMaterials;
    }
    int getNumScal()
    {
        return maxscl;
    }
    const char *getScalName(int i);

    // number of vertices for given cell shape
    static const int numVert[8];

    // Numbers for part of USG
    enum
    {
        HEXAGON = 7,
        PRISM = 6,
        PYRAMID = 5,
        TETRAHEDRON = 4,
        QUAD = 3,
        TRIANGLE = 2,
        BAR = 1,
        SAMM = 12
    };

    struct Header1
    {
        int maxn, maxe;
        int is[3];
        int jvers;
        int numw, ni, nj, nk, nf, maxb, ibfill, novice, maxr, nline, maxcy,
            lsctype, rmsize, inpech, isver, maxs, mxbl, istop;
    };

    int maxn, maxe;
    int is[3];
    int jvers;
    int numw, ni, nj, nk, nf, maxb, ibfill, novice, maxr, nline, maxcy,
        lsctype, rmsize, inpech, isver, maxs, mxbl, istop;

    struct CycCoup // Max Number of Cells in Cyclic
    {
        int ncydmf; // and coupled Cells V3000+
        int ncpdmf;
    };

    int ncydmf;
    int ncpdmf;

    struct TitleRead
    {
        char main[80], sub1[80], sub2[80];
    } title;

    struct CellTabEntry *cellTab;
    struct SammTabEntry *sammTab;
    struct BounTabEntry *bounTab;
    int *regionType;
    struct VertexTabEntry *vertexTab;

    struct LSRecord
    {
        int LS[29];
        int IS[99];
        int LS30;
        int nprsf[3];
        int nprobs;
        int dummy[128]; // make sure we have some space...
    } LSrec;

    struct PropInfo
    {
        int is;
        int lmdef[99];
    } propInfo;

    int nbnd[6];

    struct Header2
    {
        int mxtb, lturbi, lturbp, setadd, nsens, npart;
        float scale8;
        int maxcp, loc180, mver, maxscl, istype, mxstb, numcp, ioptbc,
            lturbf, lturbt, maxcrs, pbtol;
        float keysd[6];
    };

    int mxtb, lturbi, lturbp, setadd, nsens, npart;
    float scale8;
    int maxcp, loc180, mver, maxscl, istype, mxstb, numcp, ioptbc,
        lturbf, lturbt, maxcrs, pbtol;
    float keysd[6];

    struct CellTypeEntry *cellType;

    struct CoupledCells22
    {
        int master;
        int slave[24];
    } *cp22;

    struct CoupledCells23
    {
        int master;
        int slave[50];
    } *cp23;

    enum
    {
        MAX_CP = 128
    };
    struct CoupledCells30
    {
        int master;
        int slave[MAX_CP];
        char masterSide;
        char slaveSide[MAX_CP];
    } *cp30; // compressed: fills gaps

    // how many CPs are really saved
    int numRealCP;

    struct RegionSize *regionSize;

    char **scalName;

    int calcSolids, numCovCells, numCovConn,
        numOrigStarCells; // how many cells in the original Star data set
    int *cellShapeArr;

    int *cells_used;

    int numMaterials;

    /// check whether this is the file we read
    //   1 = true
    //   0 = false
    //  -1 = could not stat file
    int isFile(const char *filename);

    //  Data file index [Covise cell number]
    const int *getCovToStar()
    {
        return covToStar;
    }

    // split Prostar index [Covise cell number]
    const int *getCovToPro()
    {
        return covToPro;
    };

    // set the dump device: must be a pointer to 'void funct(const char *)'
    void setDumper(void (*newDumper)(const char *));

    int getNumSAMM()
    {
        return (jvers > 2310) ? pbtol : 0;
    }

    // access CP list:

    // get
    void getCPsizes(int &numVert, int &numConn, int &numPoly);

    // get a specified face from a certain cell:
    //            RETURN shape (0/3/4)=(none/Tri/Quad) + 0-4 vertices
    void getface(int cellNo, int faceNo, int &shape, int vert[4]);

    int findNewCell(int oldCell);

    void getCPPoly(float *xVert, float *yVert, float *zVert,
                   float *xPoly, float *yPoly, float *zPoly,
                   int *polyTab, int *connTab);

    void getMovedRegionPatch(int reqRegion, float *xv, float *yv, float *zv,
                             int *polyPtr, int *connPtr,
                             float *x, float *y, float *z);
};
}
#endif
