/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MESH_H_
#define __MESH_H_

// 15.10.99

#include "StarCD.h"

// forward definitions where we only need pointers in header

namespace covise
{
class File16;
class coDoSet;
class coDoUnstructuredGrid;
class coDoIntArr;
}

/**
 * Class
 *
 */
class StarMesh
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    StarMesh(const StarMesh &);

    /// Assignment operator: NOT  IMPLEMENTED
    StarMesh &operator=(const StarMesh &);

    /// Default constructor: NOT  IMPLEMENTED
    StarMesh();

    // open a ProSTAR file
    File16 *openProstar(const char *dir, const char *casename, const char *ending);

    // Data for the mesh
    float *m_x, *m_y, *m_z;
    int *m_elem, *m_conn, *m_type, *m_cellType;
    int m_numElem, m_numConn, m_numCoord;
    int m_numScal, m_numMat;
    char **m_scalName;

    // Data for the boundary
    int d_numReg;
    struct Bpatch
    {
        int regNo;
        float *x, *y, *z;
        int *poly, *conn;
        int nPoly, nConn, nPoin;
        int numAttr;
        char *attrName[64], *attrVal[64];
    } d_bound[StarCD::MAX_REGIONS];

    // return 0 if ok, -1 on error
    int getRegionPatches(File16 *file16, int regNo, Bpatch *boun, int seq);

    // my status
    int d_status;

    // Translation Prostar-Covise
    StarCD::int32 *d_proToCov;

public:
    /// Constructor: put in File16 location and Region numbers
    StarMesh(const char *dir, const char *casename, const char *file16Name,
             int numReg, const int *regNo);

    /// check whether we found everything: 0 ok, 1 not all regions found, -1 error
    int getStatus();

    /// Destructor : virtual in case we derive objects
    virtual ~StarMesh();

    /// get a coDoSet with the boundary patches
    coDistributedObject **getBCPatches(const char *objName);

    /// get the grid
    coDoUnstructuredGrid *getGrid(const char *objName);

    /// get the Cell types
    coDoIntArr *getCellTypes(const char *objName);

    /// add an attribute to a region: return 0=ok -1=err)
    int addAttrib(int regNo, const char *name, const char *val);

    /// get number of cells
    int numCells();

    /// convert from ProStar numbering to Covise numbering in situ
    void convertMap(StarCD::int32 *index, int num);

    /// get the number of regions we use for interaction
    int getNumInteractRegions();

    /// attach the attributes for the Plot-Data
    int attachPlotAttrib(coDistributedObject *obj);
};
#endif
