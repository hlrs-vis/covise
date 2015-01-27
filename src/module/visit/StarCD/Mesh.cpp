/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Mesh.h"
#include <appl/ApplInterface.h>
#include <star/File16.h>
#include <api/coModule.h>
#include <assert.h>
#include <sys/param.h> // for MAXPATHLEN
#include <api/coFeedback.h>

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
StarMesh::StarMesh(const StarMesh &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
StarMesh &StarMesh::operator=(const StarMesh &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
StarMesh::StarMesh()
{
    assert(0);
}

/// ------------------------------------------------------------------------

#define STRDUP(x) (strcpy(new char[strlen(x) + 1], (x)))

/// Constructor: put in File16 location and Region numbers
StarMesh::StarMesh(const char *compdir, const char *casename, const char *meshdir,
                   int numReg, const int *regNo)
{
    // initialize all fields
    m_x = m_y = m_z = NULL;
    m_elem = m_conn = m_type = m_cellType = NULL;
    m_scalName = NULL;
    m_numElem = m_numConn = m_numCoord = m_numScal = m_numMat = 0;
    m_scalName = NULL;
    d_numReg = 0;
    d_proToCov = NULL;

    // no mesh yet
    d_status = -1;

    File16 *file16 = openProstar(compdir, casename, meshdir);
    if (!file16)
    {
        return;
    }

    // got a mesh
    d_status = 0;

    // get the size of the grid
    file16->getMeshSize(m_numElem, m_numConn, m_numCoord);

    // alloc data space
    m_x = new float[m_numCoord];
    m_y = new float[m_numCoord];
    m_z = new float[m_numCoord];
    m_elem = new int[m_numElem];
    m_type = new int[m_numElem];
    m_conn = new int[m_numConn];
    m_cellType = new int[m_numConn]; // StarCD cell types

    m_numMat = file16->getNumMat();
    m_numScal = file16->getNumScal();

    if (m_numScal)
    {
        m_scalName = new char *[m_numScal];
        int i;
        for (i = 0; i < m_numScal; i++)
            m_scalName[i] = STRDUP(file16->getScalName(i + 1));
    }
    else
        m_scalName = NULL;

    // get the grid
    file16->getMesh(m_elem, m_conn, m_type, m_x, m_y, m_z, m_cellType);

    // get the region patches
    d_numReg = 0;
    int i;
    for (i = 0; i < numReg; i++)
    {
        if (getRegionPatches(file16, regNo[i], &d_bound[d_numReg], i))
            d_status = 1;
        else
            d_numReg++;
    }

    // get backward mapping: Prostar->Covise
    const int *covToPro = file16->getCovToPro();
    int maxProstar = file16->getMaxProstarIdx();
    d_proToCov = new StarCD::int32[maxProstar + 1];
    for (i = 0; i <= maxProstar; i++)
        d_proToCov[i] = 0;
    for (i = 0; i < m_numElem; i++)
        d_proToCov[covToPro[i] + 1] = i;

    delete file16;
}

/// Destructor

StarMesh::~StarMesh()
{
    int i;
    for (i = 0; i < d_numReg; i++)
    {
        int j;
        for (j = 0; j < d_bound[i].numAttr; j++)
        {
            delete[] d_bound[i].attrName[j];
            delete[] d_bound[i].attrVal[j];
        }
        delete[] d_bound[i].x;
        delete[] d_bound[i].y;
        delete[] d_bound[i].z;
        delete[] d_bound[i].poly;
        delete[] d_bound[i].conn;
    }
    delete[] m_x;
    delete[] m_y;
    delete[] m_z;
    delete[] m_elem;
    delete[] m_conn;
    delete[] m_type;
    delete[] m_cellType;
    delete[] d_proToCov;
}

///////////////////////////////////////////////////////////////////////////////
/// Open ProSTAR file

File16 *StarMesh::openProstar(const char *compdir, const char *casename, const char *meshdir)
{
    /// try to open the file
    char filename[MAXPATHLEN + 64];
    int fd;

    if (meshdir)
    {
        // try case.mdl
        sprintf(filename, "%s/%s.mdl", meshdir, casename);
        fd = open(filename, O_RDONLY);
        if (fd < 0)
        {
            // try case16
            sprintf(filename, "%s/%s16", meshdir, casename);
            fd = open(filename, O_RDONLY);
        }
    }
    else
    {
        // try case.mdl
        sprintf(filename, "%s/%s.mdl", compdir, casename);
        fd = open(filename, O_RDONLY);
        if (fd < 0)
        {
            // try case16
            sprintf(filename, "%s/%s16", compdir, casename);
            fd = open(filename, O_RDONLY);
        }
    }

    if (fd < 0)
        return NULL;

    // use as a File16
    File16 *file16 = new File16(fd);
    if (file16->isValid())
    {
        Covise::sendInfo("%s", filename);
    }
    else
    {
        delete file16;
        file16 = NULL;
    }

    // whether we read it or not: we don't need the file any longer
    close(fd);
    file16->createMap(0); // we can't know whether there are solids...
    return file16;
}

/////////////////////////////////////////////////////////////////////////////////
/// Read Boundary patches from File16 into local buffer

int StarMesh::getRegionPatches(File16 *file16, int regNo, Bpatch *boun, int seq)
{
    static const char *colors[]
        = { "red", "green", "blue", "yellow", "magenta", "cyan" };

    // get sizes
    file16->getRegionPatchSize(regNo, boun->nPoly, boun->nConn, boun->nPoin);

    if (boun->nPoly)
    {
        // alloc space
        boun->x = new float[boun->nPoin];
        boun->y = new float[boun->nPoin];
        boun->z = new float[boun->nPoin];
        boun->poly = new int[boun->nPoly];
        boun->conn = new int[boun->nConn];

        file16->getRegionPatch(regNo, boun->poly, boun->conn,
                               boun->x, boun->y, boun->z);
        boun->regNo = regNo;
        boun->numAttr = 2;
        boun->attrName[0] = strcpy(new char[16], "vertexOrder");
        boun->attrVal[0] = strcpy(new char[8], "2");
        boun->attrName[1] = strcpy(new char[8], "COLOR");
        boun->attrVal[1] = strcpy(new char[8], colors[seq % 6]);

        return 0;
    }
    else
        return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// create a coDoSet with the boundary patches
coDistributedObject **StarMesh::getBCPatches(const char *objName)
{
    coDistributedObject **polygons
        = new coDistributedObject *[StarCD::MAX_REGIONS + 1];

    int i;
    char buffer[128];

    for (i = 0; i < d_numReg; i++)
    {
        Bpatch *boun = &d_bound[i];
        sprintf(buffer, "%s_%d", objName, boun->regNo);

        // make the polygon object
        coDoPolygons *poly
            = new coDoPolygons(buffer, boun->nPoin, boun->x, boun->y, boun->z,
                               boun->nConn, boun->conn,
                               boun->nPoly, boun->poly);
        poly->addAttribute("vertexOrder", "2");
#ifdef VERBOSE
        cerr << "Created Polygon: nPoin=" << nPoin
             << " nConn=" << nConn << " nPoly=" << nPoly << endl;
#endif
        polygons[i] = poly;
    }

    polygons[d_numReg] = NULL;

    return polygons;
}

////////////////////////////////////////////////////////////////////////////////
/// get the grid
coDoUnstructuredGrid *StarMesh::getGrid(const char *objName)
{
    coDoUnstructuredGrid *mesh
        = new coDoUnstructuredGrid((char *)objName, m_numElem, m_numConn, m_numCoord,
                                   m_elem, m_conn, m_x, m_y, m_z, m_type);
    return mesh;
}

/// get the Cell types
coDoIntArr *StarMesh::getCellTypes(const char *objName)
{
    coDoIntArr *type = new coDoIntArr((char *)objName, 1, &m_numElem, m_cellType);
    return type;
}

////////////////////////////////////////////////////////////////////////////////
/// get the status

/// check whether we found everything: 0 ok, 1 not all regions found, -1 error
int StarMesh::getStatus()
{
    return d_status;
}

////////////////////////////////////////////////////////////////////////////////
/// add attribute to region
int StarMesh::addAttrib(int regNo, const char *name, const char *val)
{
    int i = 0;
    Bpatch *boun = d_bound;
    while (i < d_numReg && boun->regNo != regNo)
    {
        i++;
        boun++;
    }

    if (i < d_numReg)
    {
        boun->attrName[boun->numAttr] = strcpy(new char[strlen(name) + 1], name);
        boun->attrVal[boun->numAttr] = strcpy(new char[strlen(val) + 1], val);
        boun->numAttr++;
        return 0;
    }
    else
        return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// get number of cells
int StarMesh::numCells()
{
    return m_numElem;
}

////////////////////////////////////////////////////////////////////////////////
/// convert from ProStar numbering to Covise numbering

void StarMesh::convertMap(StarCD::int32 *index, int num)
{
    int i;
    for (i = 0; i < num; i++)
        index[i] = d_proToCov[index[i]]; // covise counts from 0
}

////////////////////////////////////////////////////////////////////////////////
/// get the number of regions we use for interaction
int StarMesh::getNumInteractRegions()
{
    return d_numReg;
}

////////////////////////////////////////////////////////////////////////////////
/// attach the attributes for the Plot-Data
int StarMesh::attachPlotAttrib(coDistributedObject *obj)
{
    // the Resituals object must be unstructured scalar data
    if (!obj || !obj->isType("USTSDT") || m_numMat <= 0)
    {
        StarCD::getModule()->sendWarning("Coupling module sent no residuals or mesh incorrect");
        return coModule::FAIL;
    }

    // number of residuals received
    coDoFloat *data = (coDoFloat *)obj;
    int numObj = data->getNumPoints();

    // this is what is should be, otherwise we try to correct
    if (numObj != (9 + m_numScal) * m_numMat)
    {
        Covise::sendWarning("Residual descriprion not consistent");
        while (m_numMat && numObj < (9 + m_numScal) * m_numMat)
            m_numMat--;
    }

    int i, j;
    const char **name = new const char *[numObj];
    char **val = new char *[numObj];
    const char *label[63] = { "U", "V", "W", "P", "K", "EPS", "T", "VIS", "DEN" };
    for (i = 0; i < m_numScal; i++)
        label[10 + i] = m_scalName[i];

    char **vPtr = val;
    const char **nPtr = name;

    if (m_numMat > 1)
        for (i = 0; i < m_numMat; i++)
            for (j = 0; i < (9 + m_numScal); j++)
            {
                *nPtr = "LABEL";
                nPtr++;
                *vPtr = new char[32];
                sprintf(*vPtr, "Mat%d: %s", i, label[j]);
                vPtr++;
            }
    else
        for (j = 0; j < (9 + m_numScal); j++)
        {
            *nPtr = "LABEL";
            nPtr++;
            *vPtr = new char[32];
            sprintf(*vPtr, "%s", label[j]);
            vPtr++;
        }

    obj->addAttributes(numObj, name, val);
    obj->addAttribute("MODULE", "StarCD");

    return coModule::SUCCESS;
}
