/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISEIO_H
#define COVISEIO_H

#include <util/coviseCompat.h>
#include <appl/ApplInterface.h>
#include <file/covWriteFiles.h>
#include <file/covReadFiles.h>
#include <do/coDoData.h>
#include <string>

namespace covise
{

class coDoLines;
class coDoPixelImage;
class coDoPoints;
class coDoPolygons;
class coDoQuads;
class coDoRectilinearGrid;
class coDoSpheres;
class coDoStructuredGrid;
class coDoText;
class coDoTexture;
class coDoTriangleStrips;
class coDoTriangles;
class coDoUniformGrid;
class coDoUnstructuredGrid;

class READEREXPORT CoviseIO
{
private:
    class doInfo
    {
    public:
        doInfo(coDistributedObject *o, int64_t offset)
        {
            obj = o;
            fileOffset = offset;
        }
        coDistributedObject *obj;
        int64_t fileOffset;
    };
    typedef std::vector<doInfo> ObjectList;
    typedef std::list<std::string> ObjectNameList;

    //  Local data
    int n_coord, n_elem, n_conn;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *radius;

    std::string grid_Path;

    //  Shared memory data
    coDoUniformGrid *ugrid;
    coDoPolygons *pol;
    coDoPoints *pts;
    coDoLines *lin;
    coDoTriangles *triang;
    coDoQuads *quads;
    coDoTriangleStrips *tri;
    coDoRectilinearGrid *rgrid;
    coDoStructuredGrid *sgrid;
    coDoUnstructuredGrid *mesh;
    coDoFloat *us3d;
    coDoVec3 *us3dv;
    coDoPixelImage *pixelimage;
    coDoTexture *texture;
    coDoText *txt;
    coDoRGBA *rgba;
    coDoSpheres *sph;

    int setsRead;
    int firstStepToRead;
    int numStepsToRead;
    int skipSteps;

    void readattrib(int fd, coDistributedObject *tmp_Object);
    void skipattrib(int fd);
    coDistributedObject *readData(int fd, const char *Name);
    void skipData(int fd);
    void writeobj(int fd, const coDistributedObject *tmp_Object);
    bool force;
    ObjectNameList objectNameList;
    ObjectList objectList;

protected:
    virtual int covOpenInFile(const char *grid_Path);
    virtual int covCloseInFile(int fd);

public:
    coDistributedObject *ReadFile(const char *filename, const char *ObjectName, bool force = false, int firstStep = 0, int numSteps = 0, int skipSteps = 0);
    int WriteFile(const char *filename, const coDistributedObject *Object);
    CoviseIO()
    {
        force = false;
    }
    virtual ~CoviseIO()
    {
    }
};
}
#endif
