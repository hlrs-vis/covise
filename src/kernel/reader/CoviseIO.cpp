/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CoviseIO.h"

#include <do/coDoSet.h>
#include <do/coDoPolygons.h>
#include <do/coDoGeometry.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoTexture.h>
#include <do/coDoPixelImage.h>
#include <do/coDoText.h>
#include <do/coDoOctTree.h>
#include <do/coDoData.h>
#include <do/coDoSpheres.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoIntArr.h>

#ifdef _WIN32
#include <io.h>

#define lseek64 _lseeki64
#define ltell64 _telli64
#else
static off_t ltell64(int fd)
{
    return lseek64(fd, 0, SEEK_CUR);
}
#endif

using namespace covise;

typedef struct
{
    int n_elem;
    int n_conn;
    int n_coord;
} USG_HEADER;

typedef struct
{
    int xs;
    int ys;
    int zs;
} STR_HEADER;

int CoviseIO::WriteFile(const char *filename, const coDistributedObject *Object)
{
    if (filename != NULL)
    {
        grid_Path = filename;
        int fd = covOpenOutFile(grid_Path.c_str());
        if (!fd)
        {
            Covise::sendError("failed to open %s for writing: %s", filename, strerror(errno));
            return 0; //bt
        }

        // Write the object
        writeobj(fd, Object);
        objectNameList.clear();

        return covCloseOutFile(fd);
    }

    else
        return 0;
}

coDistributedObject *CoviseIO::ReadFile(const char *filename, const char *objectName, bool force, int firstStep, int numSteps, int skipNumSteps)
{
    coDistributedObject *tmp_obj;
    this->force = force;

    if (filename != NULL)
    {
        grid_Path = filename;
        int fd = this->covOpenInFile(grid_Path.c_str());
        if (!fd)
        {
            Covise::sendError("failed to open %s for reading: %s", filename, strerror(errno));
            return NULL; //bt
        }

        firstStepToRead = firstStep;
        numStepsToRead = numSteps;
        skipSteps = skipNumSteps;
        setsRead = 0;

        tmp_obj = readData(fd, objectName);
        ObjectList::iterator it = objectList.begin();
        it++; // das erste darf nicht geloescht werden, das wird spaeter von simple module gemacht.
        if (it != objectList.end())
            for (; it != objectList.end(); ++it)
                delete it->obj;
        objectList.clear();
        if (!this->covCloseInFile(fd))
        {
            delete tmp_obj;
            return NULL;
        }
        else
            return tmp_obj;
    }
    else
        return NULL;
}

void CoviseIO::writeobj(int fd, const coDistributedObject *data_obj)
{
    coDoSet *set;
    coDoGeometry *geo;

    const char *gtype;
    USG_HEADER usg_h;
    STR_HEADER s_h;
    const coDistributedObject *do1;
    const coDistributedObject *do2;
    const coDistributedObject *do3;
    const coDistributedObject *do4;
    const coDistributedObject *const *objs;
    int numsets, i, numAttr;
    const char **an, **at, **an2, **at2;
    if (data_obj->getRefCount() > 1) // this object is used multiple times, if we have already written this object, we just add a reference
    {
        int n = 0;
        for (ObjectNameList::iterator it = objectNameList.begin(); it != objectNameList.end(); ++it)
        {
            if (*it == data_obj->getName())
            {
                //we found it so insert a reference
                covWriteOBJREF(fd, n);
                return; // that is all
            }
            n++;
        }
    }
    // store the object name in a list for later reference if this object has been referenced
    objectNameList.push_back(std::string(data_obj->getName()));
    if (data_obj != 0L)
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "SETELE") == 0)
        {
            set = (coDoSet *)data_obj;
            objs = set->getAllElements(&numsets);
            covWriteSetBegin(fd, numsets);
            for (i = 0; i < numsets; i++)
                writeobj(fd, objs[i]);
            numAttr = set->getAllAttributes(&an, &at);
            covWriteSetEnd(fd, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "INTARR") == 0)
        {
            coDoIntArr *arr = (coDoIntArr *)data_obj;
            int numDim = arr->getNumDimensions();
            int numElem = arr->getSize();
            numAttr = arr->getAllAttributes(&an, &at);
            covWriteINTARR(fd, numDim, numElem, arr->getDimensionPtr(), arr->getAddress(), (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "INTDT ") == 0)
        {
            coDoInt *arr = (coDoInt *)data_obj;
            int numPoints = arr->getNumPoints();
            numAttr = arr->getAllAttributes(&an, &at);
            covWriteINTDT(fd, numPoints, arr->getAddress(), (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "BYTEDT") == 0)
        {
            coDoByte *arr = (coDoByte *)data_obj;
            int numPoints = arr->getNumPoints();
            numAttr = arr->getAllAttributes(&an, &at);
            covWriteBYTEDT(fd, numPoints, arr->getAddress(), (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "GEOMET") == 0)
        {
            int has_colors, has_normals, has_texture;
            geo = (coDoGeometry *)data_obj;

            do1 = geo->getGeometry();
            do2 = geo->getColors();
            do3 = geo->getNormals();
            do4 = geo->getTexture();

            const int falseVal = 0;
            const int trueVal = !falseVal;
            if (do2)
                has_colors = trueVal;
            else
                has_colors = falseVal;
            if (do3)
                has_normals = trueVal;
            else
                has_normals = falseVal;
            if (do4)
                has_texture = trueVal;
            else
                has_texture = falseVal;

            covWriteGeometryBegin(fd, has_colors, has_normals, has_texture);
            if (do1)
                writeobj(fd, do1);
            if (do2)
                writeobj(fd, do2);
            if (do3)
                writeobj(fd, do3);
            if (do4)
                writeobj(fd, do4);
            numAttr = geo->getAllAttributes(&an, &at);
            covWriteGeometryEnd(fd, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            mesh = (coDoUnstructuredGrid *)data_obj;
            mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
            mesh->getTypeList(&tl);
            mesh->getGridSize(&(usg_h.n_elem), &(usg_h.n_conn), &(usg_h.n_coord));
            numAttr = mesh->getAllAttributes(&an, &at);
            covWriteUNSGRD(fd, usg_h.n_elem, usg_h.n_conn, usg_h.n_coord,
                           el, vl, tl, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            pts = (coDoPoints *)data_obj;
            pts->getAddresses(&x_coord, &y_coord, &z_coord);
            n_elem = pts->getNumPoints();
            numAttr = pts->getAllAttributes(&an, &at);
            covWritePOINTS(fd, n_elem, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "SPHERE") == 0)
        {
            sph = (coDoSpheres *)data_obj;
            sph->getAddresses(&x_coord, &y_coord, &z_coord, &radius);
            n_elem = sph->getNumSpheres();
            numAttr = sph->getAllAttributes(&an, &at);
            covWriteSPHERES(fd, n_elem, x_coord, y_coord, z_coord, radius, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "DOTEXT") == 0)
        {
            char *data;
            txt = (coDoText *)data_obj;
            txt->getAddress(&data);
            n_elem = txt->getTextLength();
            numAttr = txt->getAllAttributes(&an, &at);
            covWriteDOTEXT(fd, n_elem, data, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "POLYGN") == 0)
        {
            pol = (coDoPolygons *)data_obj;
            pol->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = pol->getNumPolygons();
            usg_h.n_conn = pol->getNumVertices();
            usg_h.n_coord = pol->getNumPoints();
            numAttr = pol->getAllAttributes(&an, &at);
            covWritePOLYGN(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord, x_coord, y_coord, z_coord,
                           (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "LINES") == 0)
        {
            lin = (coDoLines *)data_obj;
            lin->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = lin->getNumLines();
            usg_h.n_conn = lin->getNumVertices();
            usg_h.n_coord = lin->getNumPoints();
            numAttr = lin->getAllAttributes(&an, &at);
            covWriteLINES(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "TRITRI") == 0)
        {
            triang = (coDoTriangles *)data_obj;
            triang->getAddresses(&x_coord, &y_coord, &z_coord, &vl);
            usg_h.n_conn = triang->getNumVertices();
            usg_h.n_coord = triang->getNumPoints();
            numAttr = triang->getAllAttributes(&an, &at);
            covWriteTRI(fd, usg_h.n_conn, vl, usg_h.n_coord, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "QUADS") == 0)
        {
            quads = (coDoQuads *)data_obj;
            quads->getAddresses(&x_coord, &y_coord, &z_coord, &vl);
            usg_h.n_conn = quads->getNumVertices();
            usg_h.n_coord = quads->getNumPoints();
            numAttr = quads->getAllAttributes(&an, &at);
            covWriteQUADS(fd, usg_h.n_conn, vl, usg_h.n_coord, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            tri = (coDoTriangleStrips *)data_obj;
            tri->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = tri->getNumStrips();
            usg_h.n_conn = tri->getNumVertices();
            usg_h.n_coord = tri->getNumPoints();
            numAttr = tri->getAllAttributes(&an, &at);
            covWriteTRIANG(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord, x_coord, y_coord, z_coord,
                           (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid = (coDoRectilinearGrid *)data_obj;
            rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            rgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            numAttr = rgrid->getAllAttributes(&an, &at);
            covWriteRCTGRD(fd, s_h.xs, s_h.ys, s_h.zs, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid = (coDoStructuredGrid *)data_obj;
            sgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            sgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            numAttr = sgrid->getAllAttributes(&an, &at);
            covWriteSTRGRD(fd, s_h.xs, s_h.ys, s_h.zs, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            float x_min, y_min, z_min, x_max, y_max, z_max;

            ugrid = (coDoUniformGrid *)data_obj;
            ugrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            ugrid->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);
            numAttr = ugrid->getAllAttributes(&an, &at);
            covWriteUNIGRD(fd, s_h.xs, s_h.ys, s_h.zs,
                           x_min, x_max, y_min, y_max, z_min, z_max, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "USTSDT") == 0)
        {
            us3d = (coDoFloat *)data_obj;
            us3d->getAddress(&z_coord);
            n_elem = us3d->getNumPoints();
            numAttr = us3d->getAllAttributes(&an, &at);
            covWriteUSTSDT(fd, n_elem, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "USTTDT") == 0)
        {
            coDoTensor *ut3d;
            int type;

            ut3d = (coDoTensor *)data_obj;
            ut3d->getAddress(&z_coord);
            type = ut3d->getTensorType();
            n_elem = us3d->getNumPoints();
            numAttr = us3d->getAllAttributes(&an, &at);
            covWriteUSTTDT(fd, n_elem, type, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "RGBADT") == 0)
        {
            rgba = (coDoRGBA *)data_obj;
            rgba->getAddress((int **)(void *)(&z_coord));
            n_elem = rgba->getNumPoints();
            numAttr = rgba->getAllAttributes(&an, &at);
            covWriteRGBADT(fd, n_elem, (int *)z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "USTVDT") == 0)
        {
            us3dv = (coDoVec3 *)data_obj;
            us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
            n_elem = us3dv->getNumPoints();
            numAttr = us3dv->getAllAttributes(&an, &at);
            covWriteUSTVDT(fd, n_elem, x_coord, y_coord, z_coord, (char **)an, (char **)at, numAttr);
        }
        else if (strcmp(gtype, "IMAGE") == 0)
        {
            int intPixelImageWidth = 0, intPixelImageHeight = 0, intPixelImageSize = 0,
                intPixelImageFormatId = 0;
            char *strPixelImageBuffer = NULL;
            pixelimage = (coDoPixelImage *)data_obj;
            numAttr = pixelimage->getAllAttributes(&an, &at);
            intPixelImageWidth = pixelimage->getWidth();
            intPixelImageHeight = pixelimage->getHeight();
            intPixelImageSize = pixelimage->getPixelsize();
            intPixelImageFormatId = pixelimage->getFormat();
            //int intPixelImageBufferLength = intPixelImageHeight*intPixelImageWidth*intPixelImageSize;
            //pointer to pixel data
            strPixelImageBuffer = pixelimage->getPixels();
            covWriteIMAGE(fd, intPixelImageWidth, intPixelImageHeight, intPixelImageSize,
                          intPixelImageFormatId, strPixelImageBuffer, (char **)an, (char **)at, numAttr);
        }

        else if (strcmp(gtype, "TEXTUR") == 0)
        {
            int intNumberOfBorderPixels = 0, intNumberOfComponents = 0, intLevel = 0,
                intNumberOfVertices = 0, intNumberOfCoordinates = 0;
            int *intVertexIndices;
            float **fltCoords;
            texture = (coDoTexture *)data_obj; //our object as texture recognized
            int intPixelImageWidth = 0, intPixelImageHeight = 0, intPixelImageSize = 0,
                intPixelImageFormatId = 0;
            char *strPixelImageBuffer = NULL;
            int numTextAttr;
            coDoPixelImage *pixelimageS;
            pixelimageS = (coDoPixelImage *)texture->getBuffer();
            numAttr = pixelimageS->getAllAttributes(&an, &at);

            intPixelImageWidth = pixelimageS->getWidth();
            intPixelImageHeight = pixelimageS->getHeight();
            intPixelImageSize = pixelimageS->getPixelsize();
            intPixelImageFormatId = pixelimageS->getFormat();
            //int intPixelImageBufferLength = intPixelImageHeight*intPixelImageWidth*intPixelImageSize;
            strPixelImageBuffer = pixelimageS->getPixels();

            intNumberOfBorderPixels = texture->getBorder();
            intNumberOfComponents = texture->getComponents();
            intLevel = texture->getLevel();
            intNumberOfCoordinates = texture->getNumCoordinates();
            //bt: this I found in the file ColorTex.cpp
            intNumberOfVertices = intNumberOfCoordinates;
            intVertexIndices = texture->getVertices();
            fltCoords = texture->getCoordinates();
            numTextAttr = texture->getAllAttributes(&an2, &at2);

            covWriteTEXTUR(fd, intPixelImageWidth, intPixelImageHeight, intPixelImageSize,
                           intPixelImageFormatId, strPixelImageBuffer, (char **)an, (char **)at, numAttr,
                           intNumberOfBorderPixels, intNumberOfComponents, intLevel,
                           intNumberOfCoordinates, intNumberOfVertices, intVertexIndices,
                           fltCoords, (char **)an2, (char **)at2, numTextAttr);
        }
        else if (strcmp(gtype, "OCTREE") == 0)
        {
            int numCellLists, numMacroCellLists, numCellBBoxes, numGridBBoxes;
            int *cellList, *macroCellList;
            float *cellBBox, *gridBBox;
            int *fX, *fY, *fZ, *max_no_levels;

            coDoOctTree *octtree = (coDoOctTree *)data_obj;
            octtree->getAddresses(&cellList, &macroCellList, &cellBBox, &gridBBox, &fX, &fY, &fZ, &max_no_levels);
            numCellLists = octtree->getNumCellLists();
            numMacroCellLists = octtree->getNumMacroCellLists();
            numCellBBoxes = octtree->getNumCellBBoxes();
            numGridBBoxes = octtree->getNumGridBBoxes();

            covWriteOCTREE(fd, numCellLists, numMacroCellLists, numCellBBoxes, numGridBBoxes, cellList, macroCellList, cellBBox, gridBBox, *fX, *fY, *fZ, *max_no_levels);
        }
        else
        {
            Covise::sendError("ERROR: unsupported DataType");
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'mesh_in'");
    }
}

void CoviseIO::readattrib(int fd, coDistributedObject *tmp_Object)
{
    int num = 0, size = 0;
    covReadNumAttributes(fd, &num, &size);
    if (num > 0)
    {
        std::vector<char> buf(size);
        std::vector<char *> atNam(num), atVal(num);
        atNam[0] = &buf[0];
        covReadAttributes(fd, &atNam[0], &atVal[0], num, size);
        if (tmp_Object)
            tmp_Object->addAttributes(num, &atNam[0], &atVal[0]);
    }
    return;
}

int
CoviseIO::covOpenInFile(const char *grid_Path)
{
    return ::covOpenInFile(const_cast<char *>(grid_Path));
}

int
CoviseIO::covCloseInFile(int fd)
{
    return ::covCloseInFile(fd);
}

coDistributedObject *CoviseIO::readData(int fd, const char *Name)
{
    //cout << "readData with object " << Name << "entered " << count << " times" << endl;
    coDoGeometry *geo;
    coDoSet *set;

    char buf[300], Data_Type[7];
    USG_HEADER usg_h;
    STR_HEADER s_h;
    coDistributedObject **tmp_objs, *do1, *do2, *do3, *do4;
    int numsets, i, t2, t3, t4;
    int startstep, endstep;

    //read(fp,Data_Type,6);
    size_t infoIndex = objectList.size();
    objectList.push_back(doInfo(NULL, ltell64(abs(fd))));

    if (::covReadDescription(fd, Data_Type) == -1 && !force)
    {
        Covise::sendError("ERROR: file is probably not a COVISE file - saving you from greater harm");
        Covise::sendError("ERROR: if it's an ASCII COVISE file then try RWCoviseASCII");
        return NULL;
    }
    Data_Type[6] = '\0';

    if (Name != NULL)
    {
        if (strcmp(Data_Type, "OBJREF") == 0)
        {
            // find the object in the object list, if we did not skip it, just return it, otherwise go back and read it.
            int objNum;
            covReadOBJREF(fd, &objNum);
            int n = 0;
            for (ObjectList::iterator it = objectList.begin(); it != objectList.end() && n <= objNum; it++)
            {
                if (n == objNum)
                {
                    if (it->obj)
                    {
                        it->obj->incRefCount();
                        return it->obj;
                    }
                    else
                    {
                        int64_t currentPos;
                        currentPos = ltell64(abs(fd));
                        lseek64(abs(fd), it->fileOffset, SEEK_SET);
                        it->obj = readData(fd, Name);
                        lseek64(abs(fd), currentPos, SEEK_SET);
                        return it->obj;
                    }
                }
                n++;
            }
            return NULL;
        }
        if (strcmp(Data_Type, "SETELE") == 0)
        {
            ::covReadSetBegin(fd, &numsets);
            if (setsRead == 0)
            {
                if ((firstStepToRead < 0) || (firstStepToRead >= numsets))
                {
                    firstStepToRead = 0;
                }
                if (numStepsToRead == 0)
                {
                    numStepsToRead = numsets;
                }
                if (firstStepToRead + numStepsToRead > numsets)
                {
                    numStepsToRead = numsets - firstStepToRead;
                }

                startstep = firstStepToRead;
                endstep = startstep + numStepsToRead - 1;
                //fprintf(stderr,"startstep=%d, endstep=%d\n", startstep, endstep);
            }
            else
            {
                startstep = 0;
                endstep = numsets - 1;
            }
            setsRead++;

            tmp_objs = new coDistributedObject *[numsets + 1];
            for (i = 0; i < startstep; i++)
            {
                skipData(fd);
            }
            int readStep = 0;
            for (i = 0; i <= endstep - startstep; i++)
            {
                sprintf(buf, "%s_%d", Name, readStep);
                tmp_objs[readStep] = readData(fd, buf);
                tmp_objs[readStep + 1] = NULL;
                readStep++;
                for (int n = 0; n < skipSteps && i < endstep - startstep; n++)
                {
                    skipData(fd);
                }
            }
            for (i = endstep + 1; i < numsets; i++)
            {
                skipData(fd);
            }
            tmp_objs[i] = NULL;
            set = new coDoSet(coObjInfo(Name), tmp_objs);
            if (!(set->objectOk()))
            {
                Covise::sendError("ERROR: creation of SETELE object 'mesh' failed");
                return (NULL);
            }
            delete[] tmp_objs;

            readattrib(fd, set);
            objectList[infoIndex].obj = set;
            return (set);
        }
        else if (strcmp(Data_Type, "INTARR") == 0)
        {
            int numDim;
            int numElem;
            covReadDimINTARR(fd, &numDim);

            int *sizes = new int[numDim];
            covReadSizeINTARR(fd, numDim, sizes, &numElem);
            coDoIntArr *arr = new coDoIntArr(coObjInfo(Name), numDim, sizes);
            covReadINTARR(fd, numDim, numElem, sizes, arr->getAddress());

            readattrib(fd, arr);
            objectList[infoIndex].obj = arr;
            return (arr);
        }
        else if (strcmp(Data_Type, "INTDT ") == 0)
        {
            covReadSizeINTDT(fd, &n_elem);
            coDoInt *intobj = new coDoInt(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = intobj;
            if (intobj->objectOk())
            {
                int *data;
                intobj->getAddress(&data);
                covReadINTDT(fd, n_elem, data);
                readattrib(fd, intobj);
                return (intobj);
            }
            else
            {
                Covise::sendError("ERROR: creation of INTDT  object failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "BYTEDT") == 0)
        {
            covReadSizeBYTEDT(fd, &n_elem);
            coDoByte *byteobj = new coDoByte(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = byteobj;
            if (byteobj->objectOk())
            {
                unsigned char *data;
                byteobj->getAddress(&data);
                covReadBYTEDT(fd, n_elem, data);
                readattrib(fd, byteobj);
                return (byteobj);
            }
            else
            {
                Covise::sendError("ERROR: creation of BYTEDT object failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "GEOMET") == 0)
        {
            int ho1, ho2, ho3;
            covReadOldGeometryBegin(fd, &ho1, &ho2, &ho3);

            do1 = do2 = do3 = NULL;

            t2 = t3 = 0;
            if (ho1)
            {
                sprintf(buf, "%s_Geo", Name);
                do1 = readData(fd, buf);
            }
            if (ho2)
            {
                sprintf(buf, "%s_Col", Name);
                do2 = readData(fd, buf);
            }
            if (ho3)
            {
                sprintf(buf, "%s_Norm", Name);
                do3 = readData(fd, buf);
            }

            if (do1)
            {
                geo = new coDoGeometry(coObjInfo(Name), do1);
                objectList[infoIndex].obj = geo;
                if (!(geo->objectOk()))
                {
                    Covise::sendError("ERROR: creation of GEOMET object 'mesh' failed");
                    return (NULL);
                }
                //geo->setGeometry(t1, do1);
                if (do2)
                    geo->setColors(t2, do2);
                if (do3)
                    geo->setNormals(t3, do3);

                readattrib(fd, geo);

                return geo;
            }
        }
        else if (strcmp(Data_Type, "GEOTEX") == 0)
        {
            // read flag values into int var
            int ho1, ho2, ho3, ho4;
            covReadGeometryBegin(fd, &ho1, &ho2, &ho3, &ho4);

            do1 = do2 = do3 = do4 = NULL;

            t2 = t3 = t4 = 0;
            if (ho1)
            {
                sprintf(buf, "%s_Geo", Name);
                do1 = readData(fd, buf);
            }
            if (ho2)
            {
                sprintf(buf, "%s_Col", Name);
                do2 = readData(fd, buf);
            }
            if (ho3)
            {
                sprintf(buf, "%s_Norm", Name);
                do3 = readData(fd, buf);
            }
            if (ho4)
            {
                sprintf(buf, "%s_Texture", Name);
                do4 = readData(fd, buf);
            }
            if (do1)
            {
                geo = new coDoGeometry(coObjInfo(Name), do1);
                objectList[infoIndex].obj = geo;
                if (!(geo->objectOk()))
                {
                    Covise::sendError("ERROR: creation of GEOMET object 'mesh' failed");
                    return (NULL);
                }
                //geo->setGeometry(t1, do1);
                if (do2)
                    geo->setColors(t2, do2);
                if (do3)
                    geo->setNormals(t3, do3);
                if (do4)
                    geo->setTexture(t4, do4);
                readattrib(fd, geo);
                return geo;
            }
        }
        else if (strcmp(Data_Type, "UNSGRD") == 0)
        {

            covReadSizeUNSGRD(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);

            mesh = new coDoUnstructuredGrid(coObjInfo(Name), usg_h.n_elem, usg_h.n_conn, usg_h.n_coord, 1);
            objectList[infoIndex].obj = mesh;
            if (mesh->objectOk())
            {
                mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
                mesh->getTypeList(&tl);
                covReadUNSGRD(fd, usg_h.n_elem, usg_h.n_conn, usg_h.n_coord,
                              el, vl, tl, x_coord, y_coord, z_coord);
                readattrib(fd, mesh);
                return (mesh);
            }
            else
            {
                Covise::sendError("ERROR: creation of UNSGRD object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "POLYGN") == 0)
        {
            covReadSizeUNSGRD(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
            pol = new coDoPolygons(coObjInfo(Name), usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            objectList[infoIndex].obj = pol;
            if (pol->objectOk())
            {
                pol->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                covReadPOLYGN(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
                              x_coord, y_coord, z_coord);
                readattrib(fd, pol);
                return (pol);
            }
            else
            {
                Covise::sendError("ERROR: creation of POLYGN object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "POINTS") == 0)
        {
            covReadSizePOINTS(fd, &n_elem);
            pts = new coDoPoints(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = pts;
            if (pts->objectOk())
            {
                pts->getAddresses(&x_coord, &y_coord, &z_coord);
                covReadPOINTS(fd, n_elem, x_coord, y_coord, z_coord);
                readattrib(fd, pts);
                return (pts);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "SPHERE") == 0)
        {
            covReadSizeSPHERES(fd, &n_elem);
            sph = new coDoSpheres(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = sph;
            if (sph->objectOk())
            {
                sph->getAddresses(&x_coord, &y_coord, &z_coord, &radius);
                covReadSPHERES(fd, n_elem, x_coord, y_coord, z_coord, radius);
                readattrib(fd, sph);
                return (sph);
            }
            else
            {
                Covise::sendError("Error creation of data object 'Sphere' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "DOTEXT") == 0)
        {
            char *data;
            covReadSizeDOTEXT(fd, &n_elem);
            txt = new coDoText(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = txt;
            if (txt->objectOk())
            {
                txt->getAddress(&data);
                covReadDOTEXT(fd, n_elem, data);
                readattrib(fd, txt);
                return (txt);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "LINES") == 0)
        {
            covReadSizeLINES(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
            lin = new coDoLines(coObjInfo(Name), usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            objectList[infoIndex].obj = lin;
            if (lin->objectOk())
            {
                lin->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                covReadLINES(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
                             x_coord, y_coord, z_coord);
                readattrib(fd, lin);
                return (lin);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "TRIANG") == 0)
        {
            covReadSizeTRIANG(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
            tri = new coDoTriangleStrips(coObjInfo(Name), usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            objectList[infoIndex].obj = tri;
            if (tri->objectOk())
            {
                tri->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                covReadTRIANG(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
                              x_coord, y_coord, z_coord);
                readattrib(fd, tri);
                return (tri);
            }
            else
            {
                Covise::sendError("ERROR: creation of TRIANG object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "RCTGRD") == 0)
        {
            covReadSizeRCTGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs);
            rgrid = new coDoRectilinearGrid(coObjInfo(Name), s_h.xs, s_h.ys, s_h.zs);
            objectList[infoIndex].obj = rgrid;
            if (rgrid->objectOk())
            {
                rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
                covReadRCTGRD(fd, s_h.xs, s_h.ys, s_h.zs, x_coord, y_coord, z_coord);
                readattrib(fd, rgrid);
                return (rgrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of RCTGRD object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "UNIGRD") == 0)
        {
            float x_min, y_min, z_min, x_max, y_max, z_max;
            covReadUNIGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs, &x_min, &x_max, &y_min, &y_max,
                          &z_min, &z_max);
            ugrid = new coDoUniformGrid(coObjInfo(Name), s_h.xs, s_h.ys, s_h.zs,
                                        x_min, x_max, y_min, y_max, z_min, z_max);
            objectList[infoIndex].obj = ugrid;
            if (ugrid->objectOk())
            {
                readattrib(fd, ugrid);
                return (ugrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of UNIGRID object failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRGRD") == 0)
        {
            covReadSizeSTRGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs);
            sgrid = new coDoStructuredGrid(coObjInfo(Name), s_h.xs, s_h.ys, s_h.zs);
            objectList[infoIndex].obj = sgrid;
            if (sgrid->objectOk())
            {
                sgrid->getAddresses(&x_coord, &y_coord, &z_coord);
                covReadSTRGRD(fd, s_h.xs, s_h.ys, s_h.zs, x_coord, y_coord, z_coord);
                readattrib(fd, sgrid);
                return (sgrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of STRGRD object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "USTSDT") == 0)
        {
            covReadSizeUSTSDT(fd, &n_elem);
            us3d = new coDoFloat(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = us3d;
            if (us3d->objectOk())
            {
                us3d->getAddress(&x_coord);
                covReadUSTSDT(fd, n_elem, x_coord);
                readattrib(fd, us3d);
                return (us3d);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTSDT object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "USTTDT") == 0)
        {
            coDoTensor::TensorType type;
            covReadSizeUSTTDT(fd, &n_elem, (int *)(void *)&type);
            coDoTensor *ut3d = new coDoTensor(coObjInfo(Name), n_elem, type);
            objectList[infoIndex].obj = ut3d;
            if (ut3d->objectOk())
            {
                ut3d->getAddress(&x_coord);
                covReadUSTTDT(fd, n_elem, type, x_coord);
                readattrib(fd, ut3d);
                return (ut3d);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTSDT object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "RGBADT") == 0)
        {
            covReadSizeRGBADT(fd, &n_elem);
            rgba = new coDoRGBA(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = rgba;
            if (rgba->objectOk())
            {
                rgba->getAddress((int **)(void *)(&x_coord));
                covReadRGBADT(fd, n_elem, (int *)x_coord);
                readattrib(fd, rgba);
                return (rgba);
            }
            else
            {
                Covise::sendError("ERROR: creation of RGBADT object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "USTVDT") == 0)
        {
            covReadSizeUSTVDT(fd, &n_elem);
            us3dv = new coDoVec3(coObjInfo(Name), n_elem);
            objectList[infoIndex].obj = us3dv;
            if (us3dv->objectOk())
            {
                us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
                covReadUSTVDT(fd, n_elem, x_coord, y_coord, z_coord);
                readattrib(fd, us3dv);
                return (us3dv);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTVDT object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRSDT") == 0)
        {
            covReadSizeSTRSDT(fd, &n_elem, &s_h.xs, &s_h.ys, &s_h.zs);
            us3d = new coDoFloat(coObjInfo(Name), s_h.xs * s_h.ys * s_h.zs);
            objectList[infoIndex].obj = us3d;
            if (us3d->objectOk())
            {
                us3d->getAddress(&x_coord);
                covReadSTRSDT(fd, n_elem, x_coord, s_h.xs, s_h.ys, s_h.zs);
                readattrib(fd, us3d);
                return (us3d);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTSDT (from STRSDT) object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRVDT") == 0)
        {
            covReadSizeSTRSDT(fd, &n_elem, &s_h.xs, &s_h.ys, &s_h.zs);
            us3dv = new coDoVec3(coObjInfo(Name), s_h.xs * s_h.ys * s_h.zs);
            objectList[infoIndex].obj = us3dv;
            if (us3dv->objectOk())
            {
                us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
                covReadSTRVDT(fd, n_elem, x_coord, y_coord, z_coord, s_h.xs, s_h.ys, s_h.zs);
                readattrib(fd, us3dv);
                return (us3dv);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTVDT (from STRVDT) object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "IMAGE") == 0)
        {
            int intPixelImageWidth = 0, intPixelImageHeight = 0, intPixelImageSize = 0,
                intPixelImageFormatId = 0, intPixelImageBufferLength = 0;
            char *strBuf;

            covReadSizeIMAGE(fd, &intPixelImageWidth, &intPixelImageHeight, &intPixelImageSize,
                             &intPixelImageFormatId, &intPixelImageBufferLength);

            //Creating of image name
            char *img_name = new char[strlen(Name) + 5];
            strcpy(img_name, Name);
            strcat(img_name, "_Img");

            //Creating of image
            coDoPixelImage *image = new coDoPixelImage(img_name, intPixelImageWidth,
                                                       intPixelImageHeight, intPixelImageSize, intPixelImageFormatId);
            objectList[infoIndex].obj = image;
            if (!(image->objectOk()))
            {
                Covise::sendInfo("Image object creation failed.");
                return NULL;
            }
            //Covise::sendInfo("Image object created.");

            strBuf = image->getPixels();
            covReadIMAGE(fd, intPixelImageWidth, intPixelImageHeight, intPixelImageSize,
                         intPixelImageFormatId, intPixelImageBufferLength, strBuf);

            readattrib(fd, image);
            delete[] img_name;
            return (image);
        }

        else if (strcmp(Data_Type, "TEXTUR") == 0)
        {
            int intNumberOfBorderPixels = 0, intNumberOfComponents = 0, intLevel = 0, intNumberOfVertices = 0,
                intNumberOfCoordinates = 0;

            char *img_name = new char[strlen(Name) + 5];
            strcpy(img_name, Name);
            strcat(img_name, "_Img");
            //cout << "img_name: " << img_name << endl;
            coDoPixelImage *imageS = (coDoPixelImage *)readData(fd, img_name);

            if (!(imageS->objectOk()))
            {
                Covise::sendInfo("Image object creation failed.");
                return NULL;
            }

            int *intVertexIndices;
            float **fltCoords;

            covReadSizeTEXTUR(fd, &intNumberOfBorderPixels, &intNumberOfComponents, &intLevel,
                              &intNumberOfCoordinates, &intNumberOfVertices);

            intVertexIndices = new int[intNumberOfVertices];
            fltCoords = new float *[2];
            for (i = 0; i < 2; i++)
            {
                fltCoords[i] = new float[intNumberOfCoordinates];
            }

            covReadTEXTUR(fd, intNumberOfBorderPixels, intNumberOfComponents, intLevel,
                          intNumberOfCoordinates, intNumberOfVertices, intVertexIndices,
                          fltCoords);

            //Creating of texture
            coDoTexture *tex = new coDoTexture(coObjInfo(Name),
                                               imageS,
                                               intNumberOfBorderPixels,
                                               intNumberOfComponents,
                                               intLevel,
                                               intNumberOfVertices,
                                               intVertexIndices,
                                               intNumberOfCoordinates,
                                               fltCoords);

            objectList[infoIndex].obj = tex;
            if (!tex)
            {
                Covise::sendError("ERROR: creation of TEXTURE object failed");
                return (NULL);
            }

            delete[] img_name;
            delete[] intVertexIndices;
            delete[] fltCoords[0];
            delete[] fltCoords[1];
            delete[] fltCoords;

            readattrib(fd, tex);
            return (tex);
        }
        else if (strcmp(Data_Type, "OCTREE") == 0)
        {
            int numCellLists, numMacroCellLists, numCellBBoxes, numGridBBoxes;
            int *cellLists = NULL, *macroCellLists = NULL;
            float *cellBBoxes = NULL, *gridBBoxes = NULL;
            int *fX = NULL, *fY = NULL, *fZ = NULL, *max_no_levels = NULL;

            covReadSizeOCTREE(fd, &numCellLists, &numMacroCellLists, &numCellBBoxes, &numGridBBoxes);

            coDoOctTree *octtree = new coDoOctTree(coObjInfo(Name), numCellLists, numMacroCellLists, numCellBBoxes, numGridBBoxes);
            objectList[infoIndex].obj = octtree;
            if (octtree->objectOk())
            {
                octtree->getAddresses(&cellLists, &macroCellLists, &cellBBoxes, &gridBBoxes, &fX, &fY, &fZ, &max_no_levels);

                covReadOCTREE(fd, &numCellLists, &numMacroCellLists, &numCellBBoxes, &numGridBBoxes, cellLists, macroCellLists, cellBBoxes, gridBBoxes, fX, fY, fZ, max_no_levels);
                return octtree;
            }
            else
            {
                Covise::sendError("ERROR: creation of OCTREE object failed");
                return (NULL);
            }
        }
        else
        {
            Covise::sendError("ERROR: Reading file '%s', file does not seem to be in COVISE format at all", grid_Path.c_str());
            return (NULL);
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct");
        return (NULL);
    }
    return (NULL);
}

void CoviseIO::skipData(int fd)
{
    char Data_Type[7];
    USG_HEADER usg_h;
    STR_HEADER s_h;
    //coDistributedObject **tmp_objs, *do1, *do2, *do3, *do4;
    int numsets, i; // t2, t3

    int length = 0;

    objectList.push_back(doInfo(NULL, ltell64(abs(fd))));

    if (::covReadDescription(fd, Data_Type) == -1 && !force)
    {
        Covise::sendError("ERROR: file is probably not a COVISE file - saving you from greater harm");
        Covise::sendError("ERROR: if it's an ASCII COVISE file then try RWCoviseASCII");
        return;
    }
    Data_Type[6] = '\0';

    if (strcmp(Data_Type, "OBJREF") == 0)
    {
        int objNum;
        ssize_t retval;
        retval = read(abs(fd), &objNum, sizeof(int));
        if (retval == -1)
            fprintf(stderr, "COV_READ_FLOAT failed");
    }
    if (strcmp(Data_Type, "SETELE") == 0)
    {
        ::covReadSetBegin(fd, &numsets);
        for (i = 0; i < numsets; i++)
        {
            skipData(fd);
        }

        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "INTARR") == 0)
    {
        int numDim;
        int numElem;
        covReadDimINTARR(fd, &numDim);

        int *sizes = new int[numDim];
        covReadSizeINTARR(fd, numDim, sizes, &numElem);
        //coDoIntArr *arr = new coDoIntArr(coObjInfo(Name),numDim,sizes);
        //covReadINTARR(fd, numDim, numElem, sizes, arr->getAddress() );
        lseek64(abs(fd), numDim * numElem * sizeof(int), SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "INTDT ") == 0)
    {
        int numElem;
        covReadSizeINTDT(fd, &numElem);
        lseek64(abs(fd), numElem * sizeof(int), SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "BYTEDT") == 0)
    {
        int numElem;
        covReadSizeBYTEDT(fd, &numElem);
        lseek64(abs(fd), numElem * sizeof(char), SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "GEOMET") == 0)
    {
        int ho1, ho2, ho3;
        covReadOldGeometryBegin(fd, &ho1, &ho2, &ho3);

        //t2 = 0;
        //t3 = 0;
        if (ho1)
        {
            skipData(fd);
        }
        if (ho2)
        {
            skipData(fd);
        }
        if (ho3)
        {
            skipData(fd);
        }

        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "GEOTEX") == 0)
    {
        // read flag values into int var
        int ho1, ho2, ho3, ho4;
        covReadGeometryBegin(fd, &ho1, &ho2, &ho3, &ho4);

        if (ho1)
        {
            skipData(fd);
        }
        if (ho2)
        {
            skipData(fd);
        }
        if (ho3)
        {
            skipData(fd);
        }
        if (ho4)
        {
            skipData(fd);
        }

        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "UNSGRD") == 0)
    {

        covReadSizeUNSGRD(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
        //mesh = new coDoUnstructuredGrid(coObjInfo(Name), usg_h.n_elem,usg_h.n_conn, usg_h.n_coord, 1);
        //covReadUNSGRD(fd, usg_h.n_elem,usg_h.n_conn, usg_h.n_coord,
        //      el, vl, tl, x_coord, y_coord, z_coord);

        length = 3 * sizeof(float) * usg_h.n_coord // coordinates
                 + sizeof(int) * usg_h.n_conn // vl
                 + 2 * sizeof(int) * usg_h.n_elem; // el + tl

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "POLYGN") == 0)
    {
        covReadSizeUNSGRD(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
        //pol = new coDoPolygons(coObjInfo(Name), usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
        //covReadPOLYGN(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
        //      x_coord, y_coord, z_coord );

        length = 3 * sizeof(float) * usg_h.n_coord // coordinates
                 + sizeof(int) * usg_h.n_conn // vl
                 + sizeof(int) * usg_h.n_elem; // el

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "POINTS") == 0)
    {
        covReadSizePOINTS(fd, &n_elem);
        //pts = new coDoPoints(coObjInfo(Name), n_elem);
        //   covReadPOINTS(fd, n_elem, x_coord, y_coord, z_coord);

        length = 3 * sizeof(float) * n_elem; // coordinates

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "SPHERE") == 0)
    {
        covReadSizeSPHERES(fd, &n_elem);
        //sph = new coDoSpheres(coObjInfo(Name), n_elem);
        //covReadSPHERES(fd, n_elem, x_coord, y_coord, z_coord, radius);

        length = 4 * sizeof(float) * n_elem; // coordinates + points

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "DOTEXT") == 0)
    {
        //char *data;
        covReadSizeDOTEXT(fd, &n_elem);
        //txt = new coDoText(coObjInfo(Name), n_elem);
        //covReadDOTEXT(fd, n_elem, data);

        length = sizeof(char) * n_elem;

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "LINES") == 0)
    {
        covReadSizeLINES(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
        //lin = new coDoLines(coObjInfo(Name), usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
        //covReadLINES(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
        //      x_coord, y_coord, z_coord);

        length = +3 * sizeof(float) * usg_h.n_conn // coordinates
                 + sizeof(int) * usg_h.n_elem // el
                 + sizeof(int) * usg_h.n_conn; // vl

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "TRIANG") == 0)
    {
        covReadSizeTRIANG(fd, &usg_h.n_elem, &usg_h.n_conn, &usg_h.n_coord);
        //tri = new coDoTriangleStrips(coObjInfo(Name), usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
        //covReadTRIANG(fd, usg_h.n_elem, el, usg_h.n_conn, vl, usg_h.n_coord,
        //      x_coord, y_coord, z_coord);

        length = +3 * sizeof(float) * usg_h.n_conn // coordinates
                 + sizeof(int) * usg_h.n_elem // el
                 + sizeof(int) * usg_h.n_conn; // vl

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "RCTGRD") == 0)
    {
        covReadSizeRCTGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs);
        //rgrid = new coDoRectilinearGrid(coObjInfo(Name),s_h.xs,s_h.ys, s_h.zs);
        //covReadRCTGRD(fd, s_h.xs,s_h.ys, s_h.zs, x_coord, y_coord, z_coord);

        length = +3 * sizeof(int) // number of nodes in each direction
                 + 3 * sizeof(float) * (s_h.xs + s_h.ys + s_h.zs); // coords in each direction

        lseek64(abs(fd), length, SEEK_CUR);

        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "UNIGRD") == 0)
    {
        //covReadUNIGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs, &x_min, &x_max, &y_min, &y_max,
        //   &z_min, &z_max);
        //ugrid = new coDoUniformGrid(coObjInfo(Name),s_h.xs,s_h.ys, s_h.zs,
        //   x_min, x_max, y_min, y_max, z_min, z_max);

        length = +3 * sizeof(int) // number of nodes in each direction
                 + 6 * sizeof(float); // dx, dy, dz

        lseek64(abs(fd), length, SEEK_CUR);

        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "STRGRD") == 0)
    {
        covReadSizeSTRGRD(fd, &s_h.xs, &s_h.ys, &s_h.zs);
        //sgrid = new coDoStructuredGrid(coObjInfo(Name), s_h.xs, s_h.ys, s_h.zs);
        //covReadSTRGRD(fd, s_h.xs, s_h.ys, s_h.zs, x_coord, y_coord, z_coord);

        // coordinates
        length = sizeof(float) * (s_h.xs * s_h.ys * s_h.zs);

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "USTSDT") == 0)
    {
        covReadSizeUSTSDT(fd, &n_elem);
        //us3d = new coDoFloat(coObjInfo(Name), n_elem);
        //covReadUSTSDT(fd, n_elem, x_coord );

        length = sizeof(float) * (n_elem); // scalar data ...

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "USTTDT") == 0)
    {
        coDoTensor::TensorType type;
        covReadSizeUSTTDT(fd, &n_elem, (int *)(void *)&type);
        //coDoTensor *ut3d = new coDoTensor(coObjInfo(Name), n_elem, type);

        //covReadUSTTDT(fd, n_elem, type, x_coord );
        switch (type)
        {
        case coDoTensor::S2D:
            length = 3 * sizeof(float) * (n_elem);
            break;
        case coDoTensor::F2D:
            length = 4 * sizeof(float) * (n_elem);
            break;
        case coDoTensor::S3D:
            length = 6 * sizeof(float) * (n_elem);
            break;
        case coDoTensor::F3D:
            length = 9 * sizeof(float) * (n_elem);
            break;
        default:
            break;
        }

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "RGBADT") == 0)
    {
        covReadSizeRGBADT(fd, &n_elem);
        //rgba = new coDoRGBA(coObjInfo(Name), n_elem);
        //covReadRGBADT(fd, n_elem, (int *)x_coord );

        length = 4 * sizeof(float) * (n_elem); // RGBA data ...

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "USTVDT") == 0)
    {
        covReadSizeUSTVDT(fd, &n_elem);
        //us3dv = new coDoVec3(coObjInfo(Name), n_elem);
        //covReadUSTVDT(fd, n_elem, x_coord, y_coord, z_coord );

        length = 3 * sizeof(float) * (n_elem); // vector data ...

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "STRSDT") == 0)
    {
        covReadSizeSTRSDT(fd, &n_elem, &s_h.xs, &s_h.ys, &s_h.zs);
        //us3d = new coDoFloat(coObjInfo(Name),s_h.xs*s_h.ys*s_h.zs);
        //covReadSTRSDT(fd, n_elem, x_coord, s_h.xs,s_h.ys,s_h.zs );

        length = sizeof(float) * (n_elem); // scalar data ...

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "STRVDT") == 0)
    {
        covReadSizeSTRSDT(fd, &n_elem, &s_h.xs, &s_h.ys, &s_h.zs);
        //us3dv = new coDoVec3(coObjInfo(Name),s_h.xs*s_h.ys*s_h.zs);
        //covReadSTRVDT(fd, n_elem, x_coord, y_coord, z_coord, s_h.xs, s_h.ys, s_h.zs );

        length = 3 * sizeof(float) * (n_elem); // vector data ...

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else if (strcmp(Data_Type, "IMAGE") == 0)
    {
        int intPixelImageWidth = 0, intPixelImageHeight = 0, intPixelImageSize = 0,
            intPixelImageFormatId = 0, intPixelImageBufferLength = 0;

        covReadSizeIMAGE(fd, &intPixelImageWidth, &intPixelImageHeight, &intPixelImageSize,
                         &intPixelImageFormatId, &intPixelImageBufferLength);

        //covReadIMAGE(fd, intPixelImageWidth, intPixelImageHeight, intPixelImageSize,
        //   intPixelImageFormatId, intPixelImageBufferLength, strBuf );

        length = intPixelImageSize;

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }

    else if (strcmp(Data_Type, "TEXTUR") == 0)
    {
        int intNumberOfBorderPixels = 0, intNumberOfComponents = 0, intLevel = 0, intNumberOfVertices = 0,
            intNumberOfCoordinates = 0;

        //coDoPixelImage *imageS = (coDoPixelImage *)readData(fd,img_name);
        skipData(fd);

        covReadSizeTEXTUR(fd, &intNumberOfBorderPixels, &intNumberOfComponents, &intLevel,
                          &intNumberOfCoordinates, &intNumberOfVertices);

        //intVertexIndices = new int[intNumberOfVertices];
        //fltCoords = new float*[2];
        //for (i=0; i<2; i++)
        //{
        //   fltCoords[i] = new float[intNumberOfCoordinates];
        //}

        //covReadTEXTUR(fd, intNumberOfBorderPixels, intNumberOfComponents, intLevel,
        //   intNumberOfCoordinates, intNumberOfVertices, intVertexIndices,
        //   fltCoords);

        // vertex indices
        length = sizeof(int) * (intNumberOfCoordinates)
                 + 2 * sizeof(float) * (intNumberOfCoordinates); // xy coords

        lseek64(abs(fd), length, SEEK_CUR);
        skipattrib(fd);
        return;
    }
    else
    {
        Covise::sendError("ERROR: Reading file '%s', file does not seem to be in COVISE format at all", grid_Path.c_str());
        return;
    }

    return;
}

void CoviseIO::skipattrib(int fd)
{
    int num = 0, size = 0;
    covReadNumAttributes(fd, &num, &size);

    lseek64(abs(fd), size, SEEK_CUR);
}
