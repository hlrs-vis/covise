/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                 (C)2000 VirCinity GmbH **
 ** Module IndexManifolds                                                    **
 **                                                                        **
 ** Author:                                                                **
 **                             Dirk Straka                                **
 **                          Christof Schwenzer                            **
 **                    VirCinity IT-Consulting GmbH                        **
 **                             Nobelstr. 35                               **
 **                            70569 Stuttgart                             **
 ** Date:  28.10.00  V1.0                                                  **
\**************************************************************************/

#include "IndexManifolds.h"
#include <util/coviseCompat.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor: This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

IndexManifolds::IndexManifolds(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Index Surface: Extract an Index Surface from a structured grid")
    , current_strip_length(0)
{
    maxSize[0] = maxSize[1] = maxSize[2] = 0;
    size[0] = size[1] = size[2] = 0;

    //
    // Modul-Parameter
    //

    const char *choiceVal[] = { "direction x", "direction y", "direction z" };
    DirChoice = addChoiceParam("DirChoice", "Select direction");
    DirChoice->setValue(3, choiceVal, 0);

    Index[0] = addIntSliderParam("xIndex", "Index of point in x-direction");
    Index[1] = addIntSliderParam("yIndex", "Index of point in y-direction");
    Index[2] = addIntSliderParam("zIndex", "Index of point in z-direction");
    for (int i = 0; i < 3; i++)
        Index[i]->setValue(0, 10, 0);

    GenStrips = addBooleanParam("Generate_strips", "generate strips");
    GenStrips->setValue(1);

    //
    // Input ports
    //
    inPortGrid = addInputPort("GridIn0", "StructuredGrid|RectilinearGrid|UniformGrid", "Grid");
    inPortData = addInputPort("DataIn0", "Float|Vec3|Int|RGBA", "Data");
    inPortData->setRequired(0); // NOT required!

    //
    // Output ports
    //
    outPortSurf = addOutputPort("GridOut0", "Polygons|TriangleStrips", "Layer surface");
    outPortSurfData = addOutputPort("DataOut0", "Float|Vec3|Int|RGBA", "Data mapped to layer surface");
    outPortSurfData->setDependencyPort(inPortData); // Port-Dependency

    outPortLine = addOutputPort("GridOut1", "Lines", "Line");
    outPortLineData = addOutputPort("DataOut1", "Float|Vec3|Int|RGBA", "Data mapped to line");
    outPortLineData->setDependencyPort(inPortData); // Port-Dependency

    outPortPoint = addOutputPort("GridOut2", "Points", "Point");
    outPortPointData = addOutputPort("DataOut2", "Float|Vec3|Int|RGBA", "Data mapped to point");
    outPortPointData->setDependencyPort(inPortData); // Port-Dependency
}

void IndexManifolds::setSliderBounds()
{
    for (int i = 0; i < 3; i++)
    {
        if (isPartOfSet())
        {
            if (maxSize[i] < size[i])
            {
                maxSize[i] = size[i];
                Index[i]->setValue(0, maxSize[i], Index[i]->getValue());
            }
        }
        else
        {
            maxSize[i] = size[i] - 1;
            Index[i]->setValue(0, size[i] - 1, Index[i]->getValue());
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Compute callback: Called when the module is executed
// ++++
// ++++  NEVER use input/output ports or distributed objects anywhere
// ++++        else than inside this function
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int IndexManifolds::compute(const char *)
{
    polygons = NULL;
    strips_out = NULL;
    GridObj = NULL;
    DataObj = NULL;

    outSurfName = NULL;
    outSurfDataName = NULL;

    gen_strips = GenStrips->getValue();
    //
    // retrieve data object from the required port
    //
    GridObj = inPortGrid->getCurrentObject();
    if (NULL != GridObj)
    {
        if (GridObj->isType("UNIGRD"))
        {
            static_cast<const coDoUniformGrid *>(GridObj)->getGridSize(&size[0], &size[1], &size[2]);
        }
        else if (GridObj->isType("RCTGRD"))
        {
            static_cast<const coDoRectilinearGrid *>(GridObj)->getGridSize(&size[0], &size[1], &size[2]);
        }
        else if (GridObj->isType("STRGRD"))
        {
            static_cast<const coDoStructuredGrid *>(GridObj)->getGridSize(&size[0], &size[1], &size[2]);
        }
        else
        {
            sendError("Invalid Type for Grid: %s", GridObj->getType());
            return FAIL;
        }
    }
    else
    {
        sendError("Grid: NULL");
        return FAIL;
    }
    //    cout << "sizei:" << sizei << "sizej:" << sizej << "sizek:" << sizek << endl;
    //
    // initialize Index to last computed value (or 0)
    //

    setSliderBounds();
    int dir = DirChoice->getValue();

    if (size[dir] < Index[dir]->getValue())
    { // skip this element of the set; therefore create empty objects
        if (gen_strips)
        {
            if ((strips_out = new coDoTriangleStrips(outPortSurf->getObjName(), 0, 0, 0)) != NULL)
                outPortSurf->setCurrentObject(strips_out);
            else
            {
                sendError("Error in creating strips object");
                return STOP_PIPELINE;
            }
        }
        else
        {
            if ((polygons = new coDoPolygons(outPortSurf->getObjName(), 0, 0, 0)) != NULL)
                outPortSurf->setCurrentObject(polygons);
            else
            {
                sendError("Error in creating polygon object");
                return STOP_PIPELINE;
            }
        }

        if (DataObj)
        {
            coDoAbstractData *outData = DataObj->cloneType(outPortSurf->getObjName(), 0);
            outPortSurfData->setCurrentObject(outData);
        }
        return CONTINUE_PIPELINE;
    }

    //
    // retrieve data object from the not required port
    //

    if (outPortSurfData->isConnected())
    {
        if (!inPortData->getCurrentObject())
        {
            sendError("Data: NULL");
            return FAIL;
        }

        DataObj = dynamic_cast<const coDoAbstractData *>(inPortData->getCurrentObject());
        if (NULL == DataObj)
        {
            sendError("Invalid Type for Data: %s", inPortData->getCurrentObject()->getType());
            // cerr << "Data: " << DataObj->getType() << endl;
            return FAIL;
        }
    }

    computeSurface();
    if (gen_strips)
    {
        if (strips_out == NULL)
            //error message already sent
            return FAIL;
    }
    else
    {
        if (polygons == NULL)
            //error message already sent
            return FAIL;
    }

    computeLine();

    computePoint();

    //
    // ... do whatever you like with in- or output objects,
    // BUT: do NOT delete them !!!!
    //

    //
    // tell the output port that this is his object
    //
    if (!gen_strips)
    {
        polygons->addAttribute("vertexOrder", "2");
        outPortSurf->setCurrentObject(polygons);
    }
    else
    {
        strips_out->addAttribute("vertexOrder", "2");
        outPortSurf->setCurrentObject(strips_out);
    }
    if (DataObj != NULL)
    {
        outPortSurfData->setCurrentObject(out_data);
    }

    return SUCCESS;
}

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Adpation needed
   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

void IndexManifolds::computeSurfaceSizes(int &pointcnt, int &cornercnt, int &polycnt)
{
    int fullstrips, rest;

    int dir1 = 1;
    int dir2 = 2;
    switch (DirChoice->getValue())
    {
    case 0: //i-direction
        dir1 = 1;
        dir2 = 2;
        break;
    case 1: //j-direction
        dir1 = 0;
        dir2 = 2;
        break;
    case 2: //k-direction
        dir1 = 0;
        dir2 = 1;
    }

    pointcnt = size[dir1] * size[dir2];
    if (!gen_strips)
    {
        cornercnt = (size[dir1] - 1) * (size[dir2] - 1) * 4;
        polycnt = (size[dir1] - 1) * (size[dir2] - 1);
    }
    else
    {
        fullstrips = (size[dir2] - 1) / 3;
        rest = (size[dir2] - 1) % 3;
        if (rest > 0)
        {
            polycnt = fullstrips + 1;
        }
        else
        {
            polycnt = fullstrips;
        }
        if (rest > 0)
        {
            cornercnt = 8 * fullstrips + 4 + (rest - 1) * 2;
        }
        else
        {
            cornercnt = 8 * fullstrips;
        }
        polycnt *= (size[dir1] - 1);
        cornercnt *= (size[dir1] - 1);
    }
}

void IndexManifolds::computeLine()
{
    if (!outPortLine->isConnected() && !outPortLineData->isConnected())
        return;

    if (!GridObj->isType("UNIGRD")
        && !GridObj->isType("RCTGRD")
        && !GridObj->isType("STRGRD"))
        return;

    coDoAbstractStructuredGrid *str = (coDoAbstractStructuredGrid *)GridObj;

    int sizes[3];
    str->getGridSize(&sizes[0], &sizes[1], &sizes[2]);
    bool outOfRange = false;
    for (int i = 0; i < 3; i++)
    {
        if (DirChoice->getValue() != i && Index[i]->getValue() >= sizes[i])
        {
            outOfRange = true;
            break;
        }
    }

    int dir = DirChoice->getValue();

    if (outPortLine->isConnected())
    {

        coDoLines *line = NULL;
        if (outOfRange)
        {
            line = new coDoLines(outPortLine->getObjName(), 0, 0, 0);
        }
        else
        {
            line = new coDoLines(outPortLine->getObjName(), sizes[dir], sizes[dir], 1);
            float *x, *y, *z;
            int *corners, *lines;
            line->getAddresses(&x, &y, &z, &corners, &lines);
            lines[0] = 0;
            for (int i = 0; i < sizes[dir]; i++)
            {
                corners[i] = i;
                switch (dir)
                {
                case 0:
                    str->getPointCoordinates(i, &x[i],
                                             Index[1]->getValue(), &y[i],
                                             Index[2]->getValue(), &z[i]);
                    break;
                case 1:
                    str->getPointCoordinates(Index[0]->getValue(), &x[i],
                                             i, &y[i],
                                             Index[2]->getValue(), &z[i]);
                    break;
                case 2:
                    str->getPointCoordinates(Index[0]->getValue(), &x[i],
                                             Index[1]->getValue(), &y[i],
                                             i, &z[i]);
                    break;
                }
            }
        }
        outPortLine->setCurrentObject(line);
    }

    if (DataObj && outPortLineData->isConnected())
    {
        coDoAbstractData *outData = DataObj->cloneType(outPortLineData->getObjName(), sizes[dir]);
        for (int i = 0; i < sizes[dir]; i++)
        {
            if (outOfRange)
            {
                outData->setNullValue(i);
            }
            else
            {
                int ind[3];
                for (int j = 0; j < 3; j++)
                {
                    ind[j] = Index[j]->getValue();
                }
                ind[dir] = i;

                outData->cloneValue(i, DataObj, coIndex(ind, sizes));
            }
        }
        outPortLineData->setCurrentObject(outData);
    }
}

void IndexManifolds::computePoint()
{
    if (!outPortPoint->isConnected() && !outPortPointData->isConnected())
        return;

    if (!GridObj->isType("UNIGRD")
        && !GridObj->isType("RCTGRD")
        && !GridObj->isType("STRGRD"))
        return;

    coDoAbstractStructuredGrid *str = (coDoAbstractStructuredGrid *)GridObj;

    int sizes[3];
    str->getGridSize(&sizes[0], &sizes[1], &sizes[2]);
    bool outOfRange = false;
    for (int i = 0; i < 3; i++)
    {
        if (Index[i]->getValue() >= sizes[i])
        {
            outOfRange = true;
            break;
        }
    }

    if (outPortPoint->isConnected())
    {

        coDoPoints *point = NULL;
        if (outOfRange)
        {
            point = new coDoPoints(outPortPoint->getObjName(), 0);
        }
        else
        {
            point = new coDoPoints(outPortPoint->getObjName(), 1);
            float *x, *y, *z;
            point->getAddresses(&x, &y, &z);
            str->getPointCoordinates(Index[0]->getValue(), x,
                                     Index[1]->getValue(), y,
                                     Index[2]->getValue(), z);
        }
        outPortPoint->setCurrentObject(point);
    }

    if (DataObj && outPortPointData->isConnected())
    {
        coDoAbstractData *outData = DataObj->cloneType(outPortPointData->getObjName(), 1);
        if (outOfRange)
        {
            outData->setNullValue(0);
        }
        else
        {
            outData->cloneValue(0,
                                DataObj,
                                coIndex(Index[0]->getValue(), Index[1]->getValue(), Index[2]->getValue(), sizes));
        }
        outPortLineData->setCurrentObject(outData);
    }
}

//test routine for debugging purposes
/*void showPolygons(coDoPolygons *polygons)
{
   int noOfPoints = polygons->getNumPoints();
   int noOfPolygons = polygons ->getNumPolygons();
   int noOfVertices = polygons -> getNumVertices();
   cerr << "noOfPoints: " << noOfPoints <<
      "noOfPolygons:" << noOfPolygons <<
      "noOfVertices:" << noOfVertices << endl;
   float *startX, *startY, *startZ;
   int *cornerList, *polygonList;
   polygons->getAddresses(&startX, &startY, &startZ, &cornerList, &polygonList);

int i;
cerr << "List of Coordinates" << endl;
for(i=0; i<noOfPoints; i++)
{
cerr << startX[i] << "		" << startY[i] << "		" << startZ[i] << endl;
}
cerr << "Vertices" << endl;
for(i=0;i< noOfVertices; i++)
{
cerr << cornerList[i] << endl;
}
cerr << "Polygons:" << endl;
for(i=0; i<noOfPolygons; i++)
{
cerr << polygonList[i] << endl;
}
}
void showStrips(coDoTriangleStrips *polygons)
{
int noOfPoints = polygons->getNumPoints();
int noOfPolygons = polygons ->getNumStrips();
int noOfVertices = polygons -> getNumVertices();
cerr << "noOfPoints: " << noOfPoints <<
"noOfPolygons:" << noOfPolygons <<
"noOfVertices:" << noOfVertices << endl;
float *startX, *startY, *startZ;
int *cornerList, *polygonList;
polygons->getAddresses(&startX, &startY, &startZ, &cornerList, &polygonList);

int i;
cerr << "List of Coordinates" << endl;
for(i=0; i<noOfPoints; i++)
{
cerr << startX[i] << "		" << startY[i] << "		" << startZ[i] << endl;
}
cerr << "Vertices" << endl;
for(i=0;i< noOfVertices; i++)
{
cerr << cornerList[i] << endl;
}
cerr << "Polygons:" << endl;
for(i=0; i<noOfPolygons; i++)
{
cerr << polygonList[i] << endl;
}
}
*/
coDoPolygons *IndexManifolds::computeSurface()
{

    int polycnt = 0;
    int pointcnt = 0;
    int cornercnt = 0;
    //
    // now we create an object for the output port:
    // get the name and make the Obj
    //
    outSurfName = outPortSurf->getObjName();
    if (NULL == outSurfName)
    {
        sendError("Object name not correct for 'Surface'");
        return NULL;
    }
    computeSurfaceSizes(pointcnt, cornercnt, polycnt);
    if (!gen_strips)
    {
        polygons = new coDoPolygons(outSurfName, pointcnt, cornercnt, polycnt);
    }
    else
    {
        strips_out = new coDoTriangleStrips(outSurfName, pointcnt, cornercnt, polycnt);
        num_strips = 0;
        current_corner = 0;
    }

    outSurfDataName = outPortSurfData->getObjName();
    if (NULL == outSurfName)
    {
        sendError("Object name not correct for 'Surface'");
        return NULL;
    }

    if (DataObj != NULL)
    {
        out_data = DataObj->cloneType(outSurfDataName, pointcnt);
    }
    //   cerr << "pointcnt: " << pointcnt << "corbercnt: " << cornercnt  << "Polycnt: " << polycnt << endl;

    if (GridObj->isType("UNIGRD"))
    {
        computeSurface(static_cast<const coDoUniformGrid *>(GridObj));
    }
    else if (GridObj->isType("RCTGRD"))
    {
        computeSurface(static_cast<const coDoRectilinearGrid *>(GridObj));
    }
    else if (GridObj->isType("STRGRD"))
    {
        computeSurface(static_cast<const coDoStructuredGrid *>(GridObj));
    }
    //if( gen_strips )
    //  showStrips(strips_out);
    return polygons;
}

//used by computeCoords for all structured grid types
void IndexManifolds::addCorners(
    int i,
    int j,
    int sizeA,
    int sizeB,
    int *corners)
{
    //Fill the corners list with the indices
    //into the coords list.
    //since there are one less rectangle
    //in each row or column respectively
    //than the dimensions the following
    //is done to the last but one line
    //or column
    int surfacePosition = Index[DirChoice->getValue()]->getValue();
    //   cerr << "sizeA=" << sizeA << "  sizeB=" << sizeB << endl;
    //   cerr << "i=" << i << "  j=" << j << endl;

    int sizes[3] = { 0, 0, 0 };
    const coDoAbstractStructuredGrid *grid = dynamic_cast<const coDoAbstractStructuredGrid *>(GridObj);
    if (grid)
    {
        grid->getGridSize(&sizes[0], &sizes[1], &sizes[2]);
    }
    if ((i < sizeA - 1) && (j < sizeB - 1))
    {
        if (!gen_strips)
        {
            corners[4 * (i * (sizeB - 1) + j)] = i * sizeB + j;
            corners[4 * (i * (sizeB - 1) + j) + 1] = i * sizeB + j + 1;
            corners[4 * (i * (sizeB - 1) + j) + 2] = (i + 1) * sizeB + j + 1;
            corners[4 * (i * (sizeB - 1) + j) + 3] = (i + 1) * sizeB + j;
        }
        else
        {
            if (!GridObj->isType("STRGRD"))
            {
                if (current_strip_length == 6 || j == 0)
                {
                    tl[num_strips++] = current_corner;
                    current_strip_length = 0;
                    corners[current_corner++] = (i + 1) * sizeB + j;
                    corners[current_corner++] = i * sizeB + j;
                    corners[current_corner++] = (i + 1) * sizeB + j + 1;
                    corners[current_corner++] = i * sizeB + j + 1;
                    current_strip_length += 2;
                }
                else
                {
                    corners[current_corner++] = (i + 1) * sizeB + j + 1;
                    corners[current_corner++] = i * sizeB + j + 1;
                    current_strip_length += 2;
                }
            }
            else
            {
                if (current_strip_length == 6 || j == 0)
                {
                    tl[num_strips++] = current_corner;
                    current_strip_length = 0;
                    corners[current_corner++] = (i + 1) * sizeB + j;
                    if (insert_possible(corners, i * sizeB + j))
                    {
                        corners[current_corner++] = i * sizeB + j;
                        current_strip_length++;
                    }
                    if (insert_possible(corners, (i + 1) * sizeB + j + 1))
                    {
                        corners[current_corner++] = (i + 1) * sizeB + j + 1;
                        current_strip_length++;
                    }
                    if (insert_possible(corners, i * sizeB + j + 1))
                    {
                        corners[current_corner++] = i * sizeB + j + 1;
                        current_strip_length++;
                    }
                }
                else
                {
                    if (insert_possible(corners, (i + 1) * sizeB + j + 1))
                    {
                        corners[current_corner++] = (i + 1) * sizeB + j + 1;
                        current_strip_length++;
                    }
                    if (insert_possible(corners, i * sizeB + j + 1))
                    {
                        corners[current_corner++] = i * sizeB + j + 1;
                        current_strip_length++;
                    }
                }
            }
        }
        //
        //  data
        //

        if (DataObj != NULL)
        {
            switch (DirChoice->getValue())
            {
            case 0:
                out_data->cloneValue(i * sizeB + j, DataObj, coIndex(surfacePosition, i, j, sizes));
                out_data->cloneValue(i * sizeB + j + 1, DataObj, coIndex(surfacePosition, i, j + 1, sizes));
                out_data->cloneValue((i + 1) * sizeB + j + 1, DataObj, coIndex(surfacePosition, i + 1, j + 1, sizes));
                out_data->cloneValue((i + 1) * sizeB + j, DataObj, coIndex(surfacePosition, i + 1, j, sizes));
                break;
            case 1:
                out_data->cloneValue(i * sizeB + j, DataObj, coIndex(i, surfacePosition, j, sizes));
                out_data->cloneValue((i + 1) * sizeB + j, DataObj, coIndex(i + 1, surfacePosition, j, sizes));
                out_data->cloneValue((i + 1) * sizeB + j, DataObj, coIndex(i + 1, surfacePosition, j, sizes));
                out_data->cloneValue(i * sizeB + j + 1, DataObj, coIndex(i, surfacePosition, j + 1, sizes));
                break;
            case 2:
                out_data->cloneValue(i * sizeB + j, DataObj, coIndex(i, j, surfacePosition, sizes));
                out_data->cloneValue(i * sizeB + j + 1, DataObj, coIndex(i, j + 1, surfacePosition, sizes));
                out_data->cloneValue((i + 1) * sizeB + j, DataObj, coIndex(i + 1, j, surfacePosition, sizes));
                out_data->cloneValue((i + 1) * sizeB + j + 1, DataObj, coIndex(i + 1, j + 1, surfacePosition, sizes));
                break;
            }
        }
        /*
            cerr << "corners[" << 4*(i*(sizeB-1)+j)      << "]=" << i*sizeB+j << endl;
            cerr << "corners[" << 4*(i*(sizeB-1)+j)+1    << "]=" << i*sizeB+j+1 << endl ;
            cerr << "corners[" << 4*(i*(sizeB-1)+j)+2    << "]=" << (i+1)*sizeB+j+1 << endl;
            cerr << "corners[" << 4*(i*(sizeB-1)+j)+3    << "]=" << (i+1)*sizeB+j << endl;
      */
    }
}

bool IndexManifolds::insert_possible(int *corners, int newcorner)
{
    for (int j = tl[num_strips - 1]; j < current_corner; j++)
    {
        if (x_coords[corners[j]] == x_coords[newcorner])
        {
            if (y_coords[corners[j]] == y_coords[newcorner])
            {
                if (z_coords[corners[j]] == z_coords[newcorner])
                {
                    return false;
                }
            }
        }
    }
    return true;
}

//compute the surface for rectilinera grids
void IndexManifolds::computeSurface(const coDoRectilinearGrid *grid)
{
    //   cerr << "computeSurface Rectilinear" << endl;

    int *corners, *polygonlist = 0; //FIXME: Correct usage of polygonlist
    int xSize, ySize, zSize;
    float *xGridStart, *yGridStart, *zGridStart; //arrays representing the grid
    float *xPolyStart, *yPolyStart, *zPolyStart; //array containing the coordinates of the polygons

    //xSize, ySize, zSize denote the number of points in each direction
    grid->getGridSize(&xSize, &ySize, &zSize);
    grid->getAddresses(&xGridStart, &yGridStart, &zGridStart);
    if (!gen_strips)
        polygons->getAddresses(&xPolyStart, &yPolyStart, &zPolyStart, &corners, &polygonlist);
    else
        strips_out->getAddresses(&xPolyStart, &yPolyStart, &zPolyStart, &corners, &tl);
    computeCoords(xGridStart[Index[DirChoice->getValue()]->getValue()], ySize, yGridStart, zSize, zGridStart,
                  xPolyStart, yPolyStart, zPolyStart, polygonlist, corners);
}

void IndexManifolds::computeSurface(const coDoStructuredGrid *grid)
{
    int *corners, *polygonlist = 0; //FIXME: Correct usage of polygonlist
    float *xPolyStart, *yPolyStart, *zPolyStart;
    if (!gen_strips)
        polygons->getAddresses(&xPolyStart, &yPolyStart, &zPolyStart, &corners, &polygonlist);
    else
        strips_out->getAddresses(&xPolyStart, &yPolyStart, &zPolyStart, &corners, &tl);

    int xSize, ySize, zSize;

    x_coords = xPolyStart;
    y_coords = yPolyStart;
    z_coords = zPolyStart;

    //xSize, ySize, zSize denote the number of points in each direction
    grid->getGridSize(&xSize, &ySize, &zSize);
    //   grid->getAddresses(&xGridStart, &yGridStart, &zGridStart);

    //   cerr << "computeSurface structured" << endl;
    switch (DirChoice->getValue())
    {
    case 0: //i-direction
        computeCoords(grid, Index[0]->getValue(), ySize, zSize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingX);
        break;
    case 1: //j-direction
        computeCoords(grid, Index[1]->getValue(), xSize, zSize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingY);
        break;
    case 2: //k-direction
        computeCoords(grid, Index[2]->getValue(), xSize, ySize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingZ);
    }
}

void IndexManifolds::computeCoords(
    const coDoStructuredGrid *grid,
    int surfaceIndex,
    int sizeA,
    int sizeB,
    float *xPolyStart,
    float *yPolyStart,
    float *zPolyStart,
    int *polygonlist,
    int *corners,
    missType missing)
{
    int i, j;
    for (i = 0; i < sizeA; i++)
    {
        for (j = 0; j < sizeB; j++)
        {
            switch (missing)
            {
            case missingX:
            {
                grid->getPointCoordinates(surfaceIndex, xPolyStart, i, yPolyStart, j, zPolyStart);
                break;
            }
            case missingY:
            {
                grid->getPointCoordinates(i, xPolyStart, surfaceIndex, yPolyStart, j, zPolyStart);
                break;
            }
            case missingZ:
            {
                grid->getPointCoordinates(i, xPolyStart, j, yPolyStart, surfaceIndex, zPolyStart);
                break;
            }
            }
            if (!gen_strips)
            {
                addCorners(i, j, sizeA, sizeB, corners);
            }
            xPolyStart++;
            yPolyStart++;
            zPolyStart++;
        }
    }

    if (!gen_strips)
    {
        for (i = 0; i < (sizeA - 1) * (sizeB - 1); i++)
        {
            polygonlist[i] = 4 * i;
        }
    }
    else
    {
        for (i = 0; i < sizeA; i++)
        {
            for (j = 0; j < sizeB; j++)
            {
                addCorners(i, j, sizeA, sizeB, corners);
            }
        }
        strips_out->setNumVertices(current_corner);
        strips_out->setNumStrips(num_strips);
    }
}

void IndexManifolds::computeSurface(const coDoUniformGrid *grid)
{
    int *corners;
    int *polygonlist = 0; //FIXME: Correct usage of polygonlist
    int xSize, ySize, zSize;
    float *xStart, *yStart, *zStart;
    xStart = yStart = zStart = NULL;
    float xDelta, yDelta, zDelta;
    float xMin, yMin, zMin;
    float xMax, yMax, zMax;
    grid->getGridSize(&xSize, &ySize, &zSize);
    grid->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
    grid->getDelta(&xDelta, &yDelta, &zDelta);
    if (!gen_strips)
        polygons->getAddresses(&xStart, &yStart, &zStart, &corners, &polygonlist);
    else
        strips_out->getAddresses(&xStart, &yStart, &zStart, &corners, &tl);
    // polygons ->getAddresses(&xStart, &yStart, &zStart, &corners, &polygonlist);

    switch (DirChoice->getValue())
    {
    case 0: //i-direction
        computeCoords(xMin + xDelta * Index[0]->getValue(), yMin, yDelta, ySize, zMin, zDelta, zSize, xStart, yStart, zStart, polygonlist, corners);
        break;
    case 1: //j-direction
        computeCoords(yMin + yDelta * Index[1]->getValue(), xMin, xDelta, xSize, zMin, zDelta, zSize, yStart, xStart, zStart, polygonlist, corners);
        break;
    case 2: //k-direction
        computeCoords(zMin + zDelta * Index[2]->getValue(), xMin, xDelta, xSize, yMin, yDelta, ySize, zStart, xStart, yStart, polygonlist, corners);
    }
}

//For rectilinear grids
void IndexManifolds::computeCoords(
    float surfaceCoord,
    int sizeA,
    float *gridA,
    int sizeB,
    float *gridB,
    float *surface,
    float *polyStartA,
    float *polyStartB,
    int *polygonlist,
    int *corners)
{
    int i, j;
    //   float *coordsCurr=coords;
    float *currGridA;
    float *currGridB;
    float *A = polyStartA;
    float *B = polyStartB;

    currGridA = gridA;
    for (i = 0; i < sizeA; i++)
    {
        currGridB = gridB;
        for (j = 0; j < sizeB; j++)
        {
            *surface++ = surfaceCoord;
            *A++ = *currGridA;
            *B++ = *currGridB;
            addCorners(i, j, sizeA, sizeB, corners);
            currGridB++;
        }
        currGridA++;
    }

    if (!gen_strips)
    {
        for (i = 0; i < (sizeA - 1) * (sizeB - 1); i++)
        {
            polygonlist[i] = 4 * i;
        }
    }
}

//for uniform grids
void IndexManifolds::computeCoords(
    float surfaceCoord,
    float minA,
    float deltaA,
    int sizeA,
    float minB,
    float deltaB,
    int sizeB,
    float *surface,
    float *A,
    float *B,
    int *polygonlist,
    int *corners)
{
    int i, j;
    float currA = minA;
    //   float *coordsCurr=coords;
    currA = minA;
    for (i = 0; i < sizeA; i++)
    {
        float currB = minB;
        for (j = 0; j < sizeB; j++)
        {
            *surface++ = surfaceCoord;
            *A++ = currA;
            *B++ = currB;
            addCorners(i, j, sizeA, sizeB, corners);
            currB += deltaB;
        }
        currA += deltaA;
    }
    if (!gen_strips)
    {
        for (i = 0; i < (sizeA - 1) * (sizeB - 1); i++)
        {
            polygonlist[i] = 4 * i;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void IndexManifolds::quit()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void IndexManifolds::postInst()
{
    DirChoice->show();
    Index[0]->show();
    Index[1]->show();
    Index[2]->show();
}

MODULE_MAIN(Filter, IndexManifolds)
