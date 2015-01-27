/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                 (C)2000 VirCinity GmbH **
 ** Module IndexSurface                                                    **
 **                                                                        **
 ** Author:                                                                **
 **                             Dirk Straka                                **
 **                          Christof Schwenzer                            **
 **                    VirCinity IT-Consulting GmbH                        **
 **                             Nobelstr. 35                               **
 **                            70569 Stuttgart                             **
 ** Date:  28.10.00  V1.0                                                  **
\**************************************************************************/

#include "IndexSurface.h"
#include <util/coviseCompat.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor: This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

IndexSurface::IndexSurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Index Surface: Extract an Index Surface from a structured grid")
    , lasti(0)
    , lastj(0)
    , lastk(0)
{
    const char *choiceVal[] = { "direction x", "direction y", "direction z" };

    //
    // Modul-Parameter
    //

    DirChoice = addChoiceParam("DirChoice", "Select direction");
    DirChoice->setValue(3, choiceVal, 0);

    Index = addIntSliderParam("Index", "Index of selected direction");
    Index->setValue(0, 10, 0);

    GenStrips = addBooleanParam("Generate_strips", "generate strips");
    GenStrips->setValue(1);

    //
    // Output ports (have to be first because of buggy setDependant())
    //
    outPortSurf = addOutputPort("Surface", "Polygons|TriangleStrips", "Surface");
    outPortSurfData = addOutputPort("Surface Data", "Float|Vec3", "Data");

    //
    // Input ports
    //
    inPortGrid = addInputPort("Grid", "StructuredGrid|RectilinearGrid|UniformGrid", "Grid");
    inPortData = addInputPort("Data", "Float|Vec3", "Data");
    inPortData->setRequired(0); // NOT required!
    outPortSurfData->setDependency(inPortData); // Port-Dependency

    size = -1; // size is always set to the maximum size appearing, so start with -1
    //
    // and that's all ... no init() or anything else ...
    // that's done in the lib.
    //
}

int IndexSurface::set_index_Value(const bool oldval)
{
    int last = 0;

    switch (DirChoice->getValue())
    {
    // max of slider is max of all sizes
    case 0:
        if (!isPartOfSet() || sizei > size)
            size = sizei;
        last = lasti = oldval ? Index->getValue() : lasti;
        // skip this grid
        if (isPartOfSet() && last >= sizei)
            return 0;
        break;
    case 1:
        if (!isPartOfSet() || sizej > size)
            size = sizej;
        last = lastj = oldval ? Index->getValue() : lastj;
        // skip this grid
        if (isPartOfSet() && last >= sizej)
            return 0;
        break;
    case 2:
        if (!isPartOfSet() || sizek > size)
            size = sizek;
        last = lastk = oldval ? Index->getValue() : lastk;
        // skip this grid
        if (isPartOfSet() && last >= sizek)
            return 0;
        break;
    default:
        lasti = 0, lastj = 0, lastk = 0;
        sizei = 0, sizej = 0, sizek = 0;
        fprintf(stderr, "IndexSurface: invalid value for DirChoice: %d\n",
                DirChoice->getValue());
        return 0;
    }
    Index->setValue(0, size - 1, last);
    return 1;
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

int IndexSurface::compute(const char *)
{
    GridObj = NULL;
    DataObj = NULL;
    polygons = NULL;
    strips_out = NULL;

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
            static_cast<coDoUniformGrid *>(GridObj)->getGridSize(&sizei, &sizej, &sizek);
        }
        else if (GridObj->isType("RCTGRD"))
        {
            static_cast<coDoRectilinearGrid *>(GridObj)->getGridSize(&sizei, &sizej, &sizek);
        }
        else if (GridObj->isType("STRGRD"))
        {
            static_cast<coDoStructuredGrid *>(GridObj)->getGridSize(&sizei, &sizej, &sizek);
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
    if (!set_index_Value(true))
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

        if (inPortData->getCurrentObject() != NULL)
        {
            if ((inPortData->getCurrentObject())->isType("STRSDT"))
            {
                if ((obj_sdata_out = new coDoFloat(outPortSurfData->getObjName(), 0)) != NULL)
                    outPortSurfData->setCurrentObject(obj_sdata_out);
                else
                {
                    sendError("Error in creating data object");
                    return STOP_PIPELINE;
                }
            }
            else
            {
                if ((obj_vdata_out = new coDoVec3(outPortSurfData->getObjName(), 0)) != NULL)
                    outPortSurfData->setCurrentObject(obj_vdata_out);
                else
                {
                    sendError("Error in creating data object");
                    return STOP_PIPELINE;
                }
            }
        }
        return CONTINUE_PIPELINE;
    }

    //
    // retrieve data object from the not required port
    //

    if (outPortSurfData->isConnected())
    {
        DataObj = inPortData->getCurrentObject();
        if (NULL != DataObj)
        {
            if (!DataObj->isType("STRSDT") && !DataObj->isType("STRVDT"))
            {
                sendError("Invalid Type for Data: %s", DataObj->getType());
                return FAIL;
            }
            // cerr << "Data: " << DataObj->getType() << endl;
        }
        else
        {
            sendError("Data: NULL");
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
        {
            //error message already sent
            return FAIL;
        }
    }

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
        if (DataObj->isType("STRSDT"))
            outPortSurfData->setCurrentObject(obj_sdata_out);
        else if (DataObj->isType("STRVDT"))
            outPortSurfData->setCurrentObject(obj_vdata_out);
    }

    return SUCCESS;
}

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Adpation needed
   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

void IndexSurface::computeSizes(int &pointcnt, int &cornercnt, int &polycnt)
{
    int fullstrips, rest;
    switch (DirChoice->getValue())
    {
    case 0: // direction i
        pointcnt = sizej * sizek;
        if (!gen_strips)
        {
            cornercnt = (sizej - 1) * (sizek - 1) * 4;
            polycnt = (sizej - 1) * (sizek - 1);
        }
        else
        {
            fullstrips = (sizek - 1) / 3;
            rest = (sizek - 1) % 3;
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
            polycnt *= (sizej - 1);
            cornercnt *= (sizej - 1);
        }
        break;
    case 1: // direction j
        pointcnt = sizei * sizek;
        if (!gen_strips)
        {
            cornercnt = (sizei - 1) * (sizek - 1) * 4;
            polycnt = (sizei - 1) * (sizek - 1);
        }
        else
        {
            fullstrips = (sizek - 1) / 3;
            rest = (sizek - 1) % 3;
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
            polycnt *= (sizei - 1);
            cornercnt *= (sizei - 1);
        }
        break;
    case 2: // direction k
        pointcnt = sizei * sizej;
        if (!gen_strips)
        {
            cornercnt = (sizei - 1) * (sizej - 1) * 4;
            polycnt = (sizei - 1) * (sizej - 1);
        }
        else
        {
            fullstrips = (sizej - 1) / 3;
            rest = (sizej - 1) % 3;
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
            polycnt *= (sizei - 1);
            cornercnt *= (sizei - 1);
        }
        break;
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
coDoPolygons *IndexSurface::computeSurface()
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
    computeSizes(pointcnt, cornercnt, polycnt);
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
        if (DataObj->isType("STRSDT"))
        {
            obj_sdata_out = new coDoFloat(outSurfDataName, pointcnt);
            obj_sdata_out->getAddress(&sdata);
        }

        else if (DataObj->isType("STRVDT"))
        {
            obj_vdata_out = new coDoVec3(outSurfDataName, pointcnt);
            obj_vdata_out->getAddresses(&xdata, &ydata, &zdata);
        }
    }
    //   cerr << "pointcnt: " << pointcnt << "corbercnt: " << cornercnt  << "Polycnt: " << polycnt << endl;

    if (GridObj->isType("UNIGRD"))
    {
        computeSurface(static_cast<coDoUniformGrid *>(GridObj));
    }
    else if (GridObj->isType("RCTGRD"))
    {
        computeSurface(static_cast<coDoRectilinearGrid *>(GridObj));
    }
    else if (GridObj->isType("STRGRD"))
    {
        computeSurface(static_cast<coDoStructuredGrid *>(GridObj));
    }
    //if( gen_strips )
    //  showStrips(strips_out);
    return polygons;
}

//used by computeCoords for all structured grid types
void IndexSurface::addCorners(
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
    int surfacePosition = Index->getValue();
    //   cerr << "sizeA=" << sizeA << "  sizeB=" << sizeB << endl;
    //   cerr << "i=" << i << "  j=" << j << endl;
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

        float coord[3];
        if (DataObj != NULL)
        {
            switch (DirChoice->getValue())
            {
            case 0:
                if (DataObj->isType("STRSDT"))
                {
                    ((coDoFloat *)DataObj)->getPointValue(surfacePosition, i, j, &sdata[i * sizeB + j]);
                    ((coDoFloat *)DataObj)->getPointValue(surfacePosition, i, j + 1, &sdata[i * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(surfacePosition, i + 1, j + 1, &sdata[(i + 1) * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(surfacePosition, i + 1, j, &sdata[(i + 1) * sizeB + j]);
                }
                else if (DataObj->isType("STRVDT"))
                {
                    ((coDoVec3 *)DataObj)->getPointValue(surfacePosition, i, j, &coord[0]);
                    xdata[i * sizeB + j] = coord[0];
                    ydata[i * sizeB + j] = coord[1];
                    zdata[i * sizeB + j] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(surfacePosition, i, j + 1, &coord[0]);
                    xdata[i * sizeB + j + 1] = coord[0];
                    ydata[i * sizeB + j + 1] = coord[1];
                    zdata[i * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(surfacePosition, i + 1, j + 1, &coord[0]);
                    xdata[(i + 1) * sizeB + j + 1] = coord[0];
                    ydata[(i + 1) * sizeB + j + 1] = coord[1];
                    zdata[(i + 1) * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(surfacePosition, i + 1, j, &coord[0]);
                    xdata[(i + 1) * sizeB + j] = coord[0];
                    ydata[(i + 1) * sizeB + j] = coord[1];
                    zdata[(i + 1) * sizeB + j] = coord[2];
                }
                break;
            case 1:
                if (DataObj->isType("STRSDT"))
                {
                    ((coDoFloat *)DataObj)->getPointValue(i, surfacePosition, j, &sdata[i * sizeB + j]);
                    ((coDoFloat *)DataObj)->getPointValue(i, surfacePosition, j + 1, &sdata[i * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(i + 1, surfacePosition, j + 1, &sdata[(i + 1) * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(i + 1, surfacePosition, j, &sdata[(i + 1) * sizeB + j]);
                }
                else if (DataObj->isType("STRVDT"))
                {
                    ((coDoVec3 *)DataObj)->getPointValue(i, surfacePosition, j, &coord[0]);
                    xdata[i * sizeB + j] = coord[0];
                    ydata[i * sizeB + j] = coord[1];
                    zdata[i * sizeB + j] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i, surfacePosition, j + 1, &coord[0]);
                    xdata[i * sizeB + j + 1] = coord[0];
                    ydata[i * sizeB + j + 1] = coord[1];
                    zdata[i * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i + 1, surfacePosition, j + 1, &coord[0]);
                    xdata[(i + 1) * sizeB + j + 1] = coord[0];
                    ydata[(i + 1) * sizeB + j + 1] = coord[1];
                    zdata[(i + 1) * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i + 1, surfacePosition, j, &coord[0]);
                    xdata[(i + 1) * sizeB + j] = coord[0];
                    ydata[(i + 1) * sizeB + j] = coord[1];
                    zdata[(i + 1) * sizeB + j] = coord[2];
                }
                break;
            case 2:
                if (DataObj->isType("STRSDT"))
                {
                    ((coDoFloat *)DataObj)->getPointValue(i, j, surfacePosition, &sdata[i * sizeB + j]);
                    ((coDoFloat *)DataObj)->getPointValue(i, j + 1, surfacePosition, &sdata[i * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(i + 1, j + 1, surfacePosition, &sdata[(i + 1) * sizeB + j + 1]);
                    ((coDoFloat *)DataObj)->getPointValue(i + 1, j, surfacePosition, &sdata[(i + 1) * sizeB + j]);
                    break;
                }
                else if (DataObj->isType("STRVDT"))
                {
                    ((coDoVec3 *)DataObj)->getPointValue(i, j, surfacePosition, &coord[0]);
                    xdata[i * sizeB + j] = coord[0];
                    ydata[i * sizeB + j] = coord[1];
                    zdata[i * sizeB + j] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i, j + 1, surfacePosition, &coord[0]);
                    xdata[i * sizeB + j + 1] = coord[0];
                    ydata[i * sizeB + j + 1] = coord[1];
                    zdata[i * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i + 1, j + 1, surfacePosition, &coord[0]);
                    xdata[(i + 1) * sizeB + j + 1] = coord[0];
                    ydata[(i + 1) * sizeB + j + 1] = coord[1];
                    zdata[(i + 1) * sizeB + j + 1] = coord[2];
                    ((coDoVec3 *)DataObj)->getPointValue(i + 1, j, surfacePosition, &coord[0]);
                    xdata[(i + 1) * sizeB + j] = coord[0];
                    ydata[(i + 1) * sizeB + j] = coord[1];
                    zdata[(i + 1) * sizeB + j] = coord[2];
                }
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

bool IndexSurface::insert_possible(int *corners, int newcorner)
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
void IndexSurface::computeSurface(coDoRectilinearGrid *grid)
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
    switch (DirChoice->getValue())
    {
    case 0: //i-direction
        computeCoords(xGridStart[Index->getValue()], ySize, yGridStart, zSize, zGridStart,
                      xPolyStart, yPolyStart, zPolyStart, polygonlist, corners);
        break;
    case 1: //j-direction
        computeCoords(yGridStart[Index->getValue()], xSize, xGridStart, zSize, zGridStart,
                      yPolyStart, xPolyStart, zPolyStart, polygonlist, corners);
        break;
    case 2: //k-direction
        computeCoords(zGridStart[Index->getValue()], xSize, xGridStart, ySize, yGridStart,
                      zPolyStart, xPolyStart, yPolyStart, polygonlist, corners);
        break;
    }
}

void IndexSurface::computeSurface(coDoStructuredGrid *grid)
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
        computeCoords(grid, Index->getValue(), ySize, zSize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingX);
        break;
    case 1: //j-direction
        computeCoords(grid, Index->getValue(), xSize, zSize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingY);
        break;
    case 2: //k-direction
        computeCoords(grid, Index->getValue(), xSize, ySize, xPolyStart, yPolyStart, zPolyStart, polygonlist, corners, missingZ);
    }
}

void IndexSurface::computeCoords(
    coDoStructuredGrid *grid,
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

void IndexSurface::computeSurface(coDoUniformGrid *grid)
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
        computeCoords(xMin + xDelta * Index->getValue(), yMin, yDelta, ySize, zMin, zDelta, zSize, xStart, yStart, zStart, polygonlist, corners);
        break;
    case 1: //j-direction
        computeCoords(yMin + yDelta * Index->getValue(), xMin, xDelta, xSize, zMin, zDelta, zSize, yStart, xStart, zStart, polygonlist, corners);
        break;
    case 2: //k-direction
        computeCoords(zMin + zDelta * Index->getValue(), xMin, xDelta, xSize, yMin, yDelta, ySize, zStart, xStart, yStart, polygonlist, corners);
    }
}

//For rectilinear grids
void IndexSurface::computeCoords(
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
void IndexSurface::computeCoords(
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
// +++++++++++++

// +++++++++++++
// ++++  Parameter callback: This one is called whenever an immediate
// ++++                      mode parameter is changed, but NOT for
// ++++                      non-immediate ports
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void IndexSurface::param(const char *name)
{
    if (!in_map_loading() && !strcmp(name, DirChoice->getName()))
    {
        set_index_Value(false);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void IndexSurface::quit()
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

void IndexSurface::postInst()
{
    DirChoice->show();
    Index->show();
}

MODULE_MAIN(Obsolete, IndexSurface)
