/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                   (C)2010 Stellba Hydro GmbH & Co. KG  **
**                                                                        **
** Description:  PolyBoundaryLines                                        **                                                                        **
**                                                                        **
** extracts boundary lines of polygons                                    **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

//#include <iostream>
//#include <fstream>
#include <vector>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <alg/coFixUsg.h>
#include "PolyBoundaryLines.h"

// for sorting
typedef std::pair<int, int> int_int_pair_;
typedef std::vector<int_int_pair_> int_int_vec_;
bool comparator(const int_int_pair_ &l, const int_int_pair_ &r)
{
    return l.first < r.first;
}

PolyBoundaryLines::PolyBoundaryLines(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Extract Boundary Lines of Polygons")
{

    // the input ports
    p_polygons = addInputPort("GridIn0", "Polygons|TriangleStrips", "the polygons");
    p_dataIn = addInputPort("DataIn0", "Float", "scalar data input");
    p_dataIn->setRequired(false);

    // the output ports
    p_boundaryLines = addOutputPort("GridOut0", "Lines", "boundary lines");
    p_dataOut = addOutputPort("DataOut0", "Float", "scalar data output");

    // parameters
    // none so far ;-)

    setCopyNonSetAttributes(0); // get rid of of COLOR-attribute
}

PolyBoundaryLines::~PolyBoundaryLines()
{
}

void PolyBoundaryLines::param(const char *paramName, bool inMapLoading)
{
    (void)paramName;
    (void)inMapLoading;
}

int PolyBoundaryLines::compute(const char *)
{
    // get input object
    int haveStrips = -1;

    const coDistributedObject *obj = NULL;
    const coDistributedObject *dataObj = NULL;

    coDoPolygons *polygons;
    const coDoFloat *scalDataIn = NULL;

    coDistributedObject *fixed;
    coDistributedObject *fixedData;
    coDoTriangleStrips *strips;
    coDoTriangleStrips *stripsIn;

    int numPoints;
    int numVertices;
    int numPrimitives;

    float *x, *y, *z; // coordinates
    int *cl, *pl, *sl; // corner list, polygon list, strips list
    float *dataVals; // data values

    // polygons
    obj = p_polygons->getCurrentObject();
    // scalar data
    dataObj = p_dataIn->getCurrentObject();
    bool haveData;
    if ((dataObj != NULL) && (dataObj->isType(USTSDT)))
    {
        haveData = true;
    }
    else
    {
        haveData = false;
    }

    // check for UPSTREAMCUT attribute
    // only look at polygons with z-coordinate > zMin
    bool have_filter = false;
    float filter_zMin;
    const char *filterAttr = obj->getAttribute("POLYBOUNDARYFILTER");
    if ((filterAttr != NULL))
    {
        have_filter = true;

        // parse POLYBOUNDARYFILTER attribute
        if (sscanf(filterAttr, "%f", &filter_zMin) != 1)
        {
            sendError("PolyBoundaryLines: wrong number of arguments parsed from POLYBOUNDARYFILTER-attribute");
        }
    }

    // FixUsg allows us to use this even if we have none-shared vertices (e.g.for CuttingSurface output)
    // use default FixUsg parameters
    coFixUsg fct(50, 0.00000, false); /* false means optimize for speed */

    int numDataObjs = 0;
    if (haveData)
    {
        numDataObjs = 1;
    }

    if (obj->isType("POLYGN"))
    {
        haveStrips = 0;

        // these are just dummies (we do not treat data here)
        coDistributedObject **fixedDataObj = new coDistributedObject *[1];
        char **outNames = new char *[1];

        int num_red = 0;
        char *fixedName = new char[strlen(p_boundaryLines->getObjName()) + 100];
        outNames[0] = new char[strlen(p_dataOut->getObjName()) + 100];
        sprintf(fixedName, "%s_fixed", p_boundaryLines->getObjName());
        sprintf(outNames[0], "%s_fixed", p_dataOut->getObjName());

        if ((num_red = fct.fixUsg(obj, // const coDistributedObject *
                                  &fixed, // coDistributedObject *
                                  fixedName, // char *
                                  numDataObjs, // int
                                  &dataObj, // const coDistributedObject *
                                  &fixedData, // coDistributedObject *
                                  (const char **)outNames)) // char **
            == coFixUsg::FIX_ERROR)
        {
            delete[] fixedName;
            Covise::sendError("Failure in fixUsg");
            return STOP_PIPELINE;
        }
        //if (num_red>0)
        //sendInfo("removed / merged %d coordinates\n", num_red);

        delete[] fixedName;
        polygons = (coDoPolygons *)fixed;

        if (haveData)
        {
            scalDataIn = dynamic_cast<const coDoFloat *>(fixedData);
            scalDataIn->getAddress(&dataVals);
        }

        numPoints = polygons->getNumPoints();
        numVertices = polygons->getNumVertices();
        numPrimitives = polygons->getNumPolygons();
        polygons->getAddresses(&x, &y, &z, &cl, &pl);
    }
    else if (obj->isType("TRIANG"))
    {
        haveStrips = 1;
        stripsIn = (coDoTriangleStrips *)obj;

        // these are just dummies (we do not treat data here)
        coDistributedObject **fixedDataObj = new coDistributedObject *[1];
        char **outNames = new char *[1];

        int num_red = 0;
        char *fixedName = new char[strlen(p_boundaryLines->getObjName()) + 100];
        outNames[0] = new char[strlen(p_dataOut->getObjName()) + 100];
        sprintf(fixedName, "%s_fixed", p_boundaryLines->getObjName());
        sprintf(outNames[0], "%s_fixed", p_dataOut->getObjName());

        if ((num_red = fct.fixUsg(obj,
                                  &fixed,
                                  fixedName,
                                  numDataObjs,
                                  &dataObj,
                                  &fixedData,
                                  (const char **)outNames))
            == coFixUsg::FIX_ERROR)
        {
            delete[] fixedName;
            Covise::sendError("Failure in fixUsg");
            return STOP_PIPELINE;
        }

        delete[] fixedName;
        strips = (coDoTriangleStrips *)fixed;

        if (haveData)
        {
            scalDataIn = dynamic_cast<const coDoFloat *>(fixedData);
            scalDataIn->getAddress(&dataVals);
        }

        numPoints = strips->getNumPoints();
        numVertices = strips->getNumVertices();
        numPrimitives = strips->getNumStrips();
        strips->getAddresses(&x, &y, &z, &cl, &sl);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_polygons->getName());
        return (FAIL);
    }

    // extract edges from polygons / triangle strips

    int_int_vec_ edges;
    bool useThisFace;

    if (haveStrips) // triangle strips
    {
        int j;
        int nnodes = 0;
        int node[3];

        for (int i = 0; i < numPrimitives; i++)
        {
            if (i < (numPrimitives - 1))
            {
                nnodes = sl[i + 1] - sl[i];
            }
            else
            {
                nnodes = numVertices - sl[i];
            }

            // now add the edges ...
            for (j = 0; j < nnodes - 2; j++)
            {
                useThisFace = true;
                node[0] = cl[sl[i] + j];
                node[1] = cl[sl[i] + j + 1];
                node[2] = cl[sl[i] + j + 2];
                if (have_filter)
                {
                    useThisFace = false;

                    if ((x[node[0]] > filter_zMin) && (x[node[1]] > filter_zMin) && (x[node[2]] > filter_zMin))
                    {
                        useThisFace = true;
                    }
                }
                if (useThisFace)
                {
                    edges.push_back(std::make_pair(node[0], node[1]));
                    edges.push_back(std::make_pair(node[1], node[2]));
                    edges.push_back(std::make_pair(node[2], node[0]));
                }
            }
        }
    }
    else // polygons
    {
        int nnodes = 0;
        int node[4]; // for now, we support only triangles and quads, no polyhedron cells ...

        for (int i = 0; i < numPrimitives; i++)
        {
            if (i < (numPrimitives - 1))
            {
                nnodes = pl[i + 1] - pl[i];
            }
            else
            {
                nnodes = numVertices - pl[i];
            }

            // now add the edges ...
            useThisFace = true;

            for (int j = 0; j < nnodes; j++)
            {
                node[j] = cl[pl[i] + j];
            }

            if (have_filter)
            {
                useThisFace = true;

                for (int j = 0; j < nnodes; j++)
                {
                    if (z[node[j]] < filter_zMin)
                    {
                        useThisFace = false;
                        break;
                    }
                }
            }
            if (useThisFace)
            {
                for (int j = 0; j < nnodes - 1; j++)
                {
                    edges.push_back(std::make_pair(node[j], node[j + 1]));
                }
                edges.push_back(std::make_pair(node[nnodes - 1], node[0]));
            }
        }
    }

    // sort node pairs in edges list so that first node number is always smaller
    int edgeNode1, edgeNode2;
    for (int i = 0; i < edges.size(); i++)
    {
        edgeNode1 = edges[i].first;
        edgeNode2 = edges[i].second;
        if (edgeNode2 < edgeNode1)
        {
            edges[i].first = edgeNode2;
            edges[i].second = edgeNode1;
        }
    }

    //fprintf(stderr,"number of polygons: %d\n", numPrimitives);
    //fprintf(stderr,"number of edges: %d\n", (int)edges.size());

    // now sort the edges list
    sort(edges.begin(), edges.end(), comparator);

    // get edges that are unique
    int *edgeIndex = new int[numPoints + 1];
    int pos = 0;
    int maxNumEdges = 0;
    edgeIndex[0] = 0;
    int edgesSize = edges.size();
    for (int nodenr = 0; nodenr < numPoints; nodenr++)
    {
        while ((pos < edgesSize) && edges[pos].first == nodenr)
        {
            pos++;
        }
        edgeIndex[nodenr + 1] = pos;
        if ((edgeIndex[nodenr + 1] - edgeIndex[nodenr]) > maxNumEdges)
        {
            maxNumEdges = edgeIndex[nodenr + 1] - edgeIndex[nodenr];
        }
    }

    int numEdges = 0;
    int *secondNodes = new int[maxNumEdges];
    int nOuterEdges = 0;

    int *usedNode = new int[numPoints]; // mark outer nodes (nodes on boundary edges)
    memset(usedNode, -1, numPoints * sizeof(int));

    int_int_vec_ outerEdges;

    for (int nodenr = 0; nodenr < numPoints; nodenr++)
    {
        numEdges = edgeIndex[nodenr + 1] - edgeIndex[nodenr]; // yes, this is correct!

        for (int i = 0; i < numEdges; i++)
        {
            secondNodes[i] = edges[edgeIndex[nodenr] + i].second;
        }

        std::sort(secondNodes, secondNodes + numEdges);

        for (int i = 0; i < numEdges; i++)
        {
            pos = 0;

            if ((i == numEdges - 1) || (secondNodes[i + pos + 1] != secondNodes[i]))
            {
                // found a unique edge!!
                usedNode[nodenr]++;
                usedNode[secondNodes[i]]++;
                outerEdges.push_back(std::make_pair(nodenr, secondNodes[i]));
                nOuterEdges++;
            }
            else
            {
                while (secondNodes[i + pos + 1] == secondNodes[i])
                {
                    pos++;
                    if ((i + pos + 1) > 3)
                        break;
                }
                i += pos;
            }
        }
    }

    delete[] edgeIndex;
    delete[] secondNodes;

    edges.clear();

    //sendInfo("PolyBoundaryLines: found %d boundary edges", nOuterEdges);

    // construct coDoLines output object

    // for renumbering ...
    pos = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (usedNode[i] > -1)
        {
            usedNode[i] = pos;
            pos++;
        }
    }

    int outLinesNumNodes = pos;

    float *xOut;
    float *yOut;
    float *zOut;
    float *dataValsOut;
    int *vlOut, *llOut; // vertex list, line list
    coDoLines *boundaryLines = new coDoLines(p_boundaryLines->getObjName(), outLinesNumNodes, nOuterEdges * 2, nOuterEdges);
    boundaryLines->getAddresses(&xOut, &yOut, &zOut, &vlOut, &llOut);

    coDoFloat *dataOut;
    if (haveData)
    {
        dataOut = new coDoFloat(p_dataOut->getObjName(), outLinesNumNodes);
        dataOut->getAddress(&dataValsOut);
    }

    // coordinates
    for (int i = 0; i < numPoints; i++)
    {
        if (usedNode[i] > -1)
        {
            xOut[usedNode[i]] = x[i];
            yOut[usedNode[i]] = y[i];
            zOut[usedNode[i]] = z[i];
            if (haveData)
            {
                dataValsOut[usedNode[i]] = dataVals[i];
            }
        }
    }

    // connectivity
    pos = 0;
    for (int i = 0; i < nOuterEdges; i++)
    {
        vlOut[2 * i + 0] = usedNode[outerEdges[i].first];
        vlOut[2 * i + 1] = usedNode[outerEdges[i].second];
        llOut[i] = 2 * i;
    }

    delete[] usedNode;
    outerEdges.clear();

    boundaryLines->addAttribute("COLOR", "White");

    // copy all attributes except the COLOR-attribute
    const char **attribName, **attribSetting;
    int numAttribs = obj->getAllAttributes(&attribName, &attribSetting);
    for (int i = 0; i < numAttribs; i++)
    {
        if ((strcmp(attribName[i], "COLOR")) && (strcmp(attribName[i], "MATERIAL")))
        {
            boundaryLines->addAttribute(attribName[i], attribSetting[i]);
        }
    }

    p_boundaryLines->setCurrentObject(boundaryLines);

    if (haveData)
    {
        p_dataOut->setCurrentObject(dataOut);
    }

    delete fixed;

    return SUCCESS;
}

MODULE_MAIN(General, PolyBoundaryLines)
