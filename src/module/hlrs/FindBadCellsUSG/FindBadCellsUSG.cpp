/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: Find cells in an unstructured grid that exhibit certain   **
 **              properties that decrease their usefulness for cfd         **
 **              simulations (e.g. faces containing small angles).         **
 **              The faces of the bad cells are output as polygons.        **
 **                                                                        **
 ** Name:        FindBadCellsUSG                                           **
 ** Category:    Tools                                                     **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "FindBadCellsUSG.h"

#include <util/coVector.h>

/*! \brief constructor
 *
 * create In/Output Ports and module parameters
 */
FindBadCellsUSG::FindBadCellsUSG(int argc, char **argv)
    : coModule(argc, argv, "Find Bad Cells in an USG")
{

    p_mesh = addInputPort("mesh", "UnstructuredGrid", "mesh");
    p_polygons = addOutputPort("polygons", "Polygons", "polygons");
    p_data = addOutputPort("data", "Float", "data");

    p_threshold = addFloatSliderParam("threshold", "Threshold for bad cell detection");
    p_threshold->setValue(0.0, 180.0, 15.0);
}

FindBadCellsUSG::~FindBadCellsUSG()
{
}

/*! \brief param callback
 *
 * called when a parameter in a module is changed.
 */
void FindBadCellsUSG::param(const char * /* name */, bool /* inMapLoading */)
{
}

/*! \brief find the 'bad' cells in an unstructured grid
 *
 * Loops through the cells of the unstructured grid(s) in the given
 * coDistributedObject and finds the ones with 'bad' properties. 
 * Creates coDoPolygons describing the faces of the bad cells, and
 * coDoFloat containing the smallest angle of the cell for each face.
 * Calls createPolygons recursivly to unpack coDoSets.
 *
 * \param polyPort the port used to create polygons
 * \param dataPort the port used to create the angles
 * \param obj the Set or USG that contains the cells to test
 * \param threshold threshold angle for 'bad' cells
 * \param level the recursion level of createPolygons, used to create coDistributedObject with distinct names
 * \return coDoPolygons and coDoFloat containing the created Objects
 */
struct polyData FindBadCellsUSG::createPolygons(coOutputPort *polyPort,
                                                coOutputPort *dataPort,
                                                const coDistributedObject *obj,
                                                float threshold,
                                                int level)
{

    const coDoUnstructuredGrid *grid = NULL;
    const coDoSet *set = NULL;
    coDistributedObject *poly = NULL;
    coDistributedObject *data = NULL;

    if ((set = dynamic_cast<const coDoSet *>(obj)))
    {

        int numChildren = set->getNumElements();
        coDistributedObject **polyList = new coDistributedObject *[numChildren + 1];
        coDistributedObject **dataList = new coDistributedObject *[numChildren + 1];

        for (int index = 0; index < numChildren; index++)
        {

            struct polyData polyData = createPolygons(polyPort, dataPort, set->getElement(index), threshold, level++);
            polyList[index] = polyData.polygons;
            ;
            dataList[index] = polyData.data;
        }

        polyList[numChildren] = NULL;
        dataList[numChildren] = NULL;

        char polyName[128];
        char dataName[128];
        snprintf(polyName, 128, "%s_%d", polyPort->getObjName(), level);
        snprintf(dataName, 128, "%s_%d", dataPort->getObjName(), level);

        poly = new coDoSet(polyName, set->getNumElements(), polyList);
        data = new coDoSet(dataName, set->getNumElements(), dataList);
    }
    else if ((grid = dynamic_cast<const coDoUnstructuredGrid *>(obj)))
    {

        int numElem, numConn, numCoord;
        int *elemList = NULL, *connList = NULL, *typeList = NULL;
        float *x = NULL, *y = NULL, *z = NULL;

        grid->getGridSize(&numElem, &numConn, &numCoord);
        grid->getAddresses(&elemList, &connList, &x, &y, &z);
        grid->getTypeList(&typeList);

        std::vector<float> dataList;
        std::vector<int> polyList;
        std::vector<int> cornerList;
        std::vector<float> px;
        std::vector<float> py;
        std::vector<float> pz;

        int corner, elem;

        for (int elemIndex = 0; elemIndex < numElem; elemIndex++)
        {

            switch (typeList[elemIndex])
            {

            case TYPE_TETRAHEDER:
            {
                float test = testTetrahedron(elemIndex, elemList, connList, x, y, z, threshold);
                if (test)
                {
                    int facelist[12] = { 0, 1, 2,
                                         0, 1, 3,
                                         0, 2, 3,
                                         1, 2, 3 };

                    polyList.push_back(cornerList.size());
                    polyList.push_back(cornerList.size() + 3);
                    polyList.push_back(cornerList.size() + 6);
                    polyList.push_back(cornerList.size() + 9);
                    corner = elemList[elemIndex];

                    for (int index = 0; index < 12; index++)
                    {
                        elem = connList[corner + facelist[index]];
                        cornerList.push_back(px.size());
                        dataList.push_back(test);
                        px.push_back(x[elem]);
                        py.push_back(y[elem]);
                        pz.push_back(z[elem]);
                    }
                }
            }
            break;

            case TYPE_PRISM:
            {

                float test = testPrism(elemIndex, elemList, connList, x, y, z, threshold);
                if (test)
                {
                    int facelist[18] = { 0, 2, 1,
                                         3, 5, 4,
                                         0, 3, 5, 2,
                                         3, 4, 1, 0,
                                         1, 4, 5, 2 };

                    polyList.push_back(cornerList.size());
                    polyList.push_back(cornerList.size() + 3);

                    polyList.push_back(cornerList.size() + 6);
                    polyList.push_back(cornerList.size() + 10);
                    polyList.push_back(cornerList.size() + 14);

                    corner = elemList[elemIndex];

                    for (int index = 0; index < 18; index++)
                    {
                        elem = connList[corner + facelist[index]];
                        cornerList.push_back(px.size());
                        dataList.push_back(test);
                        px.push_back(x[elem]);
                        py.push_back(y[elem]);
                        pz.push_back(z[elem]);
                    }
                }
            }
            break;

            case TYPE_HEXAEDER:
            {

                float test = testHexaeder(elemIndex, elemList, connList, x, y, z, threshold);
                if (test)
                {
                    int facelist[24] = { 0, 1, 2, 3,
                                         4, 5, 6, 7,
                                         0, 1, 5, 4,
                                         3, 2, 6, 7,
                                         1, 2, 6, 5,
                                         0, 3, 7, 4 };

                    polyList.push_back(cornerList.size());
                    polyList.push_back(cornerList.size() + 4);
                    polyList.push_back(cornerList.size() + 8);
                    polyList.push_back(cornerList.size() + 12);
                    polyList.push_back(cornerList.size() + 16);
                    polyList.push_back(cornerList.size() + 20);

                    corner = elemList[elemIndex];

                    for (int index = 0; index < 24; index++)
                    {
                        elem = connList[corner + facelist[index]];
                        cornerList.push_back(px.size());
                        dataList.push_back(test);
                        px.push_back(x[elem]);
                        py.push_back(y[elem]);
                        pz.push_back(z[elem]);
                    }
                }
            }
            break;

            /* 
              * missing cell types: TYPE_PYRAMID,
              * TYPE_QUAD, TYPE_TRIANGLE, TYPE_BAR, TYPE_POINT 
              */
            default:
                break;
            }
        }

        poly = new coDoPolygons(polyPort->getObjName(), px.size(), &px[0], &py[0], &pz[0], cornerList.size(), &cornerList[0], polyList.size(), &polyList[0]);
        data = new coDoFloat(dataPort->getObjName(), dataList.size(), &dataList[0]);
    }

    struct polyData polyData;
    polyData.polygons = poly;
    polyData.data = data;

    return polyData;
}

int FindBadCellsUSG::compute(const char * /* port */)
{

    const coDistributedObject *grid = p_mesh->getCurrentObject();

    float threshold = p_threshold->getValue();
    printf("thresh: %f\n", threshold);
    if (grid)
    {
        struct polyData polyData = createPolygons(p_polygons, p_data, grid, threshold);
        p_polygons->setCurrentObject(polyData.polygons);
        p_data->setCurrentObject(polyData.data);
    }
    else
    {

        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

/*! \brief test a tetrahedron for faces with small angles
 *
 * \param elemIndex the index of the cell to test
 * \param elemList the element list of the usg
 * \param connList the connectivity list of the usg
 * \param x the x coordinates of the points in the usg
 * \param y the y coordinates of the points in the usg
 * \param z the z coordinates of the points in the usg
 * \param threshold the threshold for 'bad' angles
 *
 * \return the smallest angle in the cell, 0 if all angles are larger than the threshold
 */
float FindBadCellsUSG::testTetrahedron(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold)
{

    int p1 = connList[elemList[elemIndex]];
    int p2 = connList[elemList[elemIndex] + 1];
    int p3 = connList[elemList[elemIndex] + 2];
    int p4 = connList[elemList[elemIndex] + 3];

    int lines[12][4] = { { p2, p1, p3, p1 },
                         { p4, p1, p3, p1 },
                         { p2, p1, p4, p1 },

                         { p1, p2, p3, p2 },
                         { p1, p2, p4, p2 },
                         { p4, p2, p3, p2 },

                         { p1, p3, p2, p3 },
                         { p1, p3, p4, p3 },
                         { p2, p3, p4, p3 },

                         { p1, p4, p3, p4 },
                         { p1, p4, p2, p4 },
                         { p2, p4, p3, p4 } };

    for (int index = 0; index < 12; index++)
    {

        coVector a(x[lines[index][0]] - x[lines[index][1]],
                   y[lines[index][0]] - y[lines[index][1]],
                   z[lines[index][0]] - z[lines[index][1]]);

        coVector b(x[lines[index][2]] - x[lines[index][3]],
                   y[lines[index][2]] - y[lines[index][3]],
                   z[lines[index][2]] - z[lines[index][3]]);

        float alpha = (acos((a * b) / (a.length() * b.length()))) * 180 / M_PI;

        if (alpha < threshold)
            return alpha;
    }

    return 0;
}

/*! \brief test a prism for faces with small angles
 *
 * \param elemIndex the index of the cell to test
 * \param elemList the element list of the usg
 * \param connList the connectivity list of the usg
 * \param x the x coordinates of the points in the usg
 * \param y the y coordinates of the points in the usg
 * \param z the z coordinates of the points in the usg
 * \param threshold the threshold for 'bad' angles
 *
 * \return the smallest angle in the cell, 0 if all angles are larger than the threshold
 */
float FindBadCellsUSG::testPrism(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold)
{

    int p1 = connList[elemList[elemIndex]];
    int p2 = connList[elemList[elemIndex] + 1];
    int p3 = connList[elemList[elemIndex] + 2];
    int p4 = connList[elemList[elemIndex] + 3];
    int p5 = connList[elemList[elemIndex] + 4];
    int p6 = connList[elemList[elemIndex] + 5];

    int lines[18][4] = { { p2, p1, p4, p1 },
                         { p3, p1, p2, p1 },
                         { p3, p1, p4, p1 },

                         { p1, p2, p3, p2 },
                         { p1, p2, p5, p2 },
                         { p3, p2, p5, p2 },

                         { p1, p3, p2, p3 },
                         { p1, p3, p6, p3 },
                         { p2, p3, p6, p3 },

                         { p1, p4, p5, p4 },
                         { p5, p4, p6, p4 },
                         { p1, p4, p6, p4 },

                         { p2, p5, p4, p5 },
                         { p2, p5, p6, p5 },
                         { p4, p5, p6, p5 },

                         { p3, p6, p5, p6 },
                         { p4, p6, p3, p6 },
                         { p4, p6, p5, p6 } };

    for (int index = 0; index < 18; index++)
    {

        coVector a(x[lines[index][0]] - x[lines[index][1]],
                   y[lines[index][0]] - y[lines[index][1]],
                   z[lines[index][0]] - z[lines[index][1]]);

        coVector b(x[lines[index][2]] - x[lines[index][3]],
                   y[lines[index][2]] - y[lines[index][3]],
                   z[lines[index][2]] - z[lines[index][3]]);

        float alpha = (acos((a * b) / (a.length() * b.length()))) * 180 / M_PI;

        if (alpha < threshold)
            return alpha;
    }

    return 0;
}

/*! \brief test a hexaeder for faces with small angles
 *
 * \param elemIndex the index of the cell to test
 * \param elemList the element list of the usg
 * \param connList the connectivity list of the usg
 * \param x the x coordinates of the points in the usg
 * \param y the y coordinates of the points in the usg
 * \param z the z coordinates of the points in the usg
 * \param threshold the threshold for 'bad' angles
 *
 * \return the smallest angle in the cell, 0 if all angles are larger than the threshold
 */
float FindBadCellsUSG::testHexaeder(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold)
{

    int p1 = connList[elemList[elemIndex]];
    int p2 = connList[elemList[elemIndex] + 1];
    int p3 = connList[elemList[elemIndex] + 2];
    int p4 = connList[elemList[elemIndex] + 3];
    int p5 = connList[elemList[elemIndex] + 4];
    int p6 = connList[elemList[elemIndex] + 5];
    int p7 = connList[elemList[elemIndex] + 6];
    int p8 = connList[elemList[elemIndex] + 7];

    int lines[24][4] = { { p5, p1, p2, p1 },
                         { p4, p1, p5, p1 },
                         { p2, p1, p4, p1 },

                         { p1, p2, p6, p2 },
                         { p6, p2, p3, p2 },
                         { p3, p2, p1, p2 },

                         { p2, p3, p7, p3 },
                         { p7, p3, p4, p3 },
                         { p4, p3, p2, p3 },

                         { p1, p4, p3, p4 },
                         { p8, p4, p1, p4 },
                         { p3, p4, p8, p4 },

                         { p8, p5, p6, p5 },
                         { p6, p5, p1, p5 },
                         { p1, p5, p8, p5 },

                         { p5, p6, p7, p6 },
                         { p7, p6, p2, p6 },
                         { p2, p6, p5, p6 },

                         { p6, p7, p8, p7 },
                         { p3, p7, p6, p7 },
                         { p8, p7, p3, p7 },

                         { p7, p8, p5, p8 },
                         { p4, p8, p7, p8 },
                         { p5, p8, p4, p8 } };

    for (int index = 0; index < 24; index++)
    {

        coVector a(x[lines[index][0]] - x[lines[index][1]],
                   y[lines[index][0]] - y[lines[index][1]],
                   z[lines[index][0]] - z[lines[index][1]]);

        coVector b(x[lines[index][2]] - x[lines[index][3]],
                   y[lines[index][2]] - y[lines[index][3]],
                   z[lines[index][2]] - z[lines[index][3]]);

        float alpha = (acos((a * b) / (a.length() * b.length()))) * 180 / M_PI;

        if (alpha < threshold)
            return alpha;
    }

    return 0;
}

MODULE_MAIN(IO, FindBadCellsUSG)
