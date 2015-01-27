/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: dieses spitzen Modul wertet StructuredGrids aus und       **
 **              erzeugt die Oberflaeche als Polygone                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  10.09.97                                                        **
\**************************************************************************/

#include <do/coDoTriangleStrips.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoData.h>
#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>

using namespace covise;

////// our class
class GridSurface : public coSimpleModule
{
private:
    coInputPort *pGridIn, *pDataIn;
    coOutputPort *pPolyOut, *pDataOut;

public:
    /// Methods
    //coDistributedObject **ComputeObject(coDistributedObject **, char **, int, int);
    virtual int compute(const char *port);

    GridSurface(int argc, char *argv[]);
    virtual ~GridSurface() {}
};

GridSurface::GridSurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "create surface of a grid")
{
    // input
    pGridIn = addInputPort("GridIn0", "StructuredGrid|UnstructuredGrid|Polygons", "grid input");
    pDataIn = addInputPort("DataIn0", "Float", "data input");
    pDataIn->setRequired(0);

    // output
    pPolyOut = addOutputPort("GridOut0", "Polygons", "computed surface");
    pDataOut = addOutputPort("DataOut0", "Float", "data");
    pDataOut->setDependencyPort(pDataIn);
}

int GridSurface::compute(const char *)
{

    // output objects
    coDoFloat *s3d_data_out = NULL;
    coDoPolygons *poly_out = NULL;
    coDistributedObject **r;

    r = new coDistributedObject *[2];
    r[0] = NULL;
    r[1] = NULL;

    int *pl_out, *vl_out;
    int num_pl_out, num_vl_out;
    float *x_out, *y_out, *z_out;

    //input stuff
    const coDoUniformGrid *ugrid_in = NULL;
    const coDoStructuredGrid *grid_in = NULL;
    const coDoUnstructuredGrid *usg_in = NULL;
    const coDoTriangleStrips *tri_in = NULL;
    const coDoFloat *s3d_data_in = NULL;

    int usg_flag = 0;
    float *in_data = NULL;
    int numData = 0;
    int x_dim = 0, y_dim = 0, z_dim = 0;
    float *x_in = NULL, *y_in = NULL, *z_in = NULL;

    int size_e = 0, size_c = 0, size_v = 0;
    int *el_in = NULL, *cl_in = NULL, *tl_in = NULL;

    int no_strip = 0, no_vert = 0, no_points = 0;
    // counters
    int u, v;
    int b1 = 0, b2 = 0;
    int i, n, m;

    // temporary stuff
    int a, b, c, d, f = 0;

    const coDistributedObject *gridIn = pGridIn->getCurrentObject();
    if (!gridIn)
    {
        sendError("no mesh connected");
        return STOP_PIPELINE;
    }

    // get input
    const char *dataType = gridIn->getType();
    if (strcmp(dataType, "UNSGRD") == 0)
    {
        usg_flag = 1;
        usg_in = (coDoUnstructuredGrid *)gridIn;
        usg_in->getGridSize(&size_e, &size_v, &size_c);
        usg_in->getAddresses(&el_in, &cl_in, &x_in, &y_in, &z_in);
        usg_in->getTypeList(&tl_in);
    }
    else if (strcmp(dataType, "STRGRD") == 0)
    {
        usg_flag = 0;
        grid_in = (coDoStructuredGrid *)gridIn;
        grid_in->getGridSize(&x_dim, &y_dim, &z_dim);
        grid_in->getAddresses(&x_in, &y_in, &z_in);
    }
    else if (strcmp(dataType, "UNIGRD") == 0)
    {
        usg_flag = 3;
        ugrid_in = (coDoUniformGrid *)gridIn;
        ugrid_in->getGridSize(&x_dim, &y_dim, &z_dim);
    }
    else if (strcmp(dataType, "TRIANG") == 0)
    {
        usg_flag = 2;
        tri_in = (coDoTriangleStrips *)gridIn;
        tri_in->getAddresses(&x_in, &y_in, &z_in, &cl_in, &el_in);
        no_strip = tri_in->getNumStrips();
        no_vert = tri_in->getNumVertices();
        size_c = no_points = tri_in->getNumPoints();
    }
    else
    {
        Covise::sendError("incorrect data-type");
        return STOP_PIPELINE;
    }

    const coDistributedObject *dataIn = pDataIn->getCurrentObject();
    if (dataIn)
    {
        dataType = dataIn->getType();
        if (strcmp(dataType, "USTSDT") == 0)
        {
            s3d_data_in = (coDoFloat *)dataIn;
            s3d_data_in->getAddress(&in_data);
            numData = s3d_data_in->getNumPoints();
        }
        else
        {
            Covise::sendInfo("WARNING: incorrect data-type - ignored");
            dataIn = NULL;
        }
    }

    //check for empty objects
    if ((usg_flag && size_c == 0)
        || (!usg_flag && x_dim * y_dim * z_dim == 0))
    {
        r[0] = new coDoPolygons(pPolyOut->getObjName(), 0, 0, 0);
        if (dataIn)
        {
            r[1] = new coDoFloat(pDataOut->getObjName(), 0);
        }
        else
        {
            r[1] = NULL;
        }
        return CONTINUE_PIPELINE;
    }

    // check for per-vertex-data
    if (usg_flag)
    {
        if (dataIn && numData != size_c)
        {
            Covise::sendError("ERROR: data has to be per vertex !");
            return STOP_PIPELINE;
        }
    }
    else
    {
        if (dataIn && numData != x_dim * y_dim * z_dim)
        {
            Covise::sendError("ERROR: data has to be per vertex !");
            return STOP_PIPELINE;
        }
    }

    //////
    ////// UnstructuredGrid
    //////

    if (usg_flag == 1)
    {

        // generate output-objects
        // ..poly
        // (-, BAR, TRIANGLE, QUAD, TETRAHEDER, PYRAMID, PRISM, HEXAEDER, -, -, POINT)
        int elem_to_numpol[] = { 0, 1, 1, 1, 4, 5, 5, 6, 0, 0, 1 };
        int elem_to_numcon[] = { 0, 2, 3, 4, 12, 16, 18, 24, 0, 0, 1 };

        // compute how many output-polygons and vertices we'll get
        num_pl_out = 0;
        num_vl_out = 0;
        for (i = 0; i < size_e; i++)
        {
            num_pl_out += elem_to_numpol[tl_in[i]];
            num_vl_out += elem_to_numcon[tl_in[i]];
        }

        poly_out = new coDoPolygons(pPolyOut->getObjName(), size_c, num_vl_out, num_pl_out);
        poly_out->addAttribute("vertexOrder", "2");
        poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);

        // ..data
        if (dataIn)
        {
            s3d_data_out = new coDoFloat(pDataOut->getObjName(), numData, in_data);
        }

        // compute output
        num_pl_out = 0;
        num_vl_out = 0;
        for (i = 0; i < size_e; i++)
        {
            a = el_in[i];

            switch (tl_in[i])
            {
            case TYPE_BAR:
                pl_out[num_pl_out] = num_vl_out;
                num_pl_out++;

                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                num_vl_out += 2;

                break;

            case TYPE_TRIANGLE:
                pl_out[num_pl_out] = num_vl_out;
                num_pl_out++;

                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                num_vl_out += 3;

                break;

            case TYPE_QUAD:
                pl_out[num_pl_out] = num_vl_out;
                num_pl_out++;

                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                vl_out[num_vl_out + 3] = cl_in[a + 3];
                num_vl_out += 4;

                break;

            case TYPE_TETRAHEDER:
                pl_out[num_pl_out] = num_vl_out;
                pl_out[num_pl_out + 1] = num_vl_out + 3;
                pl_out[num_pl_out + 2] = num_vl_out + 6;
                pl_out[num_pl_out + 3] = num_vl_out + 9;
                num_pl_out += 4;

                // 1-4-2
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 3];
                vl_out[num_vl_out + 2] = cl_in[a + 1];
                num_vl_out += 3;

                // 2-4-3
                vl_out[num_vl_out] = cl_in[a + 1];
                vl_out[num_vl_out + 1] = cl_in[a + 3];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                num_vl_out += 3;

                // 3-4-1
                vl_out[num_vl_out] = cl_in[a + 2];
                vl_out[num_vl_out + 1] = cl_in[a + 3];
                vl_out[num_vl_out + 2] = cl_in[a];
                num_vl_out += 3;

                // 1-2-3
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                num_vl_out += 3;

                break;

            case TYPE_PYRAMID:
                pl_out[num_pl_out] = num_vl_out;
                pl_out[num_pl_out + 1] = num_vl_out + 3;
                pl_out[num_pl_out + 2] = num_vl_out + 6;
                pl_out[num_pl_out + 3] = num_vl_out + 9;
                pl_out[num_pl_out + 4] = num_vl_out + 12;
                num_pl_out += 5;

                // 1-5-2
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 4];
                vl_out[num_vl_out + 2] = cl_in[a + 1];
                num_vl_out += 3;

                // 2-5-3
                vl_out[num_vl_out] = cl_in[a + 1];
                vl_out[num_vl_out + 1] = cl_in[a + 4];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                num_vl_out += 3;

                // 3-5-4
                vl_out[num_vl_out] = cl_in[a + 2];
                vl_out[num_vl_out + 1] = cl_in[a + 4];
                vl_out[num_vl_out + 2] = cl_in[a + 3];
                num_vl_out += 3;

                // 4-5-1
                vl_out[num_vl_out] = cl_in[a + 3];
                vl_out[num_vl_out + 1] = cl_in[a + 4];
                vl_out[num_vl_out + 2] = cl_in[a];
                num_vl_out += 3;

                // 1-2-3-4
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 1] = cl_in[a + 2];
                vl_out[num_vl_out + 2] = cl_in[a + 3];
                num_vl_out += 4;

                break;

            case TYPE_PRISM:
                pl_out[num_pl_out] = num_vl_out;
                pl_out[num_pl_out + 1] = num_vl_out + 3;
                pl_out[num_pl_out + 2] = num_vl_out + 6;
                pl_out[num_pl_out + 3] = num_vl_out + 10;
                pl_out[num_pl_out + 4] = num_vl_out + 14;
                num_pl_out += 5;

                // 1-2-3
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                num_vl_out += 3;

                // 4-6-5
                vl_out[num_vl_out] = cl_in[a + 3];
                vl_out[num_vl_out + 1] = cl_in[a + 5];
                vl_out[num_vl_out + 2] = cl_in[a + 4];
                num_vl_out += 3;

                // 6-3-2-5
                vl_out[num_vl_out] = cl_in[a + 5];
                vl_out[num_vl_out + 1] = cl_in[a + 2];
                vl_out[num_vl_out + 2] = cl_in[a + 1];
                vl_out[num_vl_out + 3] = cl_in[a + 4];
                num_vl_out += 4;

                // 5-2-1-4
                vl_out[num_vl_out] = cl_in[a + 4];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a];
                vl_out[num_vl_out + 3] = cl_in[a + 3];
                num_vl_out += 4;

                // 1-3-6-4
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 2];
                vl_out[num_vl_out + 2] = cl_in[a + 5];
                vl_out[num_vl_out + 3] = cl_in[a + 3];
                num_vl_out += 4;

                break;

            case TYPE_HEXAEDER:
                pl_out[num_pl_out] = num_vl_out;
                pl_out[num_pl_out + 1] = num_vl_out + 4;
                pl_out[num_pl_out + 2] = num_vl_out + 8;
                pl_out[num_pl_out + 3] = num_vl_out + 12;
                pl_out[num_pl_out + 4] = num_vl_out + 16;
                pl_out[num_pl_out + 5] = num_vl_out + 20;
                num_pl_out += 6;

                // 1-4-8-5
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 3];
                vl_out[num_vl_out + 2] = cl_in[a + 7];
                vl_out[num_vl_out + 3] = cl_in[a + 4];
                num_vl_out += 4;

                // 8-4-3-7
                vl_out[num_vl_out] = cl_in[a + 7];
                vl_out[num_vl_out + 1] = cl_in[a + 3];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                vl_out[num_vl_out + 3] = cl_in[a + 6];
                num_vl_out += 4;

                // 7-3-2-6
                vl_out[num_vl_out] = cl_in[a + 6];
                vl_out[num_vl_out + 1] = cl_in[a + 2];
                vl_out[num_vl_out + 2] = cl_in[a + 1];
                vl_out[num_vl_out + 3] = cl_in[a + 5];
                num_vl_out += 4;

                // 6-2-1-5
                vl_out[num_vl_out] = cl_in[a + 5];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a];
                vl_out[num_vl_out + 3] = cl_in[a + 4];
                num_vl_out += 4;

                // 1-2-3-4
                vl_out[num_vl_out] = cl_in[a];
                vl_out[num_vl_out + 1] = cl_in[a + 1];
                vl_out[num_vl_out + 2] = cl_in[a + 2];
                vl_out[num_vl_out + 3] = cl_in[a + 3];
                num_vl_out += 4;

                // 5-8-7-6
                vl_out[num_vl_out] = cl_in[a + 4];
                vl_out[num_vl_out + 1] = cl_in[a + 7];
                vl_out[num_vl_out + 2] = cl_in[a + 6];
                vl_out[num_vl_out + 3] = cl_in[a + 5];
                num_vl_out += 4;

                break;

            case TYPE_POINT:
                pl_out[num_pl_out] = num_vl_out;
                num_pl_out++;

                vl_out[num_vl_out] = cl_in[a];
                num_vl_out++;

                break;

            default:
                Covise::sendError("ERROR: unknown element found !");
                return STOP_PIPELINE;
            }
        }

        // copy coordinates
        memcpy(x_out, x_in, size_c * sizeof(float));
        memcpy(y_out, y_in, size_c * sizeof(float));
        memcpy(z_out, z_in, size_c * sizeof(float));

        // return
        r[0] = poly_out;
        r[1] = s3d_data_out;

        // done
    }

    //////
    ////// StructuredGrid
    //////

    else if (usg_flag == 0)
    {
        // only 2D supported
        if (x_dim != 1 && y_dim != 1 && z_dim != 1)
        {
            Covise::sendError("ERROR: this module only supports 2D structured grids!");
            return STOP_PIPELINE;
        }

        // generate output-objects
        if (z_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), x_dim * y_dim, (x_dim - 1) * (y_dim - 1) * 4, (x_dim - 1) * (y_dim - 1));
            b1 = y_dim - 1;
            b2 = x_dim - 1;
            f = y_dim;
        }
        else if (y_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), z_dim * x_dim, (z_dim - 1) * (x_dim - 1) * 4, (z_dim - 1) * (x_dim - 1));
            b1 = z_dim - 1;
            b2 = x_dim - 1;
            f = z_dim;
        }
        else if (x_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), y_dim * z_dim, (y_dim - 1) * (z_dim - 1) * 4, (y_dim - 1) * (z_dim - 1));
            b1 = z_dim - 1;
            b2 = y_dim - 1;
            f = z_dim;
        }
        poly_out->addAttribute("vertexOrder", "2");

        poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);
        if (dataIn)
        {
            s3d_data_out = new coDoFloat(pDataOut->getObjName(), x_dim * y_dim * z_dim, in_data);
        }

        // init
        num_pl_out = 0;
        num_vl_out = 0;

        // work through the entire grid
        for (v = 0; v < b1; v++)
        {
            for (u = 0; u < b2; u++)
            {
                // add polygon
                pl_out[num_pl_out] = num_vl_out;
                num_pl_out++;

                // compute offsets
                a = f * u + v;
                b = f * u + v + 1;
                c = f * (u + 1) + v + 1;
                d = f * (u + 1) + v;

                // add vertices
                vl_out[num_vl_out] = a;
                vl_out[num_vl_out + 1] = b;
                vl_out[num_vl_out + 2] = c;
                vl_out[num_vl_out + 3] = d;
                num_vl_out += 4;

                // add coords
                x_out[a] = x_in[a];
                y_out[a] = y_in[a];
                z_out[a] = z_in[a];

                x_out[b] = x_in[b];
                y_out[b] = y_in[b];
                z_out[b] = z_in[b];

                x_out[c] = x_in[c];
                y_out[c] = y_in[c];
                z_out[c] = z_in[c];

                x_out[d] = x_in[d];
                y_out[d] = y_in[d];
                z_out[d] = z_in[d];
            }
        }

        // return
        r[0] = poly_out;
        r[1] = s3d_data_out;
    }
    //////
    ////// UniformGrid
    //////

    else if (usg_flag == 3)
    {
        // only 2D supported
        if (x_dim != 1 && y_dim != 1 && z_dim != 1)
        {
            Covise::sendError("ERROR: this module only supports 2D uniform grids!");
            return STOP_PIPELINE;
        }
        float x_min, x_max, y_min, y_max, z_min, z_max;
        ugrid_in->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);

        // init
        num_pl_out = 0;
        num_vl_out = 0;
        // generate output-objects
        if (z_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), x_dim * y_dim, (x_dim - 1) * (y_dim - 1) * 4, (x_dim - 1) * (y_dim - 1));
            b1 = y_dim - 1;
            b2 = x_dim - 1;
            f = y_dim;
            poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);

            // work through the entire grid
            for (v = 0; v < b1; v++)
            {
                for (u = 0; u < b2; u++)
                {
                    // add polygon
                    pl_out[num_pl_out] = num_vl_out;
                    num_pl_out++;

                    // compute offsets
                    a = f * u + v;
                    b = f * u + v + 1;
                    c = f * (u + 1) + v + 1;
                    d = f * (u + 1) + v;

                    // add vertices
                    vl_out[num_vl_out] = a;
                    vl_out[num_vl_out + 1] = b;
                    vl_out[num_vl_out + 2] = c;
                    vl_out[num_vl_out + 3] = d;
                    num_vl_out += 4;

                    // add coords
                    x_out[a] = x_min + ((float)(u) / (float)(b2)) * (x_max - x_min);
                    y_out[a] = y_min + ((float)(v) / (float)(b1)) * (y_max - y_min);
                    z_out[a] = z_min;

                    x_out[b] = x_out[a];
                    y_out[b] = y_min + ((float)(v + 1) / (float)(b1)) * (y_max - y_min);
                    z_out[b] = z_min;

                    x_out[c] = x_min + ((float)(u + 1) / (float)(b2)) * (x_max - x_min);
                    y_out[c] = y_out[a];
                    z_out[c] = z_min;

                    x_out[d] = x_out[c];
                    y_out[d] = y_out[b];
                    z_out[d] = z_min;
                }
            }
        }
        else if (y_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), z_dim * x_dim, (z_dim - 1) * (x_dim - 1) * 4, (z_dim - 1) * (x_dim - 1));
            b1 = z_dim - 1;
            b2 = x_dim - 1;
            f = z_dim;
            poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);

            // work through the entire grid
            for (v = 0; v < b1; v++)
            {
                for (u = 0; u < b2; u++)
                {
                    // add polygon
                    pl_out[num_pl_out] = num_vl_out;
                    num_pl_out++;

                    // compute offsets
                    a = f * u + v;
                    b = f * u + v + 1;
                    c = f * (u + 1) + v + 1;
                    d = f * (u + 1) + v;

                    // add vertices
                    vl_out[num_vl_out] = a;
                    vl_out[num_vl_out + 1] = b;
                    vl_out[num_vl_out + 2] = c;
                    vl_out[num_vl_out + 3] = d;
                    num_vl_out += 4;

                    // add coords
                    x_out[a] = x_min + ((float)(u) / (float)(b2)) * (x_max - x_min);
                    y_out[a] = y_min;
                    z_out[a] = z_min + ((float)(v) / (float)(b1)) * (z_max - z_min);

                    x_out[b] = x_out[a];
                    y_out[b] = y_min;
                    z_out[b] = z_min + ((float)(v + 1) / (float)(b1)) * (z_max - z_min);

                    x_out[c] = x_min + ((float)(u + 1) / (float)(b2)) * (x_max - x_min);
                    y_out[c] = y_min;
                    z_out[c] = z_out[a];

                    x_out[d] = x_out[c];
                    y_out[d] = y_min;
                    z_out[d] = z_out[b];
                }
            }
        }
        else if (x_dim == 1)
        {
            poly_out = new coDoPolygons(pPolyOut->getObjName(), y_dim * z_dim, (y_dim - 1) * (z_dim - 1) * 4, (y_dim - 1) * (z_dim - 1));
            b1 = z_dim - 1;
            b2 = y_dim - 1;
            f = z_dim;
            poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);

            // work through the entire grid
            for (v = 0; v < b1; v++)
            {
                for (u = 0; u < b2; u++)
                {
                    // add polygon
                    pl_out[num_pl_out] = num_vl_out;
                    num_pl_out++;

                    // compute offsets
                    a = f * u + v;
                    b = f * u + v + 1;
                    c = f * (u + 1) + v + 1;
                    d = f * (u + 1) + v;

                    // add vertices
                    vl_out[num_vl_out] = a;
                    vl_out[num_vl_out + 1] = b;
                    vl_out[num_vl_out + 2] = c;
                    vl_out[num_vl_out + 3] = d;
                    num_vl_out += 4;

                    // add coords
                    x_out[a] = x_min;
                    y_out[a] = y_min + ((float)(u) / (float)(b2)) * (y_max - y_min);
                    z_out[a] = z_min + ((float)(v) / (float)(b1)) * (z_max - z_min);

                    x_out[b] = x_min;
                    y_out[b] = y_out[a];
                    z_out[b] = z_min + ((float)(v + 1) / (float)(b1)) * (z_max - z_min);

                    x_out[c] = x_min;
                    y_out[c] = y_min + ((float)(u + 1) / (float)(b2)) * (y_max - y_min);
                    z_out[c] = z_out[a];

                    x_out[d] = x_min;
                    y_out[d] = y_out[c];
                    z_out[d] = z_out[b];
                }
            }
        }
        poly_out->addAttribute("vertexOrder", "2");

        if (dataIn)
        {
            s3d_data_out = new coDoFloat(pDataOut->getObjName(), x_dim * y_dim * z_dim, in_data);
        }

        // return
        r[0] = poly_out;
        r[1] = s3d_data_out;
    }

    // TRISTRIPS

    else if (usg_flag == 2)
    {
        int n_tri = 0;
        for (i = 0; i < no_strip; i++)
        {
            if (i == no_strip - 1)
                n = no_vert - el_in[i] - 2;
            else
                n = el_in[i + 1] - el_in[i] - 2;
            n_tri += n;
        }
        poly_out = new coDoPolygons(pPolyOut->getObjName(), no_points, n_tri * 3, n_tri);
        poly_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);
        // copy coordinates
        memcpy(x_out, x_in, no_points * sizeof(float));
        memcpy(y_out, y_in, no_points * sizeof(float));
        memcpy(z_out, z_in, no_points * sizeof(float));
        for (i = 0; i < n_tri; i++)
        {
            pl_out[i] = i * 3;
        }
        n_tri = 0;
        for (i = 0; i < no_strip; i++)
        {
            if (i == no_strip - 1)
                n = no_vert - el_in[i] - 2;
            else
                n = el_in[i + 1] - el_in[i] - 2;
            for (m = 0; m < n; m++)
            {
                vl_out[n_tri * 3 + m * 3] = cl_in[el_in[i] + m];
                if (m % 2)
                {
                    vl_out[n_tri * 3 + m * 3 + 1] = cl_in[el_in[i] + m + 1];
                    vl_out[n_tri * 3 + m * 3 + 2] = cl_in[el_in[i] + m + 2];
                }
                else
                {
                    vl_out[n_tri * 3 + m * 3 + 1] = cl_in[el_in[i] + m + 2];
                    vl_out[n_tri * 3 + m * 3 + 2] = cl_in[el_in[i] + m + 1];
                }
            }
            n_tri += n;
        }

        // return
        r[0] = poly_out;
        r[1] = NULL;
    }

    pPolyOut->setCurrentObject(r[0]);
    pDataOut->setCurrentObject(r[1]);

    //////
    ////// finished
    //////

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, GridSurface)
