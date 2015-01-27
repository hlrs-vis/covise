/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                               (C)2000    **
 **                                                                          **
 ** Description: Surface of a structured grid                                **
 **                                                                          **
 ** Name:        SDomainsurface                                              **
 ** Category:    general                                                     **
 **                                                                          **
 ** Author: Sven Kufer		                                            **
 **                                                                          **
 ** History:  								    **
 ** January-01     					       		    **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#define TO_DELETE -1
#define TRANS trans_coord[cl[num_corners_out - 1]] = cl[num_corners_out - 1]
#define LTRANS ltrans_coord[my_lcl[lnum_corners_out - 1]] = my_lcl[lnum_corners_out - 1]

#include "DomainSurface.h"
#include <util/coviseCompat.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>

SDomainsurface::SDomainsurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Domain surfaces of grids")
{
    p_grid1 = addInputPort("GridIn0", "StructuredGrid|RectilinearGrid|UniformGrid|UnstructuredGrid", "grid");
    // parameter to throw away sides of structured grids
    //  p_string = addStringParam("throw_away", " " );
    //  p_string->setValue(",");
    p_data1 = addInputPort("DataIn0", "Float|Vec3|Mat3", "data on grid points");
    p_data1->setRequired(0);

    outPort_polySet = addOutputPort("GridOut0", "Polygons", "surface of structured grid");
    outPort_dataP = addOutputPort("DataOut0", "Float|Vec3|Mat3", "data on surface");
    outPort_dataP->setDependencyPort(p_data1);

    outPort_bound = addOutputPort("GridOut1", "Lines", "feature lines ");
    outPort_dataL = addOutputPort("DataOut1", "Float|Vec3|Mat3", "data on feature lines");
    outPort_dataL->setDependencyPort(p_data1);

    param_angle = addFloatParam("angle", "feature angle");
    //param_angle->setValue(0.0, 1.0, 0.01);
    param_angle->setValue(0.1f);

    param_vertex = addFloatVectorParam("vertex", "normal for back-face culling");
    param_vertex->setValue(1.f, 0.f, 0.f);

    param_scalar = addFloatParam("scalar", "threshold for back-face culling");
    param_scalar->setValue(1.5f);

    param_double = addBooleanParam("double", "check for duplicated vertices");
    param_double->setValue(true);

    //    param_optimize = addChoiceParam("optimize", "optimize for memory or speed");
    //    char *choices[] = {(char *)"speed", (char *)"memory"};
    //    param_optimize->setValue(2, choices, Speed);
}

int SDomainsurface::compute(const char *)
{
    const coDistributedObject *data_in[2];
    coDoVec3 *v_data_out[2];
    coDoFloat *s_data_out[2];
    coDoMat3 *m_data_out[2];
    const coDoVec3 *v_data_in[2];
    const coDoFloat *s_data_in[2];
    const coDoMat3 *m_data_in[2];

    float *sdata_in[2], *sdata_out[2];
    float *mdata_in[2], *mdata_out[2];
    float *xdata_in[2], *ydata_in[2], *zdata_in[2];
    float *xdata_out[2], *ydata_out[2], *zdata_out[2];

    const coDoStructuredGrid *sgrid_in;
    const coDoUniformGrid *ugrid_in = NULL;
    const coDoRectilinearGrid *rgrid_in = NULL;

    const coDistributedObject *mesh_in;
    coDoPolygons *poly_out;
    int num_corners_out = 0, num_polygons_out = 0, num_points_out = 0;
    int *pl, *cl, *cl_out, *pl_out;
    float *x_coords_out, *y_coords_out, *z_coords_out;

    int *trans_coord; // new point number
    int x_size, y_size, z_size;
    int i, j, k;

    coDoLines *lines_out;
    int lnum_corners_out = 0, num_lines = 0, lnum_points_out = 0;
    int *ll, *lcl;
    int *my_ll, *my_lcl;
    int *ltrans_coord;

    // Get parameter values
    angle = param_angle->getValue();

    tresh = param_angle->getValue();
    scalar = param_scalar->getValue();
    param_vertex->getValue(n2x, n2y, n2z);
    doDoubleCheck = param_double->getValue();
    //    int optim = param_optimize->getValue();
    //    MEMORY_OPTIMIZED = optim-1;

    // Read structure
    mesh_in = p_grid1->getCurrentObject();

    if (mesh_in == NULL)
    {
        sendError("There is no input grid!");
        return STOP_PIPELINE;
    }

    sgrid_in = dynamic_cast<const coDoStructuredGrid *>(mesh_in);
    rgrid_in = dynamic_cast<const coDoRectilinearGrid *>(mesh_in);
    ugrid_in = dynamic_cast<const coDoUniformGrid *>(mesh_in);

    // Structured grids
    if (sgrid_in)
    {
        sgrid_in->getAddresses(&x_coords_in, &y_coords_in, &z_coords_in);
        sgrid_in->getGridSize(&x_size, &y_size, &z_size);
    }

    // Rectilinear grids
    else if (rgrid_in)
    {
        rgrid_in->getGridSize(&x_size, &y_size, &z_size);
        x_coords_in = new float[x_size * y_size * z_size];
        y_coords_in = new float[x_size * y_size * z_size];
        z_coords_in = new float[x_size * y_size * z_size];
    }

    // Uniform grids
    else if (ugrid_in)
    {
        ugrid_in->getGridSize(&x_size, &y_size, &z_size);
        x_coords_in = new float[x_size * y_size * z_size];
        y_coords_in = new float[x_size * y_size * z_size];
        z_coords_in = new float[x_size * y_size * z_size];
    }

    // Unstructured grids (general polyhedral meshes)
    else if (dynamic_cast<const coDoUnstructuredGrid *>(mesh_in))
    {
        // this was copied from the old DomainSurface, do all the work here
        coDistributedObject *meshOut, *dataOut, *linesOut, *ldataOut;

        doModule(mesh_in, p_data1->getCurrentObject(),
                 outPort_polySet->getObjName(), outPort_dataP->getObjName(),
                 outPort_bound->getObjName(), outPort_dataL->getObjName(),
                 &meshOut, &dataOut,
                 &linesOut, &ldataOut);

        outPort_polySet->setCurrentObject(meshOut);
        outPort_dataP->setCurrentObject(dataOut);
        outPort_bound->setCurrentObject(linesOut);
        outPort_dataL->setCurrentObject(ldataOut);

        return CONTINUE_PIPELINE;
    }

    else
    {
        sendWarning("Input grid is not a grid!");
        outPort_polySet->setCurrentObject(0);
        outPort_dataP->setCurrentObject(0);
        outPort_bound->setCurrentObject(0);
        outPort_dataL->setCurrentObject(0);

        return CONTINUE_PIPELINE;
    }
    /*
   //
   // get element number
   //

   char *name;
   char number[10];
   name = ((coDistributedObject *)sgrid_in)->getName();
   int pos = strlen(name)-1, t;
   int number_pos = 8;
   for ( t=0; t<=pos; t++ )
   number[t]=' ';
   number[9] = 0;
   int elem_num;
   while( name[pos] != '_' ) number[number_pos--] = name[pos--];
   //cerr << number << endl;

   sscanf( number, "%d", &elem_num );

   bool throw_away[10000][6];

   //
   // read throw aways
   //

   int num1, num2;

   elem_num--;
   //cerr << elem_num << endl;
   if( elem_num==0 )
   {
   const char *add_string = ","; //p_string->getValue();
   char teststring[800];
   const char *attr_string = sgrid_in->getAttribute("THROWAWAY") ;
   //strcpy( teststring, attr_string);
   //strcat( teststring, add_string);
   strcpy( teststring, add_string);
   cerr << endl << teststring << endl;
   i=0;
   for(s=0; s<40; s++ )
   for( i=0; i<6; i++ )
   throw_away[s][i] = false;
   i=0;
   while( teststring[i] && teststring[i]!=',' )
   {
   for ( t=0; t<=8; t++ )
   number[t]=' ';
   number_pos = 0;
   while( teststring[i]!=':' ) number[number_pos++] = teststring[i++];
   sscanf( number, "%d", &num1);
   i++;

   for ( t=0; t<=8; t++ )
   number[t]=' ';
   number_pos = 0;
   while( teststring[i] != ',' ) number[number_pos++] = teststring[i++];
   sscanf( number, "%d", &num2);
   i++;
   cerr << num1 << ":" << num2 << "," ;
   throw_away[num1][num2] = true;
   }
   cerr << endl;
   }
   */

    pl = new int[2 * (y_size - 1) * (z_size - 1) + 2 * (x_size - 1) * (z_size - 1) + 2 * (x_size - 1) * (y_size - 1)];
    cl = new int[8 * (y_size - 1) * (z_size - 1) + 8 * (x_size - 1) * (z_size - 1) + 8 * (x_size - 1) * (y_size - 1)];

    my_ll = new int[8 * y_size * z_size + 8 * x_size * z_size + 8 * x_size * y_size];
    my_lcl = new int[16 * y_size * z_size + 16 * x_size * z_size + 16 * x_size * y_size];

    trans_coord = new int[x_size * y_size * z_size];
    ltrans_coord = new int[x_size * y_size * z_size];

    for (i = 0; i < x_size * y_size * z_size; i++)
    {
        trans_coord[i] = TO_DELETE;
        ltrans_coord[i] = TO_DELETE;
    }

    /*************************/
    /* Sides with constant x  */
    /*************************/

    for (k = 0; k < z_size - 1; k++)
    {
        // j+1 = 0

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = k + 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = k;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + k + 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + k;
        LTRANS;
    }

    for (j = 0; j < y_size - 1; j++)
    {
        // k+1 = 0

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (j + 1) * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = j * z_size;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + j * z_size;
        LTRANS;

        for (k = 0; k < z_size - 1; k++)
        {
            // x = 0

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = j * z_size + k;
            TRANS;
            cl[num_corners_out++] = (j + 1) * z_size + k;
            TRANS;
            cl[num_corners_out++] = (j + 1) * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = j * z_size + k + 1;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (k == z_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (j + 1) * z_size + (k + 1) % (z_size - 1), j * z_size + (k + 1) % (z_size - 1), (j + 1) * z_size + (k + 1) % (z_size - 1) + 1) >= angle)) || (k == z_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (j + 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = j * z_size + k + 1;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (j == y_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], ((j + 1) % (y_size - 1) + 1) * z_size + k, ((j + 1) % (y_size - 1)) * z_size + k, ((j + 1) % (y_size - 1) + 1) * z_size + k + 1) >= angle)) || (j == y_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (j + 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (j + 1) * z_size + k;
                LTRANS;
            }

            if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              k, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              k, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(0, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            // x= x_size-1

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = (x_size - 1) * y_size * z_size + j * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size + k;
            TRANS;
            cl[num_corners_out++] = (x_size - 1) * y_size * z_size + j * z_size + k;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (k == z_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (x_size - 1) * y_size * z_size + (j + 1) * z_size + (k + 1) % (z_size - 1) + 1, (x_size - 1) * y_size * z_size + (j + 1) * z_size + (k + 1) % (z_size - 1), (x_size - 1) * y_size * z_size + j * z_size + (k + 1) % (z_size - 1) + 1) >= angle)) || (k == z_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + j * z_size + k + 1;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (j == y_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (x_size - 1) * y_size * z_size + ((j + 1) % (y_size - 1) + 1) * z_size + k + 1, (x_size - 1) * y_size * z_size + ((j + 1) % (y_size - 1) + 1) * z_size + k, (x_size - 1) * y_size * z_size + (j + 1) % (y_size - 1) * z_size + k + 1) >= angle)) || (j == y_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (x_size - 1) * y_size * z_size + (j + 1) * z_size + k;
                LTRANS;
            }

            if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              k, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(x_size - 1, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              k, &z_coords_in[cl[num_corners_out - 1]]);
            }
        }
    }

    /*************************/
    /* Sides with constant y  */
    /*************************/

    for (k = 0; k < z_size - 1; k++)
    {
        // i+1 = 0
        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (y_size - 1) * z_size + k + 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = (y_size - 1) * z_size + k;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = k + 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = k;
        LTRANS;
    }

    for (i = 0; i < x_size - 1; i++)
    {
        // k+1 = 0

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = i * y_size * z_size + (y_size - 1) * z_size;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = i * y_size * z_size;
        LTRANS;

        for (k = 0; k < z_size - 1; k++)
        {
            // y = y_size-1

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = i * y_size * z_size + (y_size - 1) * z_size + k;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size + k;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = i * y_size * z_size + (y_size - 1) * z_size + k + 1;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (k == z_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (i + 1) * y_size * z_size + (y_size - 1) * z_size + (k + 1) % (z_size - 1), (i + 1) * y_size * z_size + (y_size - 1) * z_size + (k + 1) % (z_size - 1) + 1, i * y_size * z_size + (y_size - 1) * z_size + (k + 1) % (z_size - 1)) >= angle)) || (k == z_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = i * y_size * z_size + (y_size - 1) * z_size + k + 1;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (i == x_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], ((i + 1) % (x_size - 1) + 1) * y_size * z_size + (y_size - 1) * z_size + k, ((i + 1) % (x_size - 1) + 1) * y_size * z_size + (y_size - 1) * z_size + k + 1, ((i + 1) % (x_size - 1)) * y_size * z_size + (y_size - 1) * z_size + k) >= angle)) || (i == x_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (y_size - 1) * z_size + k;
                LTRANS;
            }

            if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], y_size - 1, &y_coords_in[cl[num_corners_out - 4]],
                                              k, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], y_size - 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], y_size - 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], y_size - 1, &y_coords_in[cl[num_corners_out - 1]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], y_size - 1, &y_coords_in[cl[num_corners_out - 4]],
                                              k, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], y_size - 1, &y_coords_in[cl[num_corners_out - 3]],
                                              k, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], y_size - 1, &y_coords_in[cl[num_corners_out - 2]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], y_size - 1, &y_coords_in[cl[num_corners_out - 1]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            // y=0

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = i * y_size * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + k + 1;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + k;
            TRANS;
            cl[num_corners_out++] = i * y_size * z_size + k;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (k == z_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (i + 1) * y_size * z_size + (k + 1) % (z_size - 1) + 1, (i + 1) * y_size * z_size + (k + 1) % (z_size - 1), i * y_size * z_size + (k + 1) % (z_size - 1) + 1) >= angle)) || (k == z_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = i * y_size * z_size + k + 1;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (i == x_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], ((i + 1) % (x_size - 1) + 1) * y_size * z_size + k, ((i + 1) % (x_size - 1) + 1) * y_size * z_size + k + 1, ((i + 1) % (x_size - 1)) * y_size * z_size + k) >= angle)) || (i == x_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + k + 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + k;
                LTRANS;
            }

            if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], 0, &y_coords_in[cl[num_corners_out - 4]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], 0, &y_coords_in[cl[num_corners_out - 3]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], 0, &y_coords_in[cl[num_corners_out - 2]],
                                              k, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], 0, &y_coords_in[cl[num_corners_out - 1]],
                                              k, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], 0, &y_coords_in[cl[num_corners_out - 4]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], 0, &y_coords_in[cl[num_corners_out - 3]],
                                              k + 1, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], 0, &y_coords_in[cl[num_corners_out - 2]],
                                              k, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], 0, &y_coords_in[cl[num_corners_out - 1]],
                                              k, &z_coords_in[cl[num_corners_out - 1]]);
            }
        }
    }

    /*************************/
    /* Sides with constant z  */
    /*************************/

    for (j = 0; j < y_size - 1; j++)
    {
        // i+1 = 0

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (j + 2) * z_size - 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = (j + 1) * z_size - 1;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (j + 1) * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = j * z_size;
        LTRANS;
    }

    for (i = 0; i < x_size - 1; i++)
    {
        // j+1=0

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = i * y_size * z_size + z_size - 1;
        LTRANS;
        my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + z_size - 1;
        LTRANS;

        my_ll[num_lines++] = lnum_corners_out;

        my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size;
        LTRANS;
        my_lcl[lnum_corners_out++] = i * y_size * z_size;
        LTRANS;

        for (j = 0; j < y_size - 1; j++)
        {
            // z = z_size-1

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = i * y_size * z_size + (j + 2) * z_size - 1;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + (j + 2) * z_size - 1;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + (j + 1) * z_size - 1;
            TRANS;
            cl[num_corners_out++] = i * y_size * z_size + (j + 1) * z_size - 1;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (j == y_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (i + 1) * y_size * z_size + ((j + 1) % (y_size - 1) + 2) * z_size - 1, (i + 1) * y_size * z_size + ((j + 1) % (y_size - 1) + 1) * z_size - 1, i * y_size * z_size + ((j + 1) % (y_size - 1) + 2) * z_size - 1) >= angle)) || (j == y_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = i * y_size * z_size + (j + 2) * z_size - 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (j + 2) * z_size - 1;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (i == x_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], ((i + 1) % (x_size - 1) + 1) * y_size * z_size + (j + 2) * z_size - 1, ((i + 1) % (x_size - 1) + 1) * y_size * z_size + (j + 1) * z_size - 1, ((i + 1) % (x_size - 1)) * y_size * z_size + (j + 2) * z_size - 1) >= angle)) || (i == x_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (j + 2) * z_size - 1;
                LTRANS;
                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (j + 1) * z_size - 1;
                LTRANS;
            }

            if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], j + 1, &y_coords_in[cl[num_corners_out - 4]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], j, &y_coords_in[cl[num_corners_out - 2]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], j + 1, &y_coords_in[cl[num_corners_out - 4]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], j + 1, &y_coords_in[cl[num_corners_out - 3]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], j, &y_coords_in[cl[num_corners_out - 2]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], j, &y_coords_in[cl[num_corners_out - 1]],
                                              z_size - 1, &z_coords_in[cl[num_corners_out - 1]]);
            }

            // z = 0

            pl[num_polygons_out++] = num_corners_out;

            cl[num_corners_out++] = i * y_size * z_size + j * z_size;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + j * z_size;
            TRANS;
            cl[num_corners_out++] = (i + 1) * y_size * z_size + (j + 1) * z_size;
            TRANS;
            cl[num_corners_out++] = i * y_size * z_size + (j + 1) * z_size;
            TRANS;

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (j == y_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], (i + 1) * y_size * z_size + ((j + 1) % (y_size - 1)) * z_size, (i + 1) * y_size * z_size + ((j + 1) % (y_size - 1) + 1) * z_size, i * y_size * z_size + ((j + 1) % (y_size - 1)) * z_size) >= angle)) || (j == y_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (j + 1) * z_size;
                LTRANS;
                my_lcl[lnum_corners_out++] = i * y_size * z_size + (j + 1) * z_size;
                LTRANS;
            }

            if ((dynamic_cast<const coDoStructuredGrid *>(mesh_in) && (i == x_size - 2 || get_angle(cl[num_corners_out - 2], cl[num_corners_out - 1], cl[num_corners_out - 3], ((i + 1) % (x_size - 1) + 1) * y_size * z_size + j * z_size, ((i + 1) % (x_size - 1) + 1) * y_size * z_size + (j + 1) * z_size, (i + 1) % (x_size - 1) * y_size * z_size + j * z_size) >= angle)) || (i == x_size - 2 && (dynamic_cast<const coDoRectilinearGrid *>(mesh_in) || dynamic_cast<const coDoUniformGrid *>(mesh_in))))
            {
                my_ll[num_lines++] = lnum_corners_out;

                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + (j + 1) * z_size;
                LTRANS;
                my_lcl[lnum_corners_out++] = (i + 1) * y_size * z_size + j * z_size;
                LTRANS;
            }

            else if (dynamic_cast<const coDoUniformGrid *>(mesh_in))
            {
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              0, &z_coords_in[cl[num_corners_out - 4]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], j, &y_coords_in[cl[num_corners_out - 3]],
                                              0, &z_coords_in[cl[num_corners_out - 3]]);
                ugrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              0, &z_coords_in[cl[num_corners_out - 2]]);
                ugrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], j + 1, &y_coords_in[cl[num_corners_out - 1]],
                                              0, &z_coords_in[cl[num_corners_out - 1]]);
            }

            else if (dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
            {
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 4]], j, &y_coords_in[cl[num_corners_out - 4]],
                                              0, &z_coords_in[cl[num_corners_out - 4]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 3]], j, &y_coords_in[cl[num_corners_out - 3]],
                                              0, &z_coords_in[cl[num_corners_out - 3]]);
                rgrid_in->getPointCoordinates(i + 1, &x_coords_in[cl[num_corners_out - 2]], j + 1, &y_coords_in[cl[num_corners_out - 2]],
                                              0, &z_coords_in[cl[num_corners_out - 2]]);
                rgrid_in->getPointCoordinates(i, &x_coords_in[cl[num_corners_out - 1]], j + 1, &y_coords_in[cl[num_corners_out - 1]],
                                              0, &z_coords_in[cl[num_corners_out - 1]]);
            }
        }
    }

    // Delete not necessary nodes

    num_points_out = x_size * y_size * z_size - 1;
    while (trans_coord[num_points_out] == TO_DELETE)
        num_points_out--;

    for (i = 0; i < num_points_out; i++)
    {
        if (trans_coord[i] == TO_DELETE)
        {
            trans_coord[i] = num_points_out;
            trans_coord[num_points_out] = i;
            num_points_out--;
            while (trans_coord[num_points_out] == TO_DELETE)
                num_points_out--;
        }
    }

    num_points_out++;
    const char *name = outPort_polySet->getObjName();
    char **dummy = new char *[1000];
    delete[] dummy;
    poly_out = new coDoPolygons(name, num_points_out, num_corners_out, num_polygons_out);
    poly_out->addAttribute("vertexOrder", "2");
    poly_out->getAddresses(&x_coords_out, &y_coords_out, &z_coords_out, &cl_out, &pl_out);

    for (i = 0; i < num_points_out; i++)
    {
        x_coords_out[i] = x_coords_in[trans_coord[i]];
        y_coords_out[i] = y_coords_in[trans_coord[i]];
        z_coords_out[i] = z_coords_in[trans_coord[i]];
    }

    for (i = 0; i < num_corners_out; i++)
    {
        cl_out[i] = trans_coord[cl[i]];
    }

    for (i = 0; i < num_polygons_out; i++)
    {
        pl_out[i] = pl[i];
    }

    outPort_polySet->setCurrentObject(poly_out);

    /****************/
    /* Feature lines  */
    /****************/

    lnum_points_out = x_size * y_size * z_size - 1;
    while (ltrans_coord[lnum_points_out] == TO_DELETE)
        lnum_points_out--;

    for (i = 0; i < lnum_points_out; i++)
    {
        if (ltrans_coord[i] == TO_DELETE)
        {
            ltrans_coord[i] = lnum_points_out;
            ltrans_coord[lnum_points_out] = i;
            lnum_points_out--;
            while (ltrans_coord[lnum_points_out] == TO_DELETE)
                lnum_points_out--;
        }
    }

    lnum_points_out++;

    lines_out = new coDoLines(outPort_bound->getObjName(), lnum_points_out, lnum_corners_out, num_lines);
    lines_out->getAddresses(&x_coords_out, &y_coords_out, &z_coords_out, &lcl, &ll);

    for (i = 0; i < lnum_points_out; i++)
    {
        x_coords_out[i] = x_coords_in[ltrans_coord[i]];
        y_coords_out[i] = y_coords_in[ltrans_coord[i]];
        z_coords_out[i] = z_coords_in[ltrans_coord[i]];
    }

    for (i = 0; i < lnum_corners_out; i++)
    {
        lcl[i] = ltrans_coord[my_lcl[i]];
    }
    for (i = 0; i < num_lines; i++)
    {
        ll[i] = my_ll[i];
    }

    outPort_bound->setCurrentObject(lines_out);

    /**********************/
    /*  Data on polygons  */
    /**********************/

    data_in[0] = p_data1->getCurrentObject();

    if (data_in[0] != NULL)
    {
        v_data_in[0] = dynamic_cast<const coDoVec3 *>(data_in[0]);
        s_data_in[0] = dynamic_cast<const coDoFloat *>(data_in[0]);
        m_data_in[0] = dynamic_cast<const coDoMat3 *>(data_in[0]);

        if (v_data_in[0])
        {
            v_data_in[0]->getAddresses(&xdata_in[0], &ydata_in[0], &zdata_in[0]);
            v_data_out[0] = new coDoVec3(outPort_dataP->getObjName(), num_points_out);
            v_data_out[0]->getAddresses(&xdata_out[0], &ydata_out[0], &zdata_out[0]);

            for (i = 0; i < num_points_out; i++)
            {
                xdata_out[0][i] = xdata_in[0][trans_coord[i]];
                ydata_out[0][i] = ydata_in[0][trans_coord[i]];
                zdata_out[0][i] = zdata_in[0][trans_coord[i]];
            }

            outPort_dataP->setCurrentObject(v_data_out[0]);
        }

        else if (s_data_in[0])
        {
            s_data_in[0]->getAddress(&sdata_in[0]);
            s_data_out[0] = new coDoFloat(outPort_dataP->getObjName(), num_points_out);
            s_data_out[0]->getAddress(&sdata_out[0]);

            for (i = 0; i < num_points_out; i++)
            {
                sdata_out[0][i] = sdata_in[0][trans_coord[i]];
            }

            outPort_dataP->setCurrentObject(s_data_out[0]);
        }

        else if (m_data_in[0])
        {
            m_data_in[0]->getAddress(&mdata_in[0]);
            m_data_out[0] = new coDoMat3(outPort_dataP->getObjName(), num_points_out);
            m_data_out[0]->getAddress(&mdata_out[0]);

            for (i = 0; i < num_points_out; i++)
            {
                int i9 = i * 9;
                int is9 = trans_coord[i] * 9;
                for (j = 0; j < 9; j++)
                    mdata_out[0][i9 + j] = mdata_in[0][is9 + j];
            }

            outPort_dataP->setCurrentObject(m_data_out[0]);
        }
    }

    /****************/
    /* Data on lines */
    /****************/

    data_in[1] = p_data1->getCurrentObject();
    if (data_in[1] != NULL)
    {
        v_data_in[1] = dynamic_cast<const coDoVec3 *>(data_in[1]);
        s_data_in[1] = dynamic_cast<const coDoFloat *>(data_in[1]);
        m_data_in[1] = dynamic_cast<const coDoMat3 *>(data_in[1]);

        if (v_data_in[1])
        {
            v_data_in[1]->getAddresses(&xdata_in[1], &ydata_in[1], &zdata_in[1]);
            v_data_out[1] = new coDoVec3(outPort_dataL->getObjName(), lnum_points_out);
            v_data_out[1]->getAddresses(&xdata_out[1], &ydata_out[1], &zdata_out[1]);

            for (i = 0; i < lnum_points_out; i++)
            {
                xdata_out[1][i] = xdata_in[1][ltrans_coord[i]];
                ydata_out[1][i] = ydata_in[1][ltrans_coord[i]];
                zdata_out[1][i] = zdata_in[1][ltrans_coord[i]];
            }

            outPort_dataL->setCurrentObject(v_data_out[1]);
        }

        else if (s_data_in[1])
        {
            s_data_in[1]->getAddress(&sdata_in[1]);
            s_data_out[1] = new coDoFloat(outPort_dataL->getObjName(), lnum_points_out);
            s_data_out[1]->getAddress(&sdata_out[1]);

            for (i = 0; i < lnum_points_out; i++)
            {
                sdata_out[1][i] = sdata_in[1][ltrans_coord[i]];
            }

            outPort_dataL->setCurrentObject(s_data_out[1]);
        }
        else if (m_data_in[1])
        {
            m_data_in[1]->getAddress(&mdata_in[1]);
            m_data_out[1] = new coDoMat3(outPort_dataL->getObjName(), lnum_points_out);
            m_data_out[1]->getAddress(&mdata_out[1]);

            for (i = 0; i < num_points_out; i++)
            {
                int i9 = i * 9;
                int is9 = ltrans_coord[i] * 9;
                for (j = 0; j < 9; j++)
                    mdata_out[1][i9 + j] = mdata_in[1][is9 + j];
            }

            outPort_dataL->setCurrentObject(m_data_out[1]);
        }
    }

    /*********/
    /* Clean  */
    /*********/

    delete[] pl;
    delete[] cl;
    delete[] my_ll;
    delete[] my_lcl;

    delete[] trans_coord;
    delete[] ltrans_coord;

    if (dynamic_cast<const coDoUniformGrid *>(mesh_in) || dynamic_cast<const coDoRectilinearGrid *>(mesh_in))
    {
        delete[] x_coords_in;
        delete[] y_coords_in;
        delete[] z_coords_in;
    }
    return CONTINUE_PIPELINE;
}

double SDomainsurface::get_angle(int v1, int v2, int v3, int v21, int v22, int v23)
{
    double n1x, n1y, n1z, n2x, n2y, n2z, l, ang;

    n1x = ((y_coords_in[v1] - y_coords_in[v2]) * (z_coords_in[v1] - z_coords_in[v3])) - ((z_coords_in[v1] - z_coords_in[v2]) * (y_coords_in[v1] - y_coords_in[v3]));
    n1y = ((z_coords_in[v1] - z_coords_in[v2]) * (x_coords_in[v1] - x_coords_in[v3])) - ((x_coords_in[v1] - x_coords_in[v2]) * (z_coords_in[v1] - z_coords_in[v3]));
    n1z = ((x_coords_in[v1] - x_coords_in[v2]) * (y_coords_in[v1] - y_coords_in[v3])) - ((y_coords_in[v1] - y_coords_in[v2]) * (x_coords_in[v1] - x_coords_in[v3]));
    l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
    n1x /= l;
    n1y /= l;
    n1z /= l;

    n2x = ((y_coords_in[v21] - y_coords_in[v22]) * (z_coords_in[v21] - z_coords_in[v23])) - ((z_coords_in[v21] - z_coords_in[v22]) * (y_coords_in[v21] - y_coords_in[v23]));
    n2y = ((z_coords_in[v21] - z_coords_in[v22]) * (x_coords_in[v21] - x_coords_in[v23])) - ((x_coords_in[v21] - x_coords_in[v22]) * (z_coords_in[v21] - z_coords_in[v23]));
    n2z = ((x_coords_in[v21] - x_coords_in[v22]) * (y_coords_in[v21] - y_coords_in[v23])) - ((y_coords_in[v21] - y_coords_in[v22]) * (x_coords_in[v21] - x_coords_in[v23]));
    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
    n2x /= l;
    n2y /= l;
    n2z /= l;
    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
    if (ang < 0)
        ang = -ang;
    ang = 1 - ang;
    //cerr << ang << endl;
    return (ang);
}

void SDomainsurface::copyAttributesToOutObj(coInputPort **input_ports,
                                            coOutputPort **output_ports, int n)
{
    int i, j;
    const coDistributedObject *in_obj;
    coDistributedObject *out_obj;
    int num_attr;
    const char **attr_n, **attr_v;

    if (n >= 2)
        j = 0;
    else
        j = n;
    if (input_ports[j] && output_ports[n])
    {
        in_obj = input_ports[j]->getCurrentObject();
        out_obj = output_ports[n]->getCurrentObject();

        if (in_obj != NULL && out_obj != NULL)
        {
            if (in_obj->getAttribute("Probe2D") == NULL)
            {
                copyAttributes(out_obj, in_obj);
            }
            else // update Probe2D attribute
            {
                num_attr = in_obj->getAllAttributes(&attr_n, &attr_v);
                for (i = 0; i < num_attr; i++)
                {
                    if (strcmp(attr_n[i], "Probe2D") != 0)
                    {
                        out_obj->addAttribute(attr_n[i], attr_v[i]);
                    }
                }
            }
            out_obj->addAttribute("Probe2D", output_ports[1]->getObjName());
        }
    }
}

//
// unstructured grid stuff
//

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void SDomainsurface::doModule(const coDistributedObject *meshIn,
                              const coDistributedObject *dataIn,
                              const char *meshOutName,
                              const char *dataOutName,
                              const char *lineOutName,
                              const char *ldataOutName,
                              coDistributedObject **meshOut,
                              coDistributedObject **dataOut,
                              coDistributedObject **lineOut,
                              coDistributedObject **ldataOut)
{
    char colorn[255];
    const char *color_attr;
    const char *dtype = NULL;
    int data_anz = 0;
    conn_list = conn_tag = elem_list = lconn_list = lelem_list = NULL;
    x_out = y_out = z_out = lx_out = ly_out = lz_out = NULL;
    *ldataOut = *meshOut = *dataOut = *lineOut = NULL;
    int num_attr;
    const char **attr_n, **attr_v;

    //////////////// only handle non-sets ////////////////////
    tmp_grid = (coDoUnstructuredGrid *)meshIn;
    if (strcmp(tmp_grid->getType(), "UNSGRD"))
    {
        Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
        return;
    }
    tmp_grid->getGridSize(&numelem, &numconn, &numcoord);
    tmp_grid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
    tmp_grid->getTypeList(&tl); // tl-Type List
    DataType = 0;
    if (dataIn)
    {
        dtype = dataIn->getType();
        if (strcmp(dtype, "USTSDT") == 0)
        {
            USIn = (coDoFloat *)dataIn;
            data_anz = USIn->getNumPoints();
            USIn->getAddress(&u_in);
            DataType = DATA_S;
            if (data_anz == numelem)
                DataType = DATA_S_E;
        }
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            UVIn = (coDoVec3 *)dataIn;
            data_anz = UVIn->getNumPoints();
            UVIn->getAddresses(&u_in, &v_in, &w_in);
            DataType = DATA_V;
            if (data_anz == numelem)
                DataType = DATA_V_E;
        }
        else
        {
            Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
            return;
        }
    }
    if (DataType && (data_anz != numcoord) && (data_anz != numelem))
    {
        if (data_anz != 0)
            Covise::sendWarning("WARNING: Data objects dimension does not match grid ones: dummy output");
        DataType = DATA_NONE;
    }
    //	set color in geometry
    if ((color_attr = tmp_grid->getAttribute("COLOR")) == NULL)
    {
        strcpy(colorn, "White");
        color_attr = colorn;
    }
    //	Is there data in the array ?
    //      sl: if data and grid are dummy, there is no reason to alarm the gentle user
    if (DataType && data_anz != 0 && (numelem == 0 || numconn == 0 || numcoord == 0))
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }
    //      If computation is for the first time or the grid has changed
    //      create adjacency information
    //	get cells_use_coord list from shared memory
    int cuc_count;
    int *cuc, *cuc_pos;
    tmp_grid->getNeighborList(&cuc_count, &cuc, &cuc_pos);
    numelem_o = numelem;
    u_out = v_out = w_out = 0;
    lu_out = lv_out = lw_out = 0;
    // Surface polygons
    surface();
    Polygons = new coDoPolygons(meshOutName, num_vert, x_out, y_out, z_out,
                                num_conn, conn_list, num_elem, elem_list);
    if (!meshIn->getAttribute("COLOR")) // sonst koennten wir COLOR
        // 2-mal schreiben (siehe unten
        // Polygons->copyAllAttributes(meshIn));
        Polygons->addAttribute("COLOR", color_attr);
    Polygons->addAttribute("vertexOrder", "2");
    *meshOut = Polygons;
    // Contour lines
    lines();
    tmp_grid->freeNeighborList();
    Lines = new coDoLines(lineOutName, lnum_vert, lx_out, ly_out, lz_out,
                          lnum_conn, lconn_list, lnum_elem, lelem_list);
    if (!meshIn->getAttribute("COLOR"))
        Lines->addAttribute("COLOR", color_attr);
    Lines->addAttribute("vertexOrder", "2");
    *lineOut = Lines;
    // Data
    if (DataType == DATA_S)
    {
        if (num_vert != 0)
            SOut = new coDoFloat(dataOutName, num_vert, u_out);
        else
            SOut = new coDoFloat(dataOutName, 0);
        *dataOut = SOut;
        if (lnum_vert != 0)
            SlinesOut = new coDoFloat(ldataOutName, lnum_vert, lu_out);
        else
            SlinesOut = new coDoFloat(ldataOutName, 0);
        *ldataOut = SlinesOut;
        delete[] u_out;
        delete[] lu_out;
    }
    else if (DataType == DATA_V)
    {
        if (num_vert != 0)
            VOut = new coDoVec3(dataOutName, num_vert, u_out, v_out, w_out);
        else
            VOut = new coDoVec3(dataOutName, 0);
        *dataOut = VOut;
        if (lnum_vert != 0)
            VlinesOut = new coDoVec3(ldataOutName, lnum_vert, lu_out, lv_out, lw_out);
        else
            VlinesOut = new coDoVec3(ldataOutName, 0);
        *ldataOut = VlinesOut;
        delete[] u_out;
        delete[] v_out;
        delete[] w_out;
        delete[] lu_out;
        delete[] lv_out;
        delete[] lw_out;
    }
    else if (DataType == DATA_S_E)
    {
        if (num_elem)
            SOut = new coDoFloat(dataOutName, num_elem, u_out);
        else
            SOut = new coDoFloat(dataOutName, 0);
        *dataOut = SOut;
        if (lnum_elem)
            SlinesOut = new coDoFloat(ldataOutName, lnum_elem, lu_out);
        else
            SlinesOut = new coDoFloat(ldataOutName, 0);
        *ldataOut = SlinesOut;
        delete[] u_out;
        delete[] lu_out;
    }
    else if (DataType == DATA_V_E)
    {
        if (num_elem)
            VOut = new coDoVec3(dataOutName, num_elem, u_out, v_out, w_out);
        else
            VOut = new coDoVec3(dataOutName, 0);
        *dataOut = VOut;
        if (lnum_elem)
            VlinesOut = new coDoVec3(ldataOutName, lnum_elem, lu_out, lv_out, lw_out);
        else
            VlinesOut = new coDoVec3(ldataOutName, 0);
        *ldataOut = VlinesOut;
        delete[] u_out;
        delete[] v_out;
        delete[] w_out;
        delete[] lu_out;
        delete[] lv_out;
        delete[] lw_out;
    }
    else
    {
        if (dataIn && strcmp(dtype, "USTSDT") == 0)
        {
            *dataOut = new coDoFloat(dataOutName, 0);
            *ldataOut = new coDoFloat(ldataOutName, 0);
        }
        else if (dataIn && strcmp(dtype, "USTVDT") == 0)
        {
            *dataOut = new coDoVec3(dataOutName, 0);
            *ldataOut = new coDoVec3(ldataOutName, 0);
        }
        else
        {
            *dataOut = NULL;
            *ldataOut = NULL;
        }
    }
    // setting of attributes
    if (meshIn->getAttribute("Probe2D") == NULL)
    {
        Polygons->copyAllAttributes(meshIn);
    }
    else // update Probe2D attribute
    {
        num_attr = (meshIn)->getAllAttributes(&attr_n, &attr_v);
        for (int i = 0; i < num_attr; i++)
        {
            if (strcmp(attr_n[i], "Probe2D") != 0)
            {
                Polygons->addAttribute(attr_n[i], attr_v[i]);
            }
        }
    }
    Polygons->addAttribute("Probe2D", dataOutName);
    Lines->copyAllAttributes(meshIn);
    if (dataOut && *dataOut)
        (*dataOut)->copyAllAttributes(dataIn);
    if (ldataOut && *ldataOut)
        (*ldataOut)->copyAllAttributes(dataIn);
    /*
   char *times=meshIn->getAttribute("TIMESTEP");
   if (times) {
    Polygons->addAttribute("TIMESTEP", times);
    Lines->addAttribute("TIMESTEP", times);
    if (dataOut)
      if(*dataOut)
   (*dataOut)->addAttribute("TIMESTEP", times);
    if (ldataOut)
      if (*ldataOut)
   (*ldataOut)->addAttribute("TIMESTEP", times);
   }
   times=meshIn->getAttribute("READ_MODULE");
   if (times) {
   Polygons->addAttribute("READ_MODULE", times);
   Lines->addAttribute("READ_MODULE", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("READ_MODULE", times);
   if (ldataOut)
   if(ldataOut)
   (*ldataOut)->addAttribute("READ_MODULE", times);
   }
   times=meshIn->getAttribute("BLOCKINFO");
   if (times) {
   Polygons->addAttribute("BLOCKINFO", times);
   Lines->addAttribute("BLOCKINFO", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("BLOCKINFO", times);
   if (ldataOut)
   if(*ldataOut)
   (*ldataOut)->addAttribute("BLOCKINFO", times);
   }
   times=meshIn->getAttribute("PART");
   if (times) {
   Polygons->addAttribute("PART", times);
   Lines->addAttribute("PART", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("PART", times);
   if (ldataOut)
   if(*ldataOut)
   (*ldataOut)->addAttribute("PART", times);
   }
   */
    delete[] x_out;
    delete[] y_out;
    delete[] z_out;
    delete[] conn_list;
    delete[] conn_tag;
    delete[] elem_list;
    delete[] lx_out;
    delete[] ly_out;
    delete[] lz_out;
    delete[] lconn_list;
    delete[] lelem_list;
    delete[] elemMap;
}

void SDomainsurface::lines()
{
    int i, j, np, n;
    int v1, v2, v3, v4, v21, v22, v23;
    float n1x, n1y, n1z, n2x, n2y, n2z, ang, l;
    lx_out = new float[num_vert + num_bar * 2];
    ly_out = new float[num_vert + num_bar * 2];
    lz_out = new float[num_vert + num_bar * 2];

    bool vertices_found_1;
    bool vertices_found_2;
    int edge;

    vector<int> temp_lconn_list;
    vector<int> temp_lelem_list;

    Polygons->computeNeighborList();

    memset(conn_tag, -1, numcoord * sizeof(int));
    lnum_vert = 0;
    lnum_conn = 0;
    lnum_elem = 0;
    // cerr <<"TEST: " <<  num_elem << "\t" << num_bar << endl;
    for (i = 0; i < num_elem; i++)
    {
        if (i == num_elem - 1)
            np = num_conn - elem_list[i];
        else
            np = elem_list[i + 1] - elem_list[i];
        switch (np)
        {
        case 3:
        {
            v1 = conn_list[elem_list[i]];
            v2 = conn_list[elem_list[i] + 1];
            v3 = conn_list[elem_list[i] + 2];
            n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
            n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
            n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
            l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
            n1x /= l;
            n1y /= l;
            n1z /= l;
            if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v1));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v2));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v1));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v2));
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v2));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v3));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v2));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v3));
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v1, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v3));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v1));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v3));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v1));
                lnum_conn++;
            }
        }
        break;
        case 4:
        {
            v1 = conn_list[elem_list[i]];
            v2 = conn_list[elem_list[i] + 1];
            v3 = conn_list[elem_list[i] + 2];
            v4 = conn_list[elem_list[i] + 3];
            n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
            n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
            n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
            l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
            n1x /= l;
            n1y /= l;
            n1z /= l;
            if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v1));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v2));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v1));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v2));
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v2));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v3));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v2));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v3));
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v3, v4)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v3));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v4));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v3));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v4));
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v4, v1)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_lu_out.push_back(u_out[i]);
                        temp_lv_out.push_back(v_out[i]);
                        temp_lw_out.push_back(w_out[i]);
                    }
                    temp_lelem_list.push_back(lnum_conn);
                    lnum_elem++;
                    temp_lconn_list.push_back(ladd_vertex(v4));
                    lnum_conn++;
                    temp_lconn_list.push_back(ladd_vertex(v1));
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_lu_out.push_back(u_out[i]);
                    temp_lv_out.push_back(v_out[i]);
                    temp_lw_out.push_back(w_out[i]);
                }
                temp_lelem_list.push_back(lnum_conn);
                lnum_elem++;
                temp_lconn_list.push_back(ladd_vertex(v4));
                lnum_conn++;
                temp_lconn_list.push_back(ladd_vertex(v1));
                lnum_conn++;
            }
        }
        break;
        default:
        {
            // Polyhedral cells
            if (np > 4)
            {
                vertices_found_1 = false;
                // Avoid degeneracies:  choose three consecutive vertices of the polygon which are different and not collinear
                for (j = 0; j < np; j++)
                {
                    if (j < np - 2)
                    {
                        v1 = conn_list[elem_list[i] + j];
                        v2 = conn_list[elem_list[i] + j + 1];
                        v3 = conn_list[elem_list[i] + j + 2];
                    }

                    else if (j == np - 2)
                    {
                        v1 = conn_list[elem_list[i] + j];
                        v2 = conn_list[elem_list[i] + j + 1];
                        v3 = conn_list[elem_list[i]];
                    }

                    else if (j == np - 1)
                    {
                        v1 = conn_list[elem_list[i] + j];
                        v2 = conn_list[elem_list[i]];
                        v3 = conn_list[elem_list[i] + 1];
                    }

                    // Assuming the vertices of the polygon are contained within a plane, calculate its normal
                    n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
                    n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
                    n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
                    l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
                    n1x /= l;
                    n1y /= l;
                    n1z /= l;

                    if (n1x != 0 || n1y != 0 || n1z != 0)
                    {
                        vertices_found_1 = true;
                        break;
                    }
                }

                if (vertices_found_1)
                {
                    // Test for each edge of the polygon!!!
                    for (edge = 0; edge < np; edge++)
                    {
                        vertices_found_2 = false;

                        // Select an edge
                        if (edge < np - 1)
                        {
                            v1 = conn_list[elem_list[i] + edge];
                            v2 = conn_list[elem_list[i] + edge + 1];
                        }

                        else if (edge == np - 1)
                        {
                            v1 = conn_list[elem_list[i] + edge];
                            v2 = conn_list[elem_list[i]];
                        }

                        if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
                        {
                            // Avoid degeneracies:  choose three consecutive vertices of the polygon which are different and not collinear
                            for (j = 0; j < np; j++)
                            {
                                if (j < np - 2)
                                {
                                    v21 = conn_list[elem_list[n] + j];
                                    v22 = conn_list[elem_list[n] + j + 1];
                                    v23 = conn_list[elem_list[n] + j + 2];
                                }

                                else if (j == np - 2)
                                {
                                    v21 = conn_list[elem_list[n] + j];
                                    v22 = conn_list[elem_list[n] + j + 1];
                                    v23 = conn_list[elem_list[n]];
                                }

                                else if (j == np - 1)
                                {
                                    v21 = conn_list[elem_list[n] + j];
                                    v22 = conn_list[elem_list[n]];
                                    v23 = conn_list[elem_list[n] + 1];
                                }

                                // Assuming the vertices of the polygon are contained within a plane, calculate its normal
                                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));

                                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                                n2x /= l;
                                n2y /= l;
                                n2z /= l;

                                if (n2x != 0 || n2y != 0 || n2z != 0)
                                {
                                    vertices_found_2 = true;
                                    break;
                                }
                            }

                            if (vertices_found_2)
                            {
                                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                                if (ang < 0)
                                    ang = -ang;
                                ang = 1 - ang;
                                if (ang > tresh)
                                {
                                    if (DataType == DATA_S_E)
                                    {
                                        temp_lu_out.push_back(u_out[i]);
                                    }
                                    else if (DataType == DATA_V_E)
                                    {
                                        temp_lu_out.push_back(u_out[i]);
                                        temp_lv_out.push_back(v_out[i]);
                                        temp_lw_out.push_back(w_out[i]);
                                    }
                                    temp_lelem_list.push_back(lnum_conn);
                                    lnum_elem++;
                                    temp_lconn_list.push_back(ladd_vertex(v1));
                                    lnum_conn++;
                                    temp_lconn_list.push_back(ladd_vertex(v2));
                                    lnum_conn++;
                                }
                            }
                        }
                        else
                        {
                            if (DataType == DATA_S_E)
                            {
                                temp_lu_out.push_back(u_out[i]);
                            }
                            else if (DataType == DATA_V_E)
                            {
                                temp_lu_out.push_back(u_out[i]);
                                temp_lv_out.push_back(v_out[i]);
                                temp_lw_out.push_back(w_out[i]);
                            }
                            temp_lelem_list.push_back(lnum_conn);
                            lnum_elem++;
                            temp_lconn_list.push_back(ladd_vertex(v1));
                            lnum_conn++;
                            temp_lconn_list.push_back(ladd_vertex(v2));
                            lnum_conn++;
                        }
                    }
                }
            }
        }
        break;
        };
    }
    for (i = 0; i < num_bar; i++)
    {
        temp_lelem_list.push_back(lnum_conn);
        if (DataType == DATA_S_E)
        {
            temp_lu_out.push_back(u_in[elemMap[i]]);
        }
        else if (DataType == DATA_V_E)
        {
            temp_lu_out.push_back(u_in[elemMap[i]]);
            temp_lv_out.push_back(v_in[elemMap[i]]);
            temp_lw_out.push_back(w_in[elemMap[i]]);
        }
        lnum_elem++;
        // entries directly from grid lists
        for (j = 0; j < 2; j++)
        {
            temp_lconn_list.push_back(lnum_vert);
            lnum_conn++;
            if (elemMap[i] != -1)
            {
                lx_out[lnum_vert] = x_in[cl[el[elemMap[i]] + j]];
                ly_out[lnum_vert] = y_in[cl[el[elemMap[i]] + j]];
                lz_out[lnum_vert] = z_in[cl[el[elemMap[i]] + j]];
                if (DataType == DATA_S)
                {
                    temp_lu_out.push_back(u_in[cl[el[elemMap[i]] + j]]);
                }
                else if (DataType == DATA_V)
                {
                    temp_lu_out.push_back(u_in[cl[el[elemMap[i]] + j]]);
                    temp_lv_out.push_back(v_in[cl[el[elemMap[i]] + j]]);
                    temp_lw_out.push_back(w_in[cl[el[elemMap[i]] + j]]);
                }
                lnum_vert++;
            }
            else
            {
                Covise::sendError("ERROR in elemMap");
            }
        }
    }

    lconn_list = new int[temp_lconn_list.size()];
    lelem_list = new int[temp_lelem_list.size()];

    lu_out = new float[temp_lu_out.size()];
    lv_out = new float[temp_lv_out.size()];
    lw_out = new float[temp_lw_out.size()];

    for (i = 0; i < temp_lconn_list.size(); i++)
    {
        lconn_list[i] = temp_lconn_list[i];
    }

    for (i = 0; i < temp_lelem_list.size(); i++)
    {
        lelem_list[i] = temp_lelem_list[i];
    }

    if (DataType == DATA_S_E)
    {
        for (i = 0; i < temp_lelem_list.size(); i++)
        {
            lu_out[i] = temp_lu_out[i];
        }
    }

    if (DataType == DATA_V_E)
    {
        for (i = 0; i < temp_lelem_list.size(); i++)
        {
            lu_out[i] = temp_lu_out[i];
            lv_out[i] = temp_lv_out[i];
            lw_out[i] = temp_lw_out[i];
        }
    }

    if (DataType == DATA_S)
    {
        for (i = 0; i < temp_lu_out.size(); i++)
        {
            lu_out[i] = temp_lu_out[i];
        }
    }

    if (DataType == DATA_V)
    {
        for (i = 0; i < temp_lu_out.size(); i++)
        {
            lu_out[i] = temp_lu_out[i];
            lv_out[i] = temp_lv_out[i];
            lw_out[i] = temp_lw_out[i];
        }
    }

    temp_lconn_list.clear();
    temp_lelem_list.clear();
    temp_lu_out.clear();
    temp_lv_out.clear();
    temp_lw_out.clear();
}

int SDomainsurface::ladd_vertex(int v)
{
    if (conn_tag[v] >= 0)
        return (conn_tag[v]);
    conn_tag[v] = lnum_vert;
    lx_out[lnum_vert] = x_out[v];
    ly_out[lnum_vert] = y_out[v];
    lz_out[lnum_vert] = z_out[v];
    if (DataType == DATA_S)
    {
        temp_lu_out.push_back(u_out[v]);
    }
    if (DataType == DATA_V)
    {
        temp_lu_out.push_back(u_out[v]);
        temp_lv_out.push_back(v_out[v]);
        temp_lw_out.push_back(w_out[v]);
    }
    lnum_vert++;
    return (lnum_vert - 1);
};

//=====================================================================
// create the surface of a domain
//=====================================================================
void SDomainsurface::surface()
{
    // int i, a, c;
    int i, j, a, c;
    int nb; // ne deleted

    bool start_vertex_set;
    bool vertices_found;

    int next_elem_index;
    int start_vertex;
    int face;
    int next_face_index;
    int node_count;
    int v1;
    int v2;
    int v3;

    float v1_x;
    float v1_y;
    float v1_z;
    float v2_x;
    float v2_y;
    float v2_z;
    float v3_x;
    float v3_y;
    float v3_z;
    float normx;
    float normy;
    float normz;

    vector<int> temp_elem_in;
    vector<int> temp_conn_in;
    vector<int> temp_vertex_list;
    vector<int> face_nodes;
    vector<int> face_polygon;
    vector<int>::iterator it;
    vector<int>::reverse_iterator rit;

    int ct_alloc;

    ct_alloc = numcoord;
    conn_tag = new int[ct_alloc];

    memset(conn_tag, -1, numcoord * sizeof(int));
    num_vert = 0;
    num_conn = 0;
    num_elem = 0;
    num_bar = 0;
    int first = 1;

    // Compute volume-center of current element
    for (i = 0; i < numelem; i++)
    {
        switch (tl[i])
        {
        case TYPE_HEXAGON:
            c = 8;
            break;

        case TYPE_TETRAHEDER:
            c = 4;
            break;

        case TYPE_PRISM:
            c = 6;
            break;

        case TYPE_PYRAMID:
            c = 5;
            break;

        case TYPE_POLYHEDRON:
        {
            /* Calculate number of vertices of the cell */
            temp_elem_in.clear();
            temp_conn_in.clear();
            temp_vertex_list.clear();

            start_vertex_set = false;

            next_elem_index = (i < numelem - 1) ? el[i + 1] : numconn;

            /* Construct DO_Polygons Element and Connectivity Lists */
            for (j = el[i]; j < next_elem_index; j++)
            {
                if (j == el[i] && start_vertex_set == false)
                {
                    start_vertex = cl[el[i]];
                    temp_elem_in.push_back(temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }

                if (j > el[i] && start_vertex_set == true)
                {
                    if (cl[j] != start_vertex)
                    {
                        temp_conn_in.push_back(cl[j]);
                    }

                    else
                    {
                        start_vertex_set = false;
                        continue;
                    }
                }

                if (j > el[i] && start_vertex_set == false)
                {
                    start_vertex = cl[j];
                    temp_elem_in.push_back(temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }
            }

            /* Construct Vertex List */
            for (j = 0; j < temp_conn_in.size(); j++)
            {
                if (temp_vertex_list.size() == 0)
                {
                    temp_vertex_list.push_back(temp_conn_in[j]);
                }

                else
                {
                    if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[j]) == temp_vertex_list.end())
                    {
                        temp_vertex_list.push_back(temp_conn_in[j]);
                    }
                }
            }

            sort(temp_vertex_list.begin(), temp_vertex_list.end());

            c = temp_vertex_list.size();
        }
        break;

        default:
            // other possible elements are 2D and so we can't compute a
            // volume-center nor can we decide where the normal
            // has to point to
            c = 0;
            break;
        }

        x_center = 0;
        y_center = 0;
        z_center = 0;

        if (tl[i] == TYPE_POLYHEDRON)
        {
            for (a = 0; a < c; a++)
            {
                x_center += x_in[temp_vertex_list[a]];
                y_center += y_in[temp_vertex_list[a]];
                z_center += z_in[temp_vertex_list[a]];
            }
        }

        else
        {
            for (a = 0; a < c; a++)
            {
                x_center += x_in[cl[el[i] + a]];
                y_center += y_in[cl[el[i] + a]];
                z_center += z_in[cl[el[i] + a]];
            }
        }

        x_center /= (float)c;
        y_center /= (float)c;
        z_center /= (float)c;

        //converting into polygons
        switch (tl[i])
        {
        case TYPE_HEXAGON:
        {

            //Computation for hexahedra
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 4]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 5]) || test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 5]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 2]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7], cl[el[i] + 6]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7]) || test(cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 7]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7], cl[el[i]]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6], cl[el[i] + 7]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6]) || test(cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 6]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6], cl[el[i] + 1]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 3]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 7]) || test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 7]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 5]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 7]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }

            if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 5]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6]) || test(cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 6]))
                {

                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 3]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 6]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                }
            }

            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 7]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
        }
        break;
        case TYPE_TETRAHEDER:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 2], cl[el[i] + 1]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 2], cl[el[i] + 1], cl[el[i] + 3]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 3], cl[el[i] + 2]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2], cl[el[i]]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
        }
        break;
        case TYPE_PRISM:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 3]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 2], cl[el[i] + 5]) || test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 5]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 1]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3], cl[el[i] + 1]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 4]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 5]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4]) || test(cl[el[i] + 2], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 5]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                }
            }
        }
        break;
        case TYPE_PYRAMID:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4], cl[el[i] + 2]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 4], cl[el[i] + 3], cl[el[i] + 2]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i] + 4], cl[el[i] + 2], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 0]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4], cl[el[i] + 3]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 4]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                    }
                    else if (DataType == DATA_V_E)
                    {
                        temp_u_out.push_back(u_in[i]);
                        temp_v_out.push_back(v_in[i]);
                        temp_w_out.push_back(w_in[i]);
                    }
                    temp_elem_list.push_back(num_conn);
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 4]))
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                    }
                    else
                    {
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                        num_conn++;
                        temp_conn_list.push_back(add_vertex(cl[el[i]]));
                        num_conn++;
                    }
                }
            }
        }
        break;
        case TYPE_POLYHEDRON:
        {
            // Test for each face
            for (face = 0; face < temp_elem_in.size(); face++)
            {
                next_face_index = (face < temp_elem_in.size() - 1) ? temp_elem_in[face + 1] : temp_conn_in.size();

                for (node_count = temp_elem_in[face]; node_count < next_face_index; node_count++)
                {
                    face_nodes.push_back(temp_conn_in[node_count]);
                }

                face_polygon = face_nodes;
                sort(face_nodes.begin(), face_nodes.end());
                vertices_found = false;

                if (tmp_grid->getNeighbor(i, face_nodes) < 0)
                {
                    // Avoid degeneracies:  choose three consecutive vertices of the polygon which are different and not collinear
                    for (j = 0; j < face_polygon.size(); j++)
                    {
                        if (j < face_polygon.size() - 2)
                        {
                            v1 = face_polygon[j];
                            v2 = face_polygon[j + 1];
                            v3 = face_polygon[j + 2];
                        }

                        else if (j == face_polygon.size() - 2)
                        {
                            v1 = face_polygon[j];
                            v2 = face_polygon[j + 1];
                            v3 = face_polygon[0];
                        }

                        else if (j == face_polygon.size() - 1)
                        {
                            v1 = face_polygon[j];
                            v2 = face_polygon[0];
                            v3 = face_polygon[1];
                        }

                        v1_x = x_in[v1];
                        v2_x = x_in[v2];
                        v3_x = x_in[v3];

                        v1_y = y_in[v1];
                        v2_y = y_in[v2];
                        v3_y = y_in[v3];

                        v1_z = z_in[v1];
                        v2_z = z_in[v2];
                        v3_z = z_in[v3];

                        normx = (v1_y - v2_y) * (v3_z - v2_z) - (v1_z - v2_z) * (v3_y - v2_y);
                        normy = (v1_z - v2_z) * (v3_x - v2_x) - (v1_x - v2_x) * (v3_z - v2_z);
                        normz = (v1_x - v2_x) * (v3_y - v2_y) - (v1_y - v2_y) * (v3_x - v2_x);

                        if (normx != 0 || normy != 0 || normz != 0)
                        {
                            vertices_found = true;
                            break;
                        }
                    }

                    if (vertices_found)
                    {
                        if (test(v1, v2, v3))
                        {
                            if (DataType == DATA_S_E)
                            {
                                temp_u_out.push_back(u_in[i]);
                            }
                            else if (DataType == DATA_V_E)
                            {
                                temp_u_out.push_back(u_in[i]);
                                temp_v_out.push_back(v_in[i]);
                                temp_w_out.push_back(w_in[i]);
                            }
                            temp_elem_list.push_back(num_conn);
                            num_elem++;

                            if (norm_check(v1, v2, v3))
                            {
                                for (it = face_polygon.begin(); it < face_polygon.end(); it++)
                                {
                                    temp_conn_list.push_back(add_vertex(*it));
                                    num_conn++;
                                }
                            }
                            else
                            {
                                for (rit = face_polygon.rbegin(); rit < face_polygon.rend(); rit++)
                                {
                                    temp_conn_list.push_back(add_vertex(*rit));
                                    num_conn++;
                                }
                            }
                        }
                    }
                }
                face_nodes.clear();
                face_polygon.clear();
            }
        }
        break;
        case TYPE_QUAD:
        {
            if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
            {
                if (DataType == DATA_S_E)
                {
                    temp_u_out.push_back(u_in[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_u_out.push_back(u_in[i]);
                    temp_v_out.push_back(v_in[i]);
                    temp_w_out.push_back(w_in[i]);
                }
                temp_elem_list.push_back(num_conn);
                num_elem++;
                temp_conn_list.push_back(add_vertex(cl[el[i]]));
                num_conn++;
                temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                num_conn++;
                temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                num_conn++;
                temp_conn_list.push_back(add_vertex(cl[el[i] + 3]));
                num_conn++;
            }
        }
        break;
        case TYPE_TRIANGLE:
        {
            if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
            {
                if (DataType == DATA_S_E)
                {
                    temp_u_out.push_back(u_in[i]);
                }
                else if (DataType == DATA_V_E)
                {
                    temp_u_out.push_back(u_in[i]);
                    temp_v_out.push_back(v_in[i]);
                    temp_w_out.push_back(w_in[i]);
                }
                temp_elem_list.push_back(num_conn);
                num_elem++;
                temp_conn_list.push_back(add_vertex(cl[el[i]]));
                num_conn++;
                temp_conn_list.push_back(add_vertex(cl[el[i] + 1]));
                num_conn++;
                temp_conn_list.push_back(add_vertex(cl[el[i] + 2]));
                num_conn++;
            }
        }
        break;
        case TYPE_BAR: // no surface representation possible
            num_bar++;
            break;
        case TYPE_POINT: // no surface/line representation possible
            break; // but do not send an error!!!!
        default:
        {
            if (first)
                Covise::sendError("ERROR: unsupported grid type detected");
            first = 0;
            //return;
        }
            //  break; Everything is either specific or default...
        }
    }
    elemMap = new int[num_bar];
    nb = 0;
    for (i = 0; i < numelem; i++)
    {
        switch (tl[i])
        {
        case TYPE_BAR:
            elemMap[nb] = i;
            nb++;
            break;
        };
    }

    conn_list = new int[temp_conn_list.size()];
    elem_list = new int[temp_elem_list.size()];

    x_out = new float[temp_x_out.size()];
    y_out = new float[temp_y_out.size()];
    z_out = new float[temp_z_out.size()];

    u_out = new float[temp_u_out.size()];
    v_out = new float[temp_v_out.size()];
    w_out = new float[temp_w_out.size()];

    for (i = 0; i < temp_conn_list.size(); i++)
    {
        conn_list[i] = temp_conn_list[i];
    }

    for (i = 0; i < temp_elem_list.size(); i++)
    {
        elem_list[i] = temp_elem_list[i];
    }

    for (i = 0; i < temp_x_out.size(); i++)
    {
        x_out[i] = temp_x_out[i];
        y_out[i] = temp_y_out[i];
        z_out[i] = temp_z_out[i];
    }

    if (DataType == DATA_S_E)
    {
        for (i = 0; i < temp_elem_list.size(); i++)
        {
            u_out[i] = temp_u_out[i];
        }
    }

    if (DataType == DATA_V_E)
    {
        for (i = 0; i < temp_elem_list.size(); i++)
        {
            u_out[i] = temp_u_out[i];
            v_out[i] = temp_v_out[i];
            w_out[i] = temp_w_out[i];
        }
    }

    if (DataType == DATA_S)
    {
        for (i = 0; i < temp_u_out.size(); i++)
        {
            u_out[i] = temp_u_out[i];
        }
    }

    if (DataType == DATA_V)
    {
        for (i = 0; i < temp_u_out.size(); i++)
        {
            u_out[i] = temp_u_out[i];
            v_out[i] = temp_v_out[i];
            w_out[i] = temp_w_out[i];
        }
    }

    temp_elem_in.clear();
    temp_conn_in.clear();
    temp_vertex_list.clear();
    temp_conn_list.clear();
    temp_elem_list.clear();
    temp_x_out.clear();
    temp_y_out.clear();
    temp_z_out.clear();
    temp_u_out.clear();
    temp_v_out.clear();
    temp_w_out.clear();

    return;
}

int SDomainsurface::add_vertex(int v)
{
    if (conn_tag[v] >= 0)
        return (conn_tag[v]);
    conn_tag[v] = num_vert;
    temp_x_out.push_back(x_in[v]);
    temp_y_out.push_back(y_in[v]);
    temp_z_out.push_back(z_in[v]);

    if (DataType == DATA_S)
    {
        temp_u_out.push_back(u_in[v]);
    }
    if (DataType == DATA_V)
    {
        temp_u_out.push_back(u_in[v]);
        temp_v_out.push_back(v_in[v]);
        temp_w_out.push_back(w_in[v]);
    }
    num_vert++;
    return (num_vert - 1);
}

////// normals
int SDomainsurface::norm_check(int v1, int v2, int v3, int /* v4 */)
{
    int r;
    float a[3], b[3], c[3], n[3];

    // compute normal of a=v2v1 and b=v2v3
    a[0] = x_in[v1] - x_in[v2];
    a[1] = y_in[v1] - y_in[v2];
    a[2] = z_in[v1] - z_in[v2];
    b[0] = x_in[v3] - x_in[v2];
    b[1] = y_in[v3] - y_in[v2];
    b[2] = z_in[v3] - z_in[v2];
    n[0] = a[1] * b[2] - b[1] * a[2];
    n[1] = a[2] * b[0] - b[2] * a[0];
    n[2] = a[0] * b[1] - b[0] * a[1];

    // compute vector from base-point to volume-center
    c[0] = x_center - x_in[v2];
    c[1] = y_center - y_in[v2];
    c[2] = z_center - z_in[v2];
    // look if normal is correct or not
    if ((c[0] * n[0] + c[1] * n[1] + c[2] * n[2]) > 0)
        r = 0;
    else
        r = 1;

    // return wether the orientation is correct (!0) or not (0)
    return (r);
}

//=====================================================================
// test if surface should be displayed
//=====================================================================
inline int SDomainsurface::test(int v1, int v2, int v3)
{
    float l, n1x, n1y, n1z;
    n1x = ((y_in[v1] - y_in[v2]) * (z_in[v1] - z_in[v3])) - ((z_in[v1] - z_in[v2]) * (y_in[v1] - y_in[v3]));
    n1y = ((z_in[v1] - z_in[v2]) * (x_in[v1] - x_in[v3])) - ((x_in[v1] - x_in[v2]) * (z_in[v1] - z_in[v3]));
    n1z = ((x_in[v1] - x_in[v2]) * (y_in[v1] - y_in[v3])) - ((y_in[v1] - y_in[v2]) * (x_in[v1] - x_in[v3]));
    l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
    if (l != 0)
    {
        n1x /= l;
        n1y /= l;
        n1z /= l;
        return ((n1x * n2x + n1y * n2y + n1z * n2z) < scalar);
    }
    else
        return 0;
}

MODULE_MAIN(Filter, SDomainsurface)
