/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <float.h>
#include "PTask.h"
#define NRANSI
#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
#include <math.h>

#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>

#include <algorithm>

// #include <unistd.h>
// #define _DEBUG_

inline float
FMAX(float a, float b)
{
    return ((a >= b) ? a : b);
}

inline float
FMIN(float a, float b)
{
    return ((a <= b) ? a : b);
}

void
PTask::set_label(int label)
{
    label_ = label;
}

int
PTask::get_label()
{
    return label_;
}

void
PTask::set_status(status new_stat)
{
    status_ = new_stat;
}

int
PTask::get_status()
{
    return status_;
}

// the contents of y may only be changed in the OK emergency
// state if we do not get out of the domain
PTask::status
PTask::rkqs(float *y,
            const float *dydx,
            int n,
            float *x,
            float htry,
            float eps,
            float eps_abs,
            float yscal[],
            float *hdid,
            float *hnext)
{
    // we accept overshot as a default
    int i;
    status ret;
    float errmax, h, htemp, xnew, yerr[3], ytemp[3];
    float abserr = 0.0;

    h = htry;
    for (;;)
    {
        if ((ret = rkck(y, dydx, n, *x, h, ytemp, yerr)) == FINISHED_DOMAIN && emergency_ == OK)
        {
            return ret;
        }
        errmax = 0.0;
        for (i = 0; i < n; i++)
            errmax = FMAX(errmax, fabs(yerr[i] / yscal[i]));
        for (i = 0; i < n; i++)
            abserr = FMAX(abserr, fabs(yerr[i]));
#ifdef _DEBUG_
        fprintf(stderr, "rkqs, yerr:%f %f %f\n", yerr[0], yerr[1], yerr[2]);
        fprintf(stderr, "rkqs, yscal:%f %f %f\n", yscal[0], yscal[1], yscal[2]);
        fprintf(stderr, "rkqs, h:%f errmax: %f\n", h, errmax);
#endif
        // sleep(1);
        errmax /= eps; // relative error control
        // if emergency_ is not OK we fix the time step
        // !!!!!!!!!
        if (errmax > 1.0 && emergency_ == OK && abserr > eps_abs)
        {
            htemp = (float)(SAFETY * h * pow(errmax, (float)PSHRNK));
            h = (h >= 0.0f ? FMAX(htemp, 0.1f * h) : FMIN(htemp, 0.1f * h));
            xnew = (*x) + h;
            if (xnew == *x)
            {
                //Covise::sendWarning("stepsize underflow in stepper");
                return FINISHED_POINTS;
            }
            continue;
        }
        else if (emergency_ == OK)
        {
            if (errmax > ERRCON)
                *hnext = (float)(SAFETY * h * pow(errmax, (float)PGROW));
            else
                *hnext = 5.0f * h;
        }
        else
        {
            *hnext = h;
        }
        *x += (*hdid = h);
        for (i = 0; i < n; i++)
            y[i] = ytemp[i];
        break;
    }
    return ret;
}

#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON
#undef NRANSI

#define NRANSI

// calculate status (but do not set it), checking for intermediate
// calculations (not for the last one)!!!!!
PTask::status
PTask::rkck(const float *y,
            const float *dydx,
            int n,
            float x,
            float h,
            float yout[],
            float yerr[])
{
    int i;
    status ctrl1 = SERVICED;
    status ctrl2 = SERVICED;
    status ctrl3 = SERVICED;
    status ctrl4 = SERVICED;
    status ctrl5 = SERVICED;
    const float a2 = 0.2f, a3 = 0.3f, a4 = 0.6f, a5 = 1.0f, a6 = 0.875f, b21 = 0.2f,
                b31 = 3.0f / 40.0f, b32 = 9.0f / 40.0f, b41 = 0.3f, b42 = -0.9f, b43 = 1.2f,
                b51 = -11.0f / 54.0f, b52 = 2.5f, b53 = -70.0f / 27.0f, b54 = 35.0f / 27.0f,
                b61 = 1631.0f / 55296.0f, b62 = 175.0f / 512.0f, b63 = 575.0f / 13824.0f,
                b64 = 44275.0f / 110592.0f, b65 = 253.0f / 4096.0f, c1 = 37.0f / 378.0f,
                c3 = 250.0f / 621.0f, c4 = 125.0f / 594.0f, c6 = 512.0f / 1771.0f,
                dc5 = -277.00f / 14336.0f;
    const float dc1 = c1 - 2825.0f / 27648.0f, dc3 = c3 - 18575.0f / 48384.0f,
                dc4 = c4 - 13525.0f / 55296.0f, dc6 = c6 - 0.25f;
    float ak2[MAX_NUM_OF_VARS], ak3[MAX_NUM_OF_VARS], ak4[MAX_NUM_OF_VARS], ak5[MAX_NUM_OF_VARS], ak6[MAX_NUM_OF_VARS], ytemp[MAX_NUM_OF_VARS];

    for (i = 0; i < n; i++)
        ytemp[i] = y[i] + b21 * h * dydx[i];
    ctrl1 = derivs(x + a2 * h, ytemp, ak2, 0, -1);
    if (ctrl1 == FINISHED_DOMAIN && emergency_ == OK)
    {
        return FINISHED_DOMAIN;
    }
    else if (ctrl1 == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN)
    {
        // ak2 has no meaning in this case, so we give it a reasonable one
        for (i = 0; i < n; i++)
        {
            ak2[i] = dydx[i];
        }
    }
    fillIntermediate(0, x + a2 * h, ytemp, ak2, n, ctrl1);

    for (i = 0; i < n; i++)
        ytemp[i] = y[i] + h * (b31 * dydx[i] + b32 * ak2[i]);
    ctrl2 = derivs(x + a3 * h, ytemp, ak3, 0, -1);
    if (ctrl2 == FINISHED_DOMAIN && emergency_ == OK)
    {
        return FINISHED_DOMAIN;
    }
    else if (ctrl2 == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN)
    {
        // ak3 has no meaning in this case, so we give it a reasonable one
        for (i = 0; i < n; i++)
        {
            ak3[i] = ak2[i];
        }
    }
    fillIntermediate(1, x + a3 * h, ytemp, ak3, n, ctrl2);

    for (i = 0; i < n; i++)
        ytemp[i] = y[i] + h * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    ctrl3 = derivs(x + a4 * h, ytemp, ak4, 0, -1);
    if (ctrl3 == FINISHED_DOMAIN && emergency_ == OK)
    {
        return FINISHED_DOMAIN;
    }
    else if (ctrl3 == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN)
    {
        // ak4 has no meaning in this case, so we give it a reasonable one
        for (i = 0; i < n; i++)
        {
            ak4[i] = ak3[i];
        }
    }
    fillIntermediate(2, x + a4 * h, ytemp, ak4, n, ctrl3);

    for (i = 0; i < n; i++)
        ytemp[i] = y[i] + h * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
    ctrl4 = derivs(x + a5 * h, ytemp, ak5, 0, -1);
    if (ctrl4 == FINISHED_DOMAIN && emergency_ == OK)
    {
        return FINISHED_DOMAIN;
    }
    else if (ctrl4 == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN)
    {
        // ak5 has no meaning in this case, so we give it a reasonable one
        for (i = 0; i < n; i++)
        {
            ak5[i] = ak4[i];
        }
    }

    for (i = 0; i < n; i++)
        ytemp[i] = y[i] + h * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
    ctrl5 = derivs(x + a6 * h, ytemp, ak6, 0, -1);
    if (ctrl5 == FINISHED_DOMAIN && emergency_ == OK)
    {
        return FINISHED_DOMAIN;
    }
    else if (ctrl5 == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN)
    {
        // ak6 has no meaning in this case, so we give it a reasonable one
        for (i = 0; i < n; i++)
        {
            ak6[i] = ak4[i]; // not ak5, look at the value of a2, a3, a4, a5
        }
    }
    fillIntermediate(3, x + a6 * h, ytemp, ak6, n, ctrl4);

    for (i = 0; i < n; i++)
        yout[i] = y[i] + h * (c1 * dydx[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i]);
    for (i = 0; i < n; i++)
        yerr[i] = h * (dc1 * dydx[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i]);

    return SERVICED;
}

#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software m'5M. */
/* (C) Copr. 1986-92 Numerical Recipes Software m'5M. */

float
PTask::suggestInitialH(const int *cell,
                       const coDistributedObject *p_grid,
                       const coDistributedObject *p_velo)
{
    grid_methods::BoundBox bbox;
    float len;
    float vel = 1.0;
    int v_l[8]; // ASSUME THERE ARE AT MOST 8 NODES PER ELEMENT!!!!!!!!!!!!!!

    if (p_grid->isType("UNIGRD"))
    {
        float dx, dy, dz;
        coDoUniformGrid *p_uni_grid = (coDoUniformGrid *)(p_grid);
// bounding box
        p_uni_grid->getDelta(&dx, &dy, &dz);
        bbox.x_min_ = 0.0;
        bbox.y_min_ = 0.0;
        bbox.z_min_ = 0.0;
        bbox.x_max_ = dx;
        bbox.y_max_ = dy;
        bbox.z_max_ = dz;
        // velocity
        if (p_velo)
        {
            coDoVec3 *p_uni_velo = (coDoVec3 *)(p_velo);
            int x_s, y_s, z_s;
            float *u, *v, *w;
            p_uni_grid->getGridSize(&x_s, &y_s, &z_s);
            p_uni_velo->getAddresses(&u, &v, &w);
            v_l[0] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2];
            v_l[1] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2] + 1;
            v_l[2] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
            v_l[3] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
            v_l[4] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2];
            v_l[5] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2] + 1;
            v_l[6] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
            v_l[7] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
            vel = grid_methods::getMaxVel(8, v_l, u, v, w);
        }
    }
    else if (p_grid->isType("RCTGRD"))
    {
        float *x_in, *y_in, *z_in;
        coDoRectilinearGrid *p_rct_grid = (coDoRectilinearGrid *)(p_grid);
        // bounding box
        p_rct_grid->getAddresses(&x_in, &y_in, &z_in);
        bbox.x_min_ = x_in[cell[0]];
        bbox.y_min_ = y_in[cell[0]];
        bbox.z_min_ = z_in[cell[0]];
        bbox.x_max_ = x_in[cell[0] + 1];
        bbox.y_max_ = y_in[cell[0] + 1];
        bbox.z_max_ = z_in[cell[0] + 1];
        // velocity
        if (p_velo)
        {
            coDoVec3 *p_uni_velo = (coDoVec3 *)(p_velo);
            int x_s, y_s, z_s;
            float *u, *v, *w;
            p_rct_grid->getGridSize(&x_s, &y_s, &z_s);
            p_uni_velo->getAddresses(&u, &v, &w);
            v_l[0] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2];
            v_l[1] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2] + 1;
            v_l[2] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
            v_l[3] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
            v_l[4] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2];
            v_l[5] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2] + 1;
            v_l[6] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
            v_l[7] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
            vel = grid_methods::getMaxVel(8, v_l, u, v, w);
        }
    }
    else if (p_grid->isType("STRGRD"))
    {
        coDoStructuredGrid *p_str_grid = (coDoStructuredGrid *)(p_grid);
        float *x_in, *y_in, *z_in;
        p_str_grid->getAddresses(&x_in, &y_in, &z_in);
        int x_s, y_s, z_s;
        p_str_grid->getGridSize(&x_s, &y_s, &z_s);
        v_l[0] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2];
        v_l[1] = cell[0] * y_s * z_s + cell[1] * z_s + cell[2] + 1;
        v_l[2] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
        v_l[3] = cell[0] * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
        v_l[4] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2];
        v_l[5] = (cell[0] + 1) * y_s * z_s + cell[1] * z_s + cell[2] + 1;
        v_l[6] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2];
        v_l[7] = (cell[0] + 1) * y_s * z_s + (cell[1] + 1) * z_s + cell[2] + 1;
        grid_methods::getBoundBox(bbox, 8, v_l, x_in, y_in, z_in);
        // velocity
        if (p_velo)
        {
            coDoVec3 *p_uni_velo = (coDoVec3 *)(p_velo);
            float *u, *v, *w;
            p_uni_velo->getAddresses(&u, &v, &w);
            vel = grid_methods::getMaxVel(8, v_l, u, v, w);
        }
    }
    else if (p_grid->isType("UNSGRD"))
    {
        //    fprintf(stderr,"No h-hint for unstructured grids.\n");
        coDoUnstructuredGrid *p_uns_grid = (coDoUnstructuredGrid *)(p_grid);
        int *elem, *conn, *tl;
        float *x_in, *y_in, *z_in;
        p_uns_grid->getAddresses(&elem, &conn, &x_in, &y_in, &z_in);
        p_uns_grid->getTypeList(&tl);
        /*grid_methods::getBoundBox(bbox,UnstructuredGrid_Num_Nodes[tl[*cell]],
      conn+elem[*cell],x_in,z_in,y_in);*/ // Why are the coordinates given in this unusual order????

        bool start_vertex_set;
        int i;
        //int current_cell;
        int next_elem_index;
        int start_vertex;
        int numelem;
        int numconn;
        int numcoord;

        vector<int> temp_elem_in;
        vector<int> temp_conn_in;
        vector<int> temp_vertex_list;

        // Polyhedral cells
        if (tl[*cell] == TYPE_POLYHEDRON)
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Note:  the number of vertices of a polyhedral cell is not explicitly given; getBoundBox   //
            // requires this information, therfore it is necessary to calculate this number.  A naive      //
            // approach is to repeat here the procedure found in DO_UnstructuredGrid::testACell.     //
            // Although inefficient, this solution does not imply major modifications to the                   //
            // PTask::suggestInitialH function.                                                                                      //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            start_vertex_set = false;
            //current_cell = *cell;
            p_uns_grid->getGridSize(&numelem, &numconn, &numcoord);
            next_elem_index = (*cell < numelem) ? elem[*cell + 1] : numconn;

            /* Construct DO_Polygons Element and Connectivity Lists */
            for (i = elem[*cell]; i < next_elem_index; i++)
            {
                if (i == elem[*cell] && start_vertex_set == false)
                {
                    start_vertex = conn[elem[*cell]];
                    temp_elem_in.push_back((int)temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }

                if (i > elem[*cell] && start_vertex_set == true)
                {
                    if (conn[i] != start_vertex)
                    {
                        temp_conn_in.push_back(conn[i]);
                    }
                    else
                    {
                        start_vertex_set = false;
                        continue;
                    }
                }

                if (i > elem[*cell] && start_vertex_set == false)
                {
                    start_vertex = conn[i];
                    temp_elem_in.push_back((int)temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }
            }

            /* Construct Vertex List */
            for (i = 0; i < temp_conn_in.size(); i++)
            {
                if (temp_vertex_list.size() == 0)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }
                else
                {
                    if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                    {
                        temp_vertex_list.push_back(temp_conn_in[i]);
                    }
                }
            }

            std::sort(temp_vertex_list.begin(), temp_vertex_list.end());
            grid_methods::getBoundBox(bbox, (int)temp_vertex_list.size(), &temp_vertex_list[0], x_in, y_in, z_in);
            //       std::cout << "grid_methods::getBoundBox function was called for a polyhedral cell" << std::endl;
        }

        // Standard cells
        else
        {
//       std::cout << "grid_methods::getBoundBox function was called for a standard cell in PTask::suggestInitialH" << std::endl;
            grid_methods::getBoundBox(bbox, UnstructuredGrid_Num_Nodes[tl[*cell]], conn + elem[*cell], x_in, y_in, z_in);
        }

        // velocity
        if (p_velo)
        {
            coDoVec3 *p_uni_velo = (coDoVec3 *)(p_velo);
            float *u, *v, *w;
            p_uni_velo->getAddresses(&u, &v, &w);
            //       vel = grid_methods::getMaxVel(UnstructuredGrid_Num_Nodes[tl[*cell]],
            //       conn+elem[*cell],u,v,w);

            // Polyhedral cells
            if (tl[*cell] == TYPE_POLYHEDRON)
            {
                vel = grid_methods::getMaxVel((int)temp_vertex_list.size(), &temp_vertex_list[0], u, v, w);
            }

            // Standard cells
            else
            {
                vel = grid_methods::getMaxVel(UnstructuredGrid_Num_Nodes[tl[*cell]], conn + elem[*cell], u, v, w);
            }
        }
    }

    else if (p_grid->isType("POLYGN"))
    {
        //       fprintf(stderr,"No h-hint for unstructured grids.\n");
        const coDoPolygons *p_pol_grid = dynamic_cast<const coDoPolygons *>(p_grid);
        int *elem, *conn;
        float *x_in, *y_in, *z_in;
        p_pol_grid->getAddresses(&x_in, &y_in, &z_in, &conn, &elem);
        int numNodesPol;
        int numPol = p_pol_grid->getNumPolygons();
        if (*cell < numPol - 1)
        {
            numNodesPol = elem[*cell + 1] - elem[*cell];
        }
        else
        {
            int numVert = p_pol_grid->getNumVertices();
            numNodesPol = numVert - elem[*cell];
        }

        grid_methods::getBoundBox(bbox, numNodesPol,
                                  conn + elem[*cell], x_in, z_in, y_in);
        // velocity
        if (p_velo)
        {
            const coDoVec3 *p_uni_velo = dynamic_cast<const coDoVec3 *>(p_velo);
            float *u, *v, *w;
            p_uni_velo->getAddresses(&u, &v, &w);
            vel = grid_methods::getMaxVel(numNodesPol,
                                          conn + elem[*cell], u, v, w);
        }
    }
    len = bbox.length();
    return (vel != 0.0f) ? (len / vel) : 0.0f;
}

PTask::PTask(int label)
{
    grids0_ = 0;
    vels0_ = 0;
    status_ = NOT_SERVICED;
    label_ = label;
    emergency_ = OK;
}

PTask::PTask(real x_ini,
             real y_ini,
             real z_ini,
             std::vector<const coDistributedObject *> &grids0,
             std::vector<const coDistributedObject *> &vels0)
{
    grids0_ = &grids0;
    vels0_ = &vels0;
    status_ = NOT_SERVICED;
    ini_point_[0] = x_ini;
    ini_point_[1] = y_ini;
    ini_point_[2] = z_ini;
    emergency_ = OK;
}

void
PTask::fillIntermediate(int i,
                        float time,
                        float *var,
                        float *var_dot,
                        int n,
                        status ctrl)
{
    intermediate_[i].time_ = time;
    intermediate_[i].status_ = ctrl;
    intermediate_[i].var_.clear();
    intermediate_[i].var_dot_.clear();
    for (int no_var = 0; no_var < n; ++no_var)
    {
        intermediate_[i].var_.push_back(var[no_var]);
        intermediate_[i].var_dot_.push_back(var_dot[no_var]);
    }
}

float
PTask::interpolateFieldInGrid(const coDistributedObject *grid,
                              const coDistributedObject *field,
                              const int *cell,
                              float x, float y, float z)
{
    float point[3];
    point[0] = x;
    point[1] = y;
    point[2] = z;
    float value = FLT_MAX;
    float *data;
    int datalen[3] = { 1, 1, 1 };
    if (field->isType("USTSDT"))
    {
        coDoFloat *udata = (coDoFloat *)field;
        datalen[0] = udata->getNumPoints();
        udata->getAddress(&data);
    }
    else
    {
        return value;
    }
    // arrays needed by methods in grid_methods
    if (grid->isType("UNSGRD"))
    {
        coDoUnstructuredGrid *ugrid = (coDoUnstructuredGrid *)grid;
        int ne, nc, np;
        ugrid->getGridSize(&ne, &nc, &np);
        if (np != datalen[0])
            return value;
        int cell_copy(*cell);
        int diag = ugrid->interpolateField(&value, point, &cell_copy,
                                           1, 1, 1.0e10, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("STRGRD"))
    {
        coDoStructuredGrid *sgrid = (coDoStructuredGrid *)grid;
        int grid_size[3];
        sgrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = sgrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("UNIGRD"))
    {
        coDoUniformGrid *ugrid = (coDoUniformGrid *)grid;
        int grid_size[3];
        ugrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = ugrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("RCTGRD"))
    {
        coDoRectilinearGrid *rgrid = (coDoRectilinearGrid *)grid;
        int grid_size[3];
        rgrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = rgrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    return value;
}

float
PTask::mapFieldInGrid(const coDistributedObject *grid,
                      const coDistributedObject *field,
                      const int *cell,
                      float x, float y, float z)
{
    float point[3];
    point[0] = x;
    point[1] = y;
    point[2] = z;
    float value = FLT_MAX;
    float *data;
    int datalen[3] = { 1, 1, 1 };
    if (field->isType("USTSDT"))
    {
        coDoFloat *udata = (coDoFloat *)field;
        datalen[0] = udata->getNumPoints();
        udata->getAddress(&data);
    }
    else
    {
        return value;
    }
    // arrays needed by methods in grid_methods
    if (grid->isType("UNSGRD"))
    {
        coDoUnstructuredGrid *ugrid = (coDoUnstructuredGrid *)grid;
        int ne, nc, np;
        ugrid->getGridSize(&ne, &nc, &np);
        if (np != datalen[0])
            return value;
        int cell_copy(*cell);
        int diag = ugrid->mapScalarField(&value, point, &cell_copy,
                                         1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("STRGRD"))
    {
        coDoStructuredGrid *sgrid = (coDoStructuredGrid *)grid;
        int grid_size[3];
        sgrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = sgrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("UNIGRD"))
    {
        coDoUniformGrid *ugrid = (coDoUniformGrid *)grid;
        int grid_size[3];
        ugrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = ugrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    else if (grid->isType("RCTGRD"))
    {
        coDoRectilinearGrid *rgrid = (coDoRectilinearGrid *)grid;
        int grid_size[3];
        rgrid->getGridSize(grid_size, grid_size + 1, grid_size + 2);
        if (datalen[0] != grid_size[0]
            || datalen[1] != grid_size[1]
            || datalen[2] != grid_size[2])
        {
            return FLT_MAX;
        }
        int cell_str[3];
        cell_str[0] = cell[0];
        cell_str[1] = cell[1];
        cell_str[2] = cell[2];
        int diag = rgrid->interpolateField(&value, point, cell_str, 1, 1, &data);
        if (diag != 0)
            return FLT_MAX;
    }
    return value;
}
