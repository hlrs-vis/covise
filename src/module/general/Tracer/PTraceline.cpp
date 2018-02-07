/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include "PTraceline.h"
#include <math.h>

extern PTask::whatout Whatout;

const float PTraceline::TINY = 1.0e-30f;

// record index of a grid
void
PTraceline::gridAndCell::setGrid(int i)
{
    grid_ = i;
}

// get grid label
int
PTraceline::gridAndCell::whichGrid()
{
    return grid_;
}

float
PTraceline::OutputTime(float integrationTime) const
{
    return integrationTime; // x-release_time_
}

// see header
void
PTraceline::addPoint(float x, // time
                     float *y, // point coordinates
                     float *dydx, // velocity
                     int kount, // number of points up to now -> this could be removed!!!
                     int number)
{
    if (t_c_.size() < kount)
        t_c_.resize(kount);
    if (u_c_.size() < kount)
        u_c_.resize(kount);
    if (v_c_.size() < kount)
        v_c_.resize(kount);
    if (w_c_.size() < kount)
        w_c_.resize(kount);
    if (m_c_.size() < kount)
        m_c_.resize(kount);

    t_c_.push_back(x); //[kount]=x;
    u_c_.push_back(dydx[0]); //[kount]=dydx[0];
    v_c_.push_back(dydx[1]); //[kount]=dydx[1];
    w_c_.push_back(dydx[2]); //[kount]=dydx[2];
    switch (Whatout)
    {
    case V:
        m_c_.push_back(sqrt(dydx[0] * dydx[0] + dydx[1] * dydx[1] + dydx[2] * dydx[2])); //[kount]=sqrt(dydx[0]*dydx[0]+dydx[1]*dydx[1]+dydx[2]*dydx[2]);
        break;
    case VX:
        m_c_.push_back(dydx[0]); //[kount]=dydx[0];
        break;
    case VY:
        m_c_.push_back(dydx[1]); //[kount]=dydx[1];
        break;
    case VZ:
        m_c_.push_back(dydx[2]); //[kount]=dydx[2];
        break;
    case TIME:
        m_c_.push_back(OutputTime(x)); //[kount]=OutputTime(x);               //x-release_time_;
        break;
    case ID:
        m_c_.push_back((float)number); //[kount]=number;
        break;
    case V_VEC:
        if (m_c_.size() < kount * 3)
            m_c_.resize(kount * 3);
        m_c_.push_back(dydx[0]); //[3*kount+0] = dydx[0];
        m_c_.push_back(dydx[1]); //[3*kount+1] = dydx[1];
        m_c_.push_back(dydx[2]); //[3*kount+2] = dydx[2];
        break;
    default:
        // using V
        m_c_.push_back(sqrt(dydx[0] * dydx[0] + dydx[1] * dydx[1] + dydx[2] * dydx[2])); //[kount]=sqrt(dydx[0]*dydx[0]+dydx[1]*dydx[1]+dydx[2]*dydx[2]);
        break;
    }
    for (int i = 0; i < 3; i++)
    {
        if (p_c_[i].size() < kount)
            p_c_[i].resize(kount);
        p_c_[i].push_back(y[i]); //[kount]=y[i];
    }
}

extern float grid_tolerance; // see documentation

PTask::status
PTraceline::derivsForAGrid(const float *y, // point: constness violated for polygons
                           float *ydot, // velocity
                           const coDistributedObject *p_grid, // the grid
                           const coDistributedObject *p_velo, // the velocity object
                           int *cell, // cell hint
                           int search_level) // only for polygons
{
    status ret = FINISHED_DOMAIN;

    if (!p_grid || !p_velo)
        return ret;

    if (p_grid->isType("UNIGRD") && p_velo->isType("USTVDT"))
    {
        coDoUniformGrid *p_uni_grid = (coDoUniformGrid *)(p_grid);
        coDoVec3 *p_uni_velo = (coDoVec3 *)(p_velo);
        float *u[3];
        int x_s, y_s, z_s;
        p_uni_grid->getGridSize(&x_s, &y_s, &z_s);
        int nelem = p_uni_velo->getNumPoints();
        p_uni_velo->getAddresses(&u[0], &u[1], &u[2]);

        if (x_s * y_s * z_s == nelem && p_uni_grid->interpolateField(ydot, y, cell, 3, 1, u) == 0)
        {
            ret = SERVICED;
        }
    }
    else if (p_grid->isType("RCTGRD") && p_velo->isType("USTVDT"))
    {
        coDoRectilinearGrid *p_rct_grid = (coDoRectilinearGrid *)(p_grid);
        coDoVec3 *p_rct_velo = (coDoVec3 *)(p_velo);
        float *u[3];
        int x_s, y_s, z_s;
        p_rct_grid->getGridSize(&x_s, &y_s, &z_s);
        int nelem = p_rct_velo->getNumPoints();
        p_rct_velo->getAddresses(&u[0], &u[1], &u[2]);

        if (x_s * y_s * z_s == nelem && p_rct_grid->interpolateField(ydot, y, cell, 3, 1, u) == 0)
        {
            ret = SERVICED;
        }
    }
    else if (p_grid->isType("STRGRD") && p_velo->isType("USTVDT"))
    {
        coDoStructuredGrid *p_str_grid = (coDoStructuredGrid *)(p_grid);
        coDoVec3 *p_str_velo = (coDoVec3 *)(p_velo);
        float *u[3];
        int x_s, y_s, z_s;
        p_str_grid->getGridSize(&x_s, &y_s, &z_s);
        int nelem = p_str_velo->getNumPoints();
        p_str_velo->getAddresses(&u[0], &u[1], &u[2]);

        if (x_s * y_s * z_s == nelem && p_str_grid->interpolateField(ydot, y, cell, 3, 1, u) == 0)
        {
            ret = SERVICED;
        }
    }
    else if (p_grid->isType("UNSGRD") && p_velo->isType("USTVDT"))
    {
        coDoUnstructuredGrid *p_uns_grid = (coDoUnstructuredGrid *)(p_grid);
        coDoVec3 *p_uns_velo = (coDoVec3 *)(p_velo);

        // get sizes for comparison
        int nume, numv, numc;
        p_uns_grid->getGridSize(&nume, &numv, &numc);
        int numdc;
        numdc = p_uns_velo->getNumPoints();
        float *u[3]; // vector field
        p_uns_velo->getAddresses(&u[0], &u[1], &u[2]);
        if (numc == numdc && p_uns_grid->interpolateField(ydot, y, cell, 3, 1, grid_tolerance, u) == 0)
        {
            ret = SERVICED;
        }
    }
    else if (p_grid->isType("POLYGN") && p_velo->isType("USTVDT"))
    {
        const coDoPolygons *p_pol_grid = dynamic_cast<const coDoPolygons *>(p_grid);
        const coDoVec3 *p_pol_velo = dynamic_cast<const coDoVec3 *>(p_velo);
        // get sizes for comparison
        int numc;
        numc = p_pol_grid->getNumPoints();
        int numdc;
        numdc = p_pol_velo->getNumPoints();
        float *u[3]; // vector field
        p_pol_velo->getAddresses(&u[0], &u[1], &u[2]);
        float *y_cheater = const_cast<float *>(y);
        if (numc == numdc
            && p_pol_grid->interpolateField(ydot, y_cheater, cell, 3, 1,
                                            grid_tolerance, u, search_level) == 0)
        {
            ret = SERVICED;
        }
    } // dummy for other cases
    else
    {
        ydot[0] = -y[1];
        ydot[1] = y[0];
        ydot[2] = 1.0;
        ret = SERVICED;
    }

    // lf_te
    //cerr << "vel: " << ydot[0] << "  " << ydot[1] << "  " << ydot[2] << endl;

    return ret;
}

// see header
PTraceline::PTraceline(float x_ini,
                       float y_ini,
                       float z_ini,
                       std::vector<const coDistributedObject *> &grids0,
                       std::vector<const coDistributedObject *> &vels0,
                       int ts)
    : PTask(x_ini, y_ini, z_ini, grids0, vels0)
    , release_time_(0.0)
{
    ts_ = ts;
}

void
PTraceline::gridAndCell::notFound()
{
    grid_ = -1;
    cell_[0] = cell_[1] = cell_[2] = -1;
}

void
PTraceline::gridAndCell::backUp()
{
    grid_back_ = grid_;
    cell_back_[0] = cell_[0];
    cell_back_[1] = cell_[1];
    cell_back_[2] = cell_[2];
}

void
PTraceline::gridAndCell::restore()
{
    grid_ = grid_back_;
    cell_[0] = cell_back_[0];
    cell_[1] = cell_back_[1];
    cell_[2] = cell_back_[2];
}

PTraceline::gridAndCell::gridAndCell()
{
    notFound();
}

// copy constructor
PTraceline::gridAndCell::gridAndCell(const gridAndCell &cp)
{
    grid_back_ = cp.grid_back_;
    cell_back_[0] = cp.cell_back_[0];
    cell_back_[1] = cp.cell_back_[1];
    cell_back_[2] = cp.cell_back_[2];
    grid_ = cp.grid_;
    cell_[0] = cp.cell_[0];
    cell_[1] = cp.cell_[1];
    cell_[2] = cp.cell_[2];
}

PTraceline::gridAndCell &PTraceline::gridAndCell::operator=(const gridAndCell &cp)
{
    if (this == &cp)
        return *this;
    grid_back_ = cp.grid_back_;
    cell_back_[0] = cp.cell_back_[0];
    cell_back_[1] = cp.cell_back_[1];
    cell_back_[2] = cp.cell_back_[2];
    grid_ = cp.grid_;
    cell_[0] = cp.cell_[0];
    cell_[1] = cp.cell_[1];
    cell_[2] = cp.cell_[2];
    return *this;
}

float
PTraceline::getOutLength(const float *y, int kount_out_domain)
{
    int list_size = num_points();
    float xlen = y[0] - p_c_[0][list_size - 1 - kount_out_domain];
    float ylen = y[1] - p_c_[1][list_size - 1 - kount_out_domain];
    float zlen = y[2] - p_c_[2][list_size - 1 - kount_out_domain];
    float len = xlen * xlen + ylen * ylen + zlen * zlen;
    return sqrt(len);
}

float *
PTraceline::m_c_interpolate(vector<const coDistributedObject *> &,
                            vector<const coDistributedObject *> &)
{
    return m_c();
}
