/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// The use of this class has been abandoned...

#include "PPathlineStatNoControl.h"
#include <limits.h>
#include <float.h>
#include <do/coDoLines.h>
#include <do/coDoData.h>

extern int numOfAnimationSteps;
extern float stepDuration;

PPathlineStatNoControl::PPathlineStatNoControl(real x_ini,
                                               real y_ini,
                                               real z_ini,
                                               std::vector<const coDistributedObject *> &grids0,
                                               std::vector<const coDistributedObject *> &vels0, int number)
    : PPathlineStat(x_ini, y_ini, z_ini, grids0, vels0, number)
{
}

void
PPathlineStatNoControl::fillInterpolation(std::vector<int> &)
{
}

void
PPathlineStatNoControl::pathUpToTime(
    const char *line_name_time_traj,
    const char *mag_name_time_traj,
    int tick,
    std::vector<int> &interpolation,
    coDistributedObject **line,
    coDistributedObject **magn)
{
    // copy all points from x_c_, y_c, z_c_, m_c_
    // up to interpolation[tick] (inclusive).
    std::vector<float> x_l;
    std::vector<float> y_l;
    std::vector<float> z_l;
    std::vector<float> m_l;
    int i;
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *m_in = m_c();

    for (i = 0; i <= interpolation[tick]; ++i)
    {
        x_l.push_back(x_in[i]);
        y_l.push_back(y_in[i]);
        z_l.push_back(z_in[i]);
        m_l.push_back(m_in[i]);
    }
    int zero = 0;
    std::vector<int> corner_list;
    for (i = 0; i < x_l.size(); ++i)
    {
        corner_list.push_back(i);
    }
    *line = new coDoLines(line_name_time_traj, (int)x_l.size(), &x_l[0], &y_l[0], &z_l[0], (int)x_l.size(), &corner_list[0], 1, &zero);

    *magn = new coDoFloat(mag_name_time_traj, (int)m_l.size(), &m_l[0]);
}

void PPathlineStatNoControl::fillPointAtTime(std::vector<float> &points_x,
                                             std::vector<float> &points_y,
                                             std::vector<float> &points_z,
                                             std::vector<float> &mag,
                                             int tick,
                                             std::vector<int> &)
{
    if (tick >= num_points())
    {
        return;
    }
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *m_in = m_c();
    points_x.push_back(x_in[tick]);
    points_y.push_back(y_in[tick]);
    points_z.push_back(z_in[tick]);
    mag.push_back(m_in[tick]);
}

extern int search_level_polygons;

void
PPathlineStatNoControl::Solve(float, float)
{
    float x = 0.0;
    float y[3], dydx[3], ytemp[3], yerr[3];
    int i, kount = 0;
    status problems;

    for (i = 0; i < 3; i++)
        y[i] = ini_point_[i];
    problems = derivs(x, y, dydx, 0, search_level_polygons);
    if (problems == FINISHED_DOMAIN)
    {
        set_status(FINISHED_DOMAIN);
        return;
    }
    else
    {
        previousCell_0_.backUp();
        addPoint(x, y, dydx, kount, number_);
        ++kount;
    }
    while (1)
    {
        problems = rkck(y, dydx, 3, x, stepDuration, ytemp, yerr);
        // Test  with Euler...
        /*
            ytemp[0] = y[0] + dydx[0]*stepDuration;
            ytemp[1] = y[1] + dydx[1]*stepDuration;
            ytemp[2] = y[2] + dydx[2]*stepDuration;
      */

        if (problems == FINISHED_DOMAIN)
        {
            set_status(FINISHED_DOMAIN);
            break;
        }
        for (i = 0; i < 3; i++)
            y[i] = ytemp[i];
        x += stepDuration;
        previousCell_0_.backUp();
        problems = derivs(x, y, dydx, 1, -1);
        if (problems == FINISHED_DOMAIN)
        {
            set_status(FINISHED_DOMAIN);
            break;
        }
        addPoint(x, y, dydx, kount, number_);
        ++kount;
        if (kount >= numOfAnimationSteps)
        {
            break;
        }
    }
}
