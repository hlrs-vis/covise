/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PPathlineStat.h"
#include <climits>
#include <cfloat>
#include <cmath>

#include <do/coDoLines.h>
#include <do/coDoData.h>

extern int numOfAnimationSteps;
extern float stepDuration;
extern string speciesAttr;

PPathlineStat::PPathlineStat(real x_ini,
                             real y_ini,
                             real z_ini,
                             std::vector<const coDistributedObject *> &grids0,
                             std::vector<const coDistributedObject *> &vels0,
                             int number)
    : PStreamline(x_ini, y_ini, z_ini, FLT_MAX, INT_MAX, grids0, vels0, 1, number)
{
    next_tick_ = 1;
    next_time_ = stepDuration;
}

// decide whether integration should be interrupted
int
PPathlineStat::interruptIntegration(const float *,
                                    int,
                                    float,
                                    float time)
{
    int ret = 0;
    if (time >= (numOfAnimationSteps - 1) * stepDuration)
    {
        set_status(FINISHED_TIME);
        ret = 1;
    }
    return ret;
}

// fill a list of marks used for locating the correct time interval in the output
void
PPathlineStat::fillInterpolation(std::vector<int> &interpolation)
{
    interpolation.clear();
    if (num_points() == 0)
        return;
    int tick;
    int pointer = 0;
    float *times = t_c();
    for (tick = 0; tick < numOfAnimationSteps; ++tick)
    {
        while (pointer < num_points() - 1)
        {
            // check the interval pointed to by pointer
            if (times[pointer] <= tick * stepDuration
                && times[pointer + 1] >= tick * stepDuration)
            {
                break;
            }
            else
            {
                ++pointer;
            }
        }
        interpolation.push_back(pointer);
    }
}

// used when comparing floats
const float PPathlineStat::RELATIVE_TIME_TOLERANCE = 1e-6f;

// compute growing lines out of the integrated data
void
PPathlineStat::pathUpToTime(
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
    float *t_in = t_c();

    for (i = 0; i <= interpolation[tick]; ++i)
    {
        x_l.push_back(x_in[i]);
        y_l.push_back(y_in[i]);
        z_l.push_back(z_in[i]);
        m_l.push_back(m_in[i]);
    }
    // it is still numerically possible that
    // we have to add also point interpolation[tick]+1
    if (interpolation[tick] < num_points() - 1)
    {
        if (fabs(t_in[interpolation[tick] + 1] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
        {
            int p = interpolation[tick] + 1;
            x_l.push_back(x_in[p]);
            y_l.push_back(y_in[p]);
            z_l.push_back(z_in[p]);
            m_l.push_back(m_in[p]);
        }
    }
    int zero = 0;
    std::vector<int> corner_list;
    for (i = 0; i < x_l.size(); ++i)
    {
        corner_list.push_back(i);
    }
    *line = new coDoLines(line_name_time_traj, (int)x_l.size(), &x_l[0], &y_l[0], &z_l[0],
                          (int)x_l.size(), &corner_list[0], 1, &zero);
    *magn = new coDoFloat(mag_name_time_traj, (int)m_l.size(), &m_l[0]);
    if (speciesAttr.length() > 0)
        (*magn)->addAttribute("SPECIES", speciesAttr.c_str());
}

void
PPathlineStat::h_Reduce(float *h)
{
    // add *h to the last added time and
    // if greater than time for the next
    // tick, reduce it so that the addition
    // coincides with this time
    if (t_c_[num_points() - 1] + (*h) >= next_time_)
    {
        (*h) = next_time_ - t_c_[num_points() - 1];
        ++next_tick_;
        next_time_ = next_tick_ * stepDuration;
    }
}

// decrease objective next_tick_ if the stepper reduced the time
// integration step below our expectation
void
PPathlineStat::ammendNextTime(float h, float h_pre_rkqs, float h_old)
{
    if (h_pre_rkqs < h_old
        && h_pre_rkqs > h)
    {
        --next_tick_;
        next_time_ = next_tick_ * stepDuration;
    }
}

// compute moving data out of the integrated data
void PPathlineStat::fillPointAtTime(std::vector<float> &points_x,
                                    std::vector<float> &points_y,
                                    std::vector<float> &points_z,
                                    std::vector<float> &mag,
                                    int tick,
                                    std::vector<int> &interpolation)
{
    if (!(num_points() > 0
          && interpolation[tick] < num_points() - 1))
    {
        return;
    }
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *m_in = m_c();
    float *t_in = t_c();
    if (fabs(t_in[interpolation[tick]] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        int p = interpolation[tick];
        points_x.push_back(x_in[p]);
        points_y.push_back(y_in[p]);
        points_z.push_back(z_in[p]);
        mag.push_back(m_in[p]);
    }
    else if (fabs(t_in[interpolation[tick] + 1] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        int p = interpolation[tick] + 1;
        points_x.push_back(x_in[p]);
        points_y.push_back(y_in[p]);
        points_z.push_back(z_in[p]);
        mag.push_back(m_in[p]);
    }
}

int
PPathlineStat::numVertices(int tick,
                           std::vector<int> &interpolation)
{
    if (num_points() == 0)
    {
        return 0;
    }
    if (interpolation[tick] >= num_points() - 1)
        return num_points();
    int ret = 0;
    float *t_in = t_c();
    if (t_in[interpolation[tick]] - tick * stepDuration <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        ret = interpolation[tick] + 1;
    }
    if (fabs(t_in[interpolation[tick] + 1] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        ret = interpolation[tick] + 1 + 1;
    }
    return ret;
}

void
PPathlineStat::fillLineResultsAtTime(float *x_line,
                                     float *y_line,
                                     float *z_line,
                                     float *m_array,
                                     int accum_vertex,
                                     int tick,
                                     std::vector<int> &interpolation)
{
    if (num_points() == 0)
    {
        return;
    }
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *m_in = m_c();
    float *t_in = t_c();
    // fill arrays
    memcpy(x_line + accum_vertex, x_in, interpolation[tick] * sizeof(float));
    memcpy(y_line + accum_vertex, y_in, interpolation[tick] * sizeof(float));
    memcpy(z_line + accum_vertex, z_in, interpolation[tick] * sizeof(float));
    memcpy(m_array + accum_vertex, m_in, interpolation[tick] * sizeof(float));

    if (interpolation[tick] >= num_points() - 1)
    {
        x_line[accum_vertex + num_points() - 1] = x_in[num_points() - 1];
        y_line[accum_vertex + num_points() - 1] = y_in[num_points() - 1];
        z_line[accum_vertex + num_points() - 1] = z_in[num_points() - 1];
        m_array[accum_vertex + num_points() - 1] = m_in[num_points() - 1];
        return;
    }

    if (t_in[interpolation[tick]] - tick * stepDuration <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        x_line[accum_vertex + interpolation[tick]] = x_in[interpolation[tick]];
        y_line[accum_vertex + interpolation[tick]] = y_in[interpolation[tick]];
        z_line[accum_vertex + interpolation[tick]] = z_in[interpolation[tick]];
        m_array[accum_vertex + interpolation[tick]] = m_in[interpolation[tick]];
    }
    if (fabs(t_in[interpolation[tick] + 1] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        x_line[accum_vertex + interpolation[tick] + 1] = x_in[interpolation[tick] + 1];
        y_line[accum_vertex + interpolation[tick] + 1] = y_in[interpolation[tick] + 1];
        z_line[accum_vertex + interpolation[tick] + 1] = z_in[interpolation[tick] + 1];
        m_array[accum_vertex + interpolation[tick] + 1] = m_in[interpolation[tick] + 1];
    }
}

void
PPathlineStat::fillLineResultsAtTime(float *x_line,
                                     float *y_line,
                                     float *z_line,
                                     float *m_arrayU,
                                     float *m_arrayV,
                                     float *m_arrayW,
                                     int accum_vertex,
                                     int tick,
                                     std::vector<int> &interpolation)
{
    if (num_points() == 0)
    {
        return;
    }
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *u_in = u_c();
    float *v_in = v_c();
    float *w_in = w_c();
    //float *m_in=m_c();
    float *t_in = t_c();
    // fill arrays
    memcpy(x_line + accum_vertex, x_in, interpolation[tick] * sizeof(float));
    memcpy(y_line + accum_vertex, y_in, interpolation[tick] * sizeof(float));
    memcpy(z_line + accum_vertex, z_in, interpolation[tick] * sizeof(float));
    //memcpy(m_array+accum_vertex,m_in,interpolation[tick]*sizeof(float));
    memcpy(m_arrayU + accum_vertex, u_in, interpolation[tick] * sizeof(float));
    memcpy(m_arrayV + accum_vertex, v_in, interpolation[tick] * sizeof(float));
    memcpy(m_arrayW + accum_vertex, w_in, interpolation[tick] * sizeof(float));

    if (interpolation[tick] >= num_points() - 1)
    {
        x_line[accum_vertex + num_points() - 1] = x_in[num_points() - 1];
        y_line[accum_vertex + num_points() - 1] = y_in[num_points() - 1];
        z_line[accum_vertex + num_points() - 1] = z_in[num_points() - 1];
        //m_array[accum_vertex+num_points()-1] = m_in[num_points()-1];
        m_arrayU[accum_vertex + num_points() - 1] = u_in[num_points() - 1];
        m_arrayV[accum_vertex + num_points() - 1] = v_in[num_points() - 1];
        m_arrayW[accum_vertex + num_points() - 1] = w_in[num_points() - 1];
        return;
    }

    if (t_in[interpolation[tick]] - tick * stepDuration <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        x_line[accum_vertex + interpolation[tick]] = x_in[interpolation[tick]];
        y_line[accum_vertex + interpolation[tick]] = y_in[interpolation[tick]];
        z_line[accum_vertex + interpolation[tick]] = z_in[interpolation[tick]];
        //m_array[accum_vertex+interpolation[tick]] = m_in[interpolation[tick]];
        m_arrayU[accum_vertex + interpolation[tick]] = u_in[interpolation[tick]];
        m_arrayV[accum_vertex + interpolation[tick]] = v_in[interpolation[tick]];
        m_arrayW[accum_vertex + interpolation[tick]] = w_in[interpolation[tick]];
    }
    if (fabs(t_in[interpolation[tick] + 1] - tick * stepDuration) <= tick * stepDuration * RELATIVE_TIME_TOLERANCE)
    {
        x_line[accum_vertex + interpolation[tick] + 1] = x_in[interpolation[tick] + 1];
        y_line[accum_vertex + interpolation[tick] + 1] = y_in[interpolation[tick] + 1];
        z_line[accum_vertex + interpolation[tick] + 1] = z_in[interpolation[tick] + 1];
        //m_array[accum_vertex+interpolation[tick]+1] = m_in[interpolation[tick]+1];
        m_arrayU[accum_vertex + interpolation[tick] + 1] = u_in[interpolation[tick] + 1];
        m_arrayV[accum_vertex + interpolation[tick] + 1] = v_in[interpolation[tick] + 1];
        m_arrayW[accum_vertex + interpolation[tick] + 1] = w_in[interpolation[tick] + 1];
    }
}
