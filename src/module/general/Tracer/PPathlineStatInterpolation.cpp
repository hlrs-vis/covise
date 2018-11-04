/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PPathlineStatInterpolation.h"
#include <limits.h>
#include <float.h>
#include <do/coDoLines.h>
#include <do/coDoData.h>

extern PTask::whatout Whatout;
extern int numOfAnimationSteps;
extern float stepDuration;
extern string speciesAttr;

// see header
PPathlineStatInterpolation::PPathlineStatInterpolation(real x_ini,
                                                       real y_ini,
                                                       real z_ini,
                                                       std::vector<const coDistributedObject *> &grids0,
                                                       std::vector<const coDistributedObject *> &vels0, int number)
    : PPathlineStat(x_ini, y_ini, z_ini, grids0, vels0, number)
{
}

void
PPathlineStatInterpolation::h_Reduce(float *)
{
}

void
PPathlineStatInterpolation::ammendNextTime(float h, float h_pre_rkqs, float h_old)
{
    (void)h;
    (void)h_pre_rkqs;
    (void)h_old;
}

// see header
void
PPathlineStatInterpolation::pathUpToTime(const char *line_name_time_traj,
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
    int prev_tick;
    float *x_in = x_c();
    float *y_in = y_c();
    float *z_in = z_c();
    float *u_in = u_c();
    float *v_in = v_c();
    float *w_in = w_c();
    float *m_in = m_c();
    float *t_in = t_c();

    for (prev_tick = 0; prev_tick <= interpolation[tick]; ++prev_tick)
    {
        x_l.push_back(x_in[prev_tick]);
        y_l.push_back(y_in[prev_tick]);
        z_l.push_back(z_in[prev_tick]);
        m_l.push_back(m_in[prev_tick]);
    }
    // we have to add also an interpolated point between interpolation[tick]
    // and interpolation[tick]+1
    if (interpolation[tick] < num_points() - 1)
    {
        int p = interpolation[tick];
        float tau = tick * stepDuration - t_in[p];
        float T = t_in[p + 1] - t_in[p];

        x_l.push_back(interpolate(T, x_in[p], x_in[p + 1], u_in[p], u_in[p + 1], tau));
        y_l.push_back(interpolate(T, y_in[p], y_in[p + 1], v_in[p], v_in[p + 1], tau));
        z_l.push_back(interpolate(T, z_in[p], z_in[p + 1], w_in[p], w_in[p + 1], tau));
        // linear interpolation for the magnitude
        float lin = tau / T;
        m_l.push_back(m_in[p] * (1.0f - lin) + m_in[p + 1] * lin);
    }
    int zero = 0;
    std::vector<int> corner_list;
    int i;
    for (i = 0; i < x_l.size(); ++i)
    {
        corner_list.push_back(i);
    }
    *line = new coDoLines(line_name_time_traj, (int)x_l.size(), &x_l[0],
                          &y_l[0], &z_l[0],
                          (int)x_l.size(), &corner_list[0], 1, &zero);
    *magn = new coDoFloat(mag_name_time_traj, (int)m_l.size(), &m_l[0]);
    if (speciesAttr.length() > 0)
        (*magn)->addAttribute("SPECIES", speciesAttr.c_str());
}

int
PPathlineStatInterpolation::numVertices(int tick,
                                        std::vector<int> &interpolation)
{
    if (num_points() == 0)
    {
        return 0;
    }
    if (interpolation[tick] >= num_points() - 1)
        return num_points();
    int ret = interpolation[tick] + 2;
    return ret;
}

void
PPathlineStatInterpolation::fillLineResultsAtTime(float *x_line,
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
    float *t_in = t_c();
    float *u_in = u_c();
    float *v_in = v_c();
    float *w_in = w_c();
    float *m_in = m_c();

    memcpy(x_line + accum_vertex, x_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(y_line + accum_vertex, y_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(z_line + accum_vertex, z_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(m_array + accum_vertex, m_in, (interpolation[tick] + 1) * sizeof(float));

    // we have to add also an interpolated point between interpolation[tick]
    // and interpolation[tick]+1
    if (interpolation[tick] < num_points() - 1)
    {
        int p = interpolation[tick];
        float tau = tick * stepDuration - t_in[p];
        float T = t_in[p + 1] - t_in[p];

        x_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, x_in[p], x_in[p + 1], u_in[p], u_in[p + 1], tau);
        y_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, y_in[p], y_in[p + 1], v_in[p], v_in[p + 1], tau);
        z_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, z_in[p], z_in[p + 1], w_in[p], w_in[p + 1], tau);
        // linear interpolation for the magnitude
        float lin = tau / T;
        m_array[accum_vertex + interpolation[tick] + 1] = m_in[p] * (1.0f - lin) + m_in[p + 1] * lin;
    }
}

void
PPathlineStatInterpolation::fillLineResultsAtTime(float *x_line,
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
    float *t_in = t_c();
    float *u_in = u_c();
    float *v_in = v_c();
    float *w_in = w_c();
    //float *m_in=m_c();

    memcpy(x_line + accum_vertex, x_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(y_line + accum_vertex, y_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(z_line + accum_vertex, z_in, (interpolation[tick] + 1) * sizeof(float));
    //memcpy(m_array+accum_vertex,m_in,(interpolation[tick]+1)*sizeof(float));
    memcpy(m_arrayU + accum_vertex, u_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(m_arrayV + accum_vertex, v_in, (interpolation[tick] + 1) * sizeof(float));
    memcpy(m_arrayW + accum_vertex, w_in, (interpolation[tick] + 1) * sizeof(float));

    // we have to add also an interpolated point between interpolation[tick]
    // and interpolation[tick]+1
    if (interpolation[tick] < num_points() - 1)
    {
        int p = interpolation[tick];
        float tau = tick * stepDuration - t_in[p];
        float T = t_in[p + 1] - t_in[p];

        x_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, x_in[p], x_in[p + 1], u_in[p], u_in[p + 1], tau);
        y_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, y_in[p], y_in[p + 1], v_in[p], v_in[p + 1], tau);
        z_line[accum_vertex + interpolation[tick] + 1] = interpolate(T, z_in[p], z_in[p + 1], w_in[p], w_in[p + 1], tau);
        // linear interpolation for the magnitude
        float lin = tau / T;

        m_arrayU[accum_vertex + interpolation[tick] + 1] = u_in[p] * (1.0f - lin) + u_in[p + 1] * lin;
        m_arrayV[accum_vertex + interpolation[tick] + 1] = v_in[p] * (1.0f - lin) + v_in[p + 1] * lin;
        m_arrayW[accum_vertex + interpolation[tick] + 1] = w_in[p] * (1.0f - lin) + w_in[p + 1] * lin;

        //m_array[accum_vertex+interpolation[tick]+1] = m_in[p]*(1.0-lin) + m_in[p+1]*lin;
    }
}

// see header
void
PPathlineStatInterpolation::interpolateCoef(float *quadCoef,
                                            float *cubCoef,
                                            float R0, // left value
                                            float R1, // right value
                                            float Rp0, // left value of derivative
                                            float Rp1, // right value of derivative
                                            float T) // time interval length
{
    float T2 = T * T;
    float T3 = T2 * T;

    *cubCoef = T * (Rp1 + Rp0) - 2.0f * (R1 - R0);
    *cubCoef /= T3;
    *quadCoef = (R1 - R0 - Rp0 * T - (*cubCoef) * T3) / T2;
}

// see header
float
PPathlineStatInterpolation::interpolate(float T, // time interval length
                                        float R0, // left value
                                        float R1, // right value
                                        float Rp0, // left value of derivative
                                        float Rp1, // right value of derivative
                                        float tau) // identifies interpolation point with origin on the left
{
    float quad, cubic;
    interpolateCoef(&quad, &cubic, R0, R1, Rp0, Rp1, T);
    return (R0 + tau * (Rp0 + tau * (quad + tau * cubic)));
}

// get the point at discrete time 'tick' by interpoaltion using the calculated values.
void
PPathlineStatInterpolation::fillPointAtTime(std::vector<float> &points_x,
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
    float *u_in = u_c();
    float *v_in = v_c();
    float *w_in = w_c();
    float *m_in = m_c();
    float *t_in = t_c();

    int p = interpolation[tick];
    float tau = tick * stepDuration - t_in[p];
    float T = t_in[p + 1] - t_in[p];

    points_x.push_back(interpolate(T, x_in[p], x_in[p + 1], u_in[p], u_in[p + 1], tau));
    points_y.push_back(interpolate(T, y_in[p], y_in[p + 1], v_in[p], v_in[p + 1], tau));
    points_z.push_back(interpolate(T, z_in[p], z_in[p + 1], w_in[p], w_in[p + 1], tau));
    // linear interpolation for the magnitude
    float lin = tau / T;
    //    mag.push_back(m_in[p]*(1.0f-lin) + m_in[p+1]*lin);
    if (Whatout == PTask::V_VEC)
    {
        mag.push_back(u_in[p] * (1.0f - lin) + u_in[p + 1] * lin);
        mag.push_back(v_in[p] * (1.0f - lin) + v_in[p + 1] * lin);
        mag.push_back(w_in[p] * (1.0f - lin) + w_in[p + 1] * lin);
    }
    else
        mag.push_back(m_in[p] * (1.0f - lin) + m_in[p + 1] * lin);
}
