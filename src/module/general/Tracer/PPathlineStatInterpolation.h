/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PPathlineStatInterpolation
//
//  This class performs pathlines for static data with point interpolation
//  for output.
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _P_PPATHLINE_STAT_INTERPOLATION_H_
#define _P_PPATHLINE_STAT_INTERPOLATION_H_

#include "PPathlineStat.h"

/**
 * A pathline for static data.
 *
 */
class PPathlineStatInterpolation : public PPathlineStat
{
public:
    /// Constructor with initial point,
    /// set of grid and velocity objects.
    /** Constructor.
       * @param x_ini X coordinate of the initial point
       * @param y_ini Y coordinate of the initial point
       * @param z_ini Z coordinate of the initial point
       * @param grids0 list with grid objects
       * @param vels0 list with velocity objects
       */
    PPathlineStatInterpolation(real x_ini, real y_ini, real z_ini,
                               std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0, int
                                                                                                                                      number);
    /// Destructor
    virtual ~PPathlineStatInterpolation()
    {
    }
    /** Create a line for the output at time 'tick' (used when
       *  animation is 'growing lines').
       *  @param line_name_time_traj line name
       *  @param mag_name_time_traj  magnitude name
       *  @param tick  instant of time
       *  @retval line  address of pointer to new line object
       *  @retval magn  address of pointer to new magnitude object
       */
    virtual void pathUpToTime(
        const char *line_name_time_traj,
        const char *mag_name_time_traj,
        int tick,
        std::vector<int> &interpolation,
        coDistributedObject **line,
        coDistributedObject **magn);
    /** Create a point for the output at time 'tick' (used when
       *  animation is 'moving points').
       *  @retval points_x  object where the X-coordinate of the point at issue is accumulated
       *  @retval points_y  object where the Y-coordinate of the point at issue is accumulated
       *  @retval points_z  object where the Z-coordinate of the point at issue is accumulated
       *  @param tick  instant of time
       *  @param interpolation  information to look-up in the internal lists.
       */
    virtual void fillPointAtTime(std::vector<float> &points_x,
                                 std::vector<float> &points_y,
                                 std::vector<float> &points_z,
                                 std::vector<float> &mag,
                                 int tick,
                                 std::vector<int> &interpolation);

    int numVertices(int tick,
                    std::vector<int> &interpolation);

    void fillLineResultsAtTime(float *x_line,
                               float *y_line,
                               float *z_line,
                               float *m_array,
                               int accum_vertex,
                               int tick,
                               std::vector<int> &interpolation);

    // same as the other fillLineResultsAtTime() but for vector output data
    void fillLineResultsAtTime(float *x_line,
                               float *y_line,
                               float *z_line,
                               float *m_arrayU,
                               float *m_arrayV,
                               float *m_arrayW,
                               int accum_vertex,
                               int tick,
                               std::vector<int> &interpolation);

protected:
    // h_Reduce: reduce the value of *h if necessary in order to
    //           force the integration to calculate at some predetermined instant of time
    virtual void h_Reduce(float *h);
    // ammendNextTime: react if necessary if the integrator reduced the step
    //                 of integration on its own.
    virtual void ammendNextTime(float h, float h_pre_rkqs, float h_old);

private:
    // interpolateCoef: calculate square and cubic coefficients of a polynomial
    //                  interpolation formula
    void interpolateCoef(float *quadCoef, float *cubCoef, float R0, float R1,
                         float Rp0, float Rp1, float T);
    // interpolate: use a cubic interpoaltion formula given points and derivatives
    //              at the extrema of the interval
    float interpolate(float T, float R0, float R1, float Rp0, float Rp1, float tau);
};
#endif
