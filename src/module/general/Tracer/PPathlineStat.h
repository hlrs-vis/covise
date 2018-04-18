/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PPathlineStat
//
//  This class performs a pathline for static data creating animation effects
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _P_PPATHLINE_STAT_H_
#define _P_PPATHLINE_STAT_H_

#include "PStreamline.h"

/**
 * A pathline for static data.
 *
 */
class PPathlineStat : public PStreamline
{
public:
    /** Constructor
       * @param x_ini X coordinate of initial point
       * @param y_ini Y coordinate of initial point
       * @param z_ini Z coordinate of initial point
       * @param grids0 list of grids
       * @param vels0 list of velocity objects
       */
    PPathlineStat(real x_ini, real y_ini, real z_ini,
                  std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0, int
                                                                                                                         number);
    /// Destructor
    virtual ~PPathlineStat()
    {
    }
    /** Creates object for interpolation
       * @retval look-up index for locating times in the series of calculated points
       */
    virtual void fillInterpolation(std::vector<int> &interpolation);
/** Create a line for the output at time 'tick' (used when
       *  animation is 'growing lines').
       *  @param line_name_time_traj line name
       *  @param mag_name_time_traj  magnitude name
       *  @param tick  instant of time
       *  @retval line  address of pointer to new line object
       *  @retval magn  address of pointer to new magnitude object
       */

    virtual void pathUpToTime(const char *line_name_time_traj,
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

    virtual int numVertices(int tick,
                            std::vector<int> &interpolation);

    virtual void fillLineResultsAtTime(float *x_line,
                                       float *y_line,
                                       float *z_line,
                                       float *m_array,
                                       int accum_vertex,
                                       int tick,
                                       std::vector<int> &interpolation);

    // same as the other fillLineResultsAtTime() but for vector output data
    virtual void fillLineResultsAtTime(float *x_line,
                                       float *y_line,
                                       float *z_line,
                                       float *m_arrayU,
                                       float *m_arrayV,
                                       float *m_arrayW,
                                       int accum_vertex,
                                       int tick,
                                       std::vector<int> &interpolation);

protected:
    // decide whether integration should be interrupted
    virtual int interruptIntegration(const float *dydx, int kount, float length, float time);
    // decide whther the proposed time step should be reduced to fit our objective time values
    virtual void h_Reduce(float *h);
    // react if the stepper reduced the time below our proposal
    virtual void ammendNextTime(float h, float h_pre_rkqs, float h_old);

private:
    // relative tolerance for float comparison
    static const float RELATIVE_TIME_TOLERANCE;
    // discrete time counter
    int next_tick_;
    // continuous time counter
    float next_time_;
};
#endif
