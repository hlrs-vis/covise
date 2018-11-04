/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PPathlineStatNoControl
//
//  This class performs a pathline for static data without time step control...
//  Not used up to now.
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _P_PPATHLINE_STAT_NO_CTRL_H_
#define _P_PPATHLINE_STAT_NO_CTRL_H_

#include "PPathlineStat.h"

/**
 * A pathline for static data without time step control.
 *
 */
class PPathlineStatNoControl : public PPathlineStat
{
public:
    /** Constructor
       * @param x_ini X coordinate of initial point
       * @param y_ini Y coordinate of initial point
       * @param z_ini Z coordinate of initial point
       * @param grids0 list of grids
       * @param vels0 list of velocity objects
       */
    PPathlineStatNoControl(real x_ini, real y_ini, real z_ini,
                           std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0, int
                                                                                                                                  number);
    /// Destructor
    virtual ~PPathlineStatNoControl()
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

    /// Stepper without step error control
    virtual void Solve(float, float);

protected:
private:
};
#endif
