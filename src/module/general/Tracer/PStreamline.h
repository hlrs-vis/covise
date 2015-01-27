/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PStreamline
//
//  This class performs a streamline for a list of grids and velocity objects
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _P_STREAMLINE_H_
#define _P_STREAMLINE_H_

#include "PTraceline.h"

/**
 * A simple streamline.
 *
 */
class PStreamline : public PTraceline
{
public:
    /** Constructor
       * @param x_ini X coordinate of initial point
       * @param y_ini Y coordinate of initial point
       * @param z_ini Z coordinate of initial point
       * @param max_length Maximum length of a streamline
       * @param max_points Maximum amount of points of a streamline
       * @param grids0 list of grids
       * @param vels0 list of velocity objects
       * @param ts time direction
       */
    PStreamline(real x_ini, real y_ini, real z_ini, real max_length, int max_points,
                std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0, int ts, int number);
    /** Trigger the solver for this PTask.
       * @param    eps      relative error per time step for step control.
       * @param    eps_abs  absolute error per time step for step control.
       */
    virtual void Solve(float eps, float eps_abs);
    /// Destructor
    ~PStreamline()
    {
    }
    /// interpolate line
    void interpolateField(const vector<const coDistributedObject *> &field,
                          vector<float> &interpolation);
    virtual float *m_c_interpolate(vector<const coDistributedObject *> &sfield,
                                   vector<const coDistributedObject *> &);

protected:
    // derivs: evaluate velocity
    virtual status derivs(float time, const float point[3], float result[3],
                          int allGrids, int search_level);
    // test if integration should be interrupted
    virtual int interruptIntegration(const float *dydx, int kount,
                                     float length, float time);
    // do nothing for this class
    virtual void h_Reduce(float *);
    // do nothing for this class
    virtual void ammendNextTime(float h, float h_pre_rkqs, float h_old);
    // number of this streamline
    int number_;

private:
    vector<gridAndCell> hintRegister_;
    // largets permitted length
    real max_length_;
    // maximum amount of points
    int max_points_;
};
#endif
