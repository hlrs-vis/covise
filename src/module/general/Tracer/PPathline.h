/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PPathline
//
//  This class performs a pathline for dynamic data (points or growing lines may be generated)
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _P_PATHLINE_H_
#define _P_PATHLINE_H_

#include "PTraceline.h"

/**
 * A pathline in dynamic data.
 *
 */
class PPathline : public PTraceline
{
public:
    /** Change object state to load the new grid and velocity objects for the end of the time integration
       * @param    gridsNew New grid set
       * @param    velsNew  New velocity set
       * @param    x1       Realtime at the end of the integration
       */
    void setNewTimeStep(std::vector<const coDistributedObject *> &gridsNew,
                        std::vector<const coDistributedObject *> &velsNew, float x1);
    /** Constructor
       * @param x_ini X coordinate of initial point
       * @param y_ini Y coordinate of initial point
       * @param z_ini Z coordinate of initial point
       * @param grids0 list of grids
       * @param vels0 list of velocity objects
       * @param grids1 list of grids
       * @param vels1 list of velocity objects
       * @param time_0 real time of the first grid...
       * @param real_time real time at which the particle is released
       */
    PPathline(real x_ini, real y_ini, real z_ini,
              std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0,
              std::vector<const coDistributedObject *> &grids1, std::vector<const coDistributedObject *> &vels1,
              float time_0, float real_time, int number);
    /** Trigger the solver for this PTask
       * @param    eps      relative error per time step for step control
       * @param    eps_abs  absolute error per time step for step control
       */
    virtual void Solve(float eps, float eps_abs);
    // Destructor
    ~PPathline()
    {
    }
    /** Get information whether this pathline may be continued or not
       * @return status of the real reason for which this pathline was interrupted, or NOT_SERVICED otherwise
       */
    status getKeepStatus()
    {
        return keep_status_;
    }
    void interpolateField(const vector<const coDistributedObject *> &field,
                          vector<float> &interpolation);
    virtual float *m_c_interpolate(vector<const coDistributedObject *> &sfield,
                                   vector<const coDistributedObject *> &);

protected:
    virtual float OutputTime(float integrationTime) const;

private:
    vector<gridAndCell> hintRegister0_;
    vector<gridAndCell> hintRegister1_;
    vector<float> mapped_results_;

    int flag_grid_0_; // there were cell location problems for the smallest time?
    int flag_grid_1_; // there were cell location problems for the largest time?
    float last_h_; // keep track of the last time step value in the integrator
    gridAndCell previousCell_1_; // info about grid and cell for cell location at the largest time

    // velocity evaluation
    virtual status derivs(float time, const float point[3], float result[3],
                          int allGrids, int search_level);
    // pointer to list of grid objects for the largest time
    std::vector<const coDistributedObject *> *grids1_;
    // pointer to list of velocity objects for the largest time
    std::vector<const coDistributedObject *> *vels1_;
    float realTime_0_; // lowest time
    float realTime_1_; // largest time
    status keep_status_; // keep info whether this pathline cannot continued or not
    int number_;
};
#endif
