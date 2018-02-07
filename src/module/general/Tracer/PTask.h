/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PTask
//
//  Base class for all primitive tasks as defined in the concept paper.
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _P_TASK_H_
#define _P_TASK_H_

#include <do/covise_gridmethods.h>
#include <appl/ApplInterface.h>
using namespace covise;

#define real float

typedef real MYPOINT[3];

/**
 * Base class for "primitive" tasks as defined in the concept paper.
 *
 */
class PTask
{
public:
    // state of a PTask
    enum status
    {
        FINISHED_LENGTH,
        FINISHED_DOMAIN,
        FINISHED_POINTS,
        FINISHED_TIME,
        FINISHED_VELOCITY,
        FINISHED_STEP,
        SERVICED,
        NOT_SERVICED
    };
    enum whatout
    {
        V = 0,
        VX = 1,
        VY = 2,
        VZ = 3,
        TIME = 4,
        ID = 5,
        V_VEC = 6
    };
    enum
    {
        MAX_NUM_OF_VARS = 3
    };
    /** Set the identification label for the thread
       * @param    label    identificator >= 0
       */
    void set_label(int label);
    /** Get the identification label.
       * @return            identification label.
       */
    int get_label();
    /// Make an object to give a thread its identity label.
    PTask(int label);
    /// Make an object with starting point and sets of grid and velocity objects.
    PTask(real x_ini, real y_ini, real z_ini,
          std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0);
    /** Set task status to keep track of events related with this PTask
       * @param    new_stat new status
       */
    void set_status(status new_stat);
    /** Set task status to keep track of events related with this PTask
       * @param    new_stat new status
       */
    int get_status();
    /** Trigger the solver for this PTask
       * @param    eps      relative error per time step for step control
       * @param    eps_abs  absolute error per time step for step control
       */
    virtual void Solve(float eps, float eps_abs)
    {
        (void)eps;
        (void)eps_abs;
    }
    /// Destructor
    virtual ~PTask()
    {
    }

protected:
    static float interpolateFieldInGrid(const coDistributedObject *grid,
                                        const coDistributedObject *field,
                                        const int *cell,
                                        float x, float y, float z);
    static float mapFieldInGrid(const coDistributedObject *grid,
                                const coDistributedObject *field,
                                const int *cell,
                                float x, float y, float z);
    // pointer to list of pointer to objects
    std::vector<const coDistributedObject *> *grids0_;
    std::vector<const coDistributedObject *> *vels0_;
    // initial point
    MYPOINT ini_point_;
    // keep intermediate calculations of rkck...
    struct onePoint
    {
        float time_;
        status status_;
        std::vector<float> var_;
        std::vector<float> var_dot_;
    } intermediate_[4];
    // did we get we out of the domain?
    enum emergency
    {
        OK,
        GOT_OUT_OF_DOMAIN
    } emergency_;
    // enum IntegrationConstants {STEPS_PER_CELL=8,EXPLORE_OUT_OF_DOMAIN=2};
    // velocity evaluation
    virtual status derivs(float time, const float *point, float *result,
                          int allGrids, int search_level)
    {
        (void)time;
        (void)point;
        (void)result;
        (void)allGrids, (void)search_level;
        return SERVICED;
    }
    // integration stepper
    status rkqs(float *, const float *, int, float *, float, float, float,
                float[], float *, float *);
    // RK integration step
    status rkck(const float *y, const float *dydx, int n, float x, float h, float *ytemp, float *yerr);
    // suggest a time step value based on cell size and velocity
    float suggestInitialH(const int *cell, const coDistributedObject *p_grid,
                          const coDistributedObject *p_velo);

private:
    // see status
    status status_;
    // worker thread label
    int label_;
    // keep intermediate calculation of rkck
    void fillIntermediate(int i, float time, float *var, float *var_dot, int n,
                          status ctrl);
};
#endif
