/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Pathlines
//
//  This class handles bundles of PPathline objects.
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _PATHLINES_H_
#define _PATHLINES_H_

#include "Tracelines.h"

#include <util/coviseCompat.h>

/**
 * Derived class from Tracelines, used for the management of PPathline objects.
 */
class Pathlines : public Tracelines
{
public:
    /** Constructor
       * @param name_line name for line
       * @param name_magnitude name for magnitude object
       * @param grid grid object
       * @param velo velocity object
       * @param ini_p initial points object
       * @param number_per_tstep number of lines computed per time step
       * @param x_ini X coordinates of initial points
       * @param y_ini Y coordinates of initial points
       * @param z_ini Z coordinates of initial points
       * @param sfield magnitude to be mapped onto the integrated lines
       */
    Pathlines(const coModule *mod,
              const char *name_line, const char *name_magnitude,
              const coDistributedObject *grid, const coDistributedObject *velo,
              const coDistributedObject *ini_p,
              int number_per_tstep, real *x_ini, real *y_ini, real *z_ini,
              const coDistributedObject *sfield = NULL);
    /** Called every time step to create new PTasks if necessary.
       */
    virtual void createPTasks();
    /** Return 1 if all PTasks have been finished, 0 otherwise
       *  (initially there is nothing to do, and all is finished).
       * @return            all PTasks have been finished or not.
       */
    virtual int allPFinished();
/** Gather results from all PTasks after a time step.
       */
    virtual void gatherTimeStep();
    /** Are we done with all time steps?
       */
    virtual int Finished();
    /// Destructor.
    virtual ~Pathlines()
    {
    }

protected:
    virtual void loadField(vector<const coDistributedObject *> &sfield);

private:
    std::vector<const coDistributedObject *> grid_tstep1_; // objects for the end of the actual time step
    std::vector<const coDistributedObject *> velo_tstep1_;

    float realTime_0_; // time at the beginning of the actual time step
    float realTime_1_; // time at the end of the actual time step

    void setTime(); // sets realTime_?_, grid_tstep?_ and velo_tstep?_
    int no_new_ptasks_; // the number of PTasks may change dynamically
    // this is needed in a quick-and-dirty solution
    // to temporarily hide the value of no_ptasks_
    // when the value of p_skip_initial_steps_ is greater than 0
    int no_ptasks_keep_;
};
#endif
