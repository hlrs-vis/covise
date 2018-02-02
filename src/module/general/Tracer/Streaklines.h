/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Streaklines
//
//  This class handles bundles of PPathline objects and creates Streaklines
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _STREAKLINES_H_
#define _STREAKLINES_H_

#include "Pathlines.h"

/**
 * Derived class from Tracelines, used for the management of PPathline objects.
 */
class Streaklines : public Pathlines
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
       */
    Streaklines(const coModule *mod,
                const char *name_line, const char *name_magnitude,
                const coDistributedObject *grid, const coDistributedObject *velo,
                const coDistributedObject *ini_p,
                int number_per_tstep, real *x_ini, real *y_ini, real *z_ini,
                const coDistributedObject *sfield = NULL);
/** Gather results from all PTasks after a time step.
       */
    virtual void gatherTimeStep();
    /// Destructor.
    virtual ~Streaklines()
    {
    }

protected:
    /** Check if coDoPoints inputs for construction were OK.
       * @return            0 if OK, -1 otherwise.
       */
    virtual int DiagnosePoints();

private:
};
#endif
