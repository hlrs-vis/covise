/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PathlinesStat
//
//  This class handles bundles of pathline for dynamic data.
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _PATHLINES_STAT_H_
#define _PATHLINES_STAT_H_

#include "Streamlines.h"

/**
 * Derived class from Streamlines, used for the management of PPathlineStat objects.
 */
class PathlinesStat : public Streamlines
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
       * @param sfield magnitude to be mapped onto the streamlines
       */
    PathlinesStat(const coModule *mod,
                  const char *name_line, const char *name_magnitude,
                  const coDistributedObject *grid, const coDistributedObject *velo,
                  const coDistributedObject *ini_p,
                  int number_per_tstep, float *x_ini, float *y_ini, float *z_ini,
                  const coDistributedObject *sfield = NULL, int number = 0);
    /** Called every time step to create new PTasks if necessary.
       */
    virtual void createPTasks();
    // Destructor
    virtual ~PathlinesStat();
    /** Gather results from all PTasks after a time step.
       */
    virtual void gatherTimeStep();
    /** Gather results from all all time steps.
       * @param    p_line   pointer to line output port.
       * @param    p_mag    pointer to magnitude output port.
       */
    virtual void gatherAll(coOutputPort *p_line, coOutputPort *p_mag);

protected:
private:
};
#endif
