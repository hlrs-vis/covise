/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Streamlines
//
//  This class handles bundles of PStreamline objects.
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _STREAMLINES_H_
#define _STREAMLINES_H_

#include "Tracelines.h"

/**
 * Derived class from Tracelines, used for the managment of PStreamline objects.
 */
class Streamlines : public Tracelines
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
       * @param max_points maximum number of points
       * @param max_length maximum length
       * @param td time direction (+1 or -1)
       * @param sfield magnitude to be mapped onto the streamlines
       */
    Streamlines(const coModule *mod,
                const char *name_line, const char *name_magnitude,
                const coDistributedObject *grid, const coDistributedObject *velo,
                const coDistributedObject *ini_p,
                int number_per_tstep, float *x_ini, float *y_ini, float *z_ini,
                int max_points, float max_length, time_direction td,
                const coDistributedObject *sfield = NULL, int number = 0);
    /** Called every time step to create new PTasks.
       */
    virtual void createPTasks();
    // Destructor
    virtual ~Streamlines();

protected:
    // prepare object state for a new time step
    virtual void setTime();

private:
    int max_points_; // limit size of Streamlines
    real max_length_; // limit length of streamlines
    time_direction td_; // time direction (+1 or -1)
};
#endif
