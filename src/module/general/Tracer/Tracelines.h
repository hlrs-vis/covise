/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Tracelines
//
//  This class handles bundles of PTraceline objects.
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _TRACELINES_H_
#define _TRACELINES_H_

#include "HTask.h"

/**
 * Derived class from HTask, used when the generated PTasks create lines.
 */
class Tracelines : public HTask
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
    Tracelines(
        const coModule *mod,
        const char *name_line, const char *name_magnitude,
        const coDistributedObject *grid, const coDistributedObject *velo, const coDistributedObject *ini_p,
        int number_per_tstep, real *x_ini, real *y_ini, real *z_ini,
        const coDistributedObject *sfield = NULL);
    /// Destructor.
    virtual ~Tracelines()
    {
        delete[] x_ini_;
        delete[] y_ini_;
        delete[] z_ini_;
    }
/** Gather results from all PTasks after a time step.
       */
    virtual void gatherTimeStep();
    /** Gather results from all all time steps.
       * @param    p_line   pointer to line output port.
       * @param    p_mag    pointer to magnitude output port.
       */
    virtual void gatherAll(coOutputPort *p_line, coOutputPort *p_mag);
    /** Return 1 if all time steps are done.
       * @return            all time steps are done.
       */
    virtual int Finished();

protected:
    virtual void loadField(vector<const coDistributedObject *> &sfield);
    int warnOutOfDomain_;
    void WarnIfOutOfDomain();
    int number_per_tstep_; // new PTasks released
    // used in the construction of the corner list of created lines,
    // it creates an array as long as required by the longest line
    int *cornerList();
// makes lines for a time (used in gatherTimeStep)
    void linesForATime();
// makes magnitude objects for a time (used in gatherTimeStep)
    void magForATime();
    // displacement for all point according to constant vector shift
    void Verschieben(const float *shift, int no_p,
                     float *x, float *y, float *z);

    // discrete time
    int covise_time_; // starting from 0
    std::string interaction_; // value for FEEDBACK attribute

    real *x_ini_; // arrays with initial coordinates for the streamlines
    real *y_ini_;
    real *z_ini_;
    // elements of these arrays (lines_, magnitude_) are added by gatherTimeStep
    std::vector<coDistributedObject *> lines_; // lines_[i] is a set for a time step
    std::vector<coDistributedObject *> magnitude_;

    const coDistributedObject *sfield_;
    virtual int Diagnose();

private:
};
#endif
