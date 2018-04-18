/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PStreamline.h"
#include "Streamlines.h"
#include <do/coDoPoints.h>

extern bool randomStartpoint;
extern int no_start_points;

// Called every time step to create new PTasks.
void
Streamlines::createPTasks()
{
    int i, no_p;
    int shift = 0;
    setTime();

    if (td_ == BOTH)
    {
        no_p = no_ptasks_ / 2;
    }
    else
    {
        no_p = no_ptasks_;
    }

    if (td_ == FORWARDS || td_ == BOTH)
    {
        for (i = 0; i < no_p; ++i)
        {
            ptasks_[i] = new PStreamline(x_ini_[i], y_ini_[i], z_ini_[i],
                                         max_length_, max_points_, grid_tstep0_, velo_tstep0_, 1, i);
        }
        shift = no_p;
    }
    if (td_ == BACKWARDS || td_ == BOTH)
    {
        for (i = 0; i < no_p; ++i)
        {
            ptasks_[i + shift] = new PStreamline(x_ini_[i], y_ini_[i], z_ini_[i],
                                                 max_length_, max_points_, grid_tstep0_, velo_tstep0_, -1, i);
        }
    }
}

// prepare object state for a new time step
void
Streamlines::setTime()
{
    int i, point, count_p;
    float *x_c, *y_c, *z_c;
    coDoPoints *p_points;

    no_finished_ = 0;
    serviced_ = 0;
    ++covise_time_;
    ExpandGridObjects(covise_time_, grid_tstep0_);
    ExpandVeloObjects(covise_time_, velo_tstep0_);
    // clean objects from a previous time step...
    if (covise_time_)
        cleanPTasks();
    if (ini_p_)
    {
        // ...and initial points (if we get a DO_POINTS object)
        delete[] x_ini_;
        delete[] y_ini_;
        delete[] z_ini_;

        ExpandInipObjects(covise_time_, inip_tstep0_);
        no_ptasks_ = 0;
        for (i = 0; i < inip_tstep0_.size(); ++i)
        {
            if (!inip_tstep0_[i]->isType("POINTS"))
                continue;
            p_points = (coDoPoints *)(inip_tstep0_[i]);
            if (randomStartpoint && (no_start_points < p_points->getNumPoints()))
            {
                no_ptasks_ += no_start_points;
            }
            else
            {
                no_ptasks_ += p_points->getNumPoints();
            }
        }
        x_ini_ = new float[no_ptasks_];
        y_ini_ = new float[no_ptasks_];
        z_ini_ = new float[no_ptasks_];
        for (i = 0, count_p = 0; i < inip_tstep0_.size(); ++i)
        {
            if (!inip_tstep0_[i]->isType("POINTS"))
                continue;
            p_points = (coDoPoints *)(inip_tstep0_[i]);
            p_points->getAddresses(&x_c, &y_c, &z_c);
            if (randomStartpoint && (no_start_points < p_points->getNumPoints()))
            {
                for (point = 0; point < no_start_points; ++point, ++count_p)
                {
                    int num = (int)((double)rand() / (RAND_MAX + 1.0) * (p_points->getNumPoints() - 1));
                    x_ini_[count_p] = x_c[num];
                    y_ini_[count_p] = y_c[num];
                    z_ini_[count_p] = z_c[num];
                }
            }
            else
            {
                for (point = 0; point < p_points->getNumPoints(); ++point, ++count_p)
                {
                    x_ini_[count_p] = x_c[point];
                    y_ini_[count_p] = y_c[point];
                    z_ini_[count_p] = z_c[point];
                }
            }
        }
        if (td_ == BOTH)
            no_ptasks_ *= 2;
    }

    ptasks_ = new PTask *[no_ptasks_];
}

Streamlines::~Streamlines()
{
}

// see header
Streamlines::Streamlines(const coModule *mod,
                         const char *name_line, const char *name_magnitude,
                         const coDistributedObject *grid,
                         const coDistributedObject *velo,
                         const coDistributedObject *ini_p,
                         int number_per_tstep,
                         float *x_ini,
                         float *y_ini,
                         float *z_ini,
                         int max_points,
                         real max_length,
                         time_direction td,
                         const coDistributedObject *sfield,
                         int number)
    : Tracelines(mod, name_line, name_magnitude, grid, velo, ini_p,
                 (td == BOTH ? 2 : 1) * number_per_tstep, x_ini, y_ini, z_ini, sfield)
{
    (void)number;
    td_ = td;
    max_points_ = max_points;
    max_length_ = max_length;
}
