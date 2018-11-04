/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Pathlines.h"
#include "PPathline.h"
#include "Tracer.h"
#include "HTask.h"
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

#include <set>

// #define _DEBUG_

extern int task_type; // growing lines or moving points
extern float verschiebung[3];
extern string speciesAttr;
extern PTask::whatout Whatout;

// Gather results from all PTasks after a time step.
void
Pathlines::gatherTimeStep()
{
    if (task_type == Tracer::GROWING_LINES)
    {
// if growing lines is the animation effect,
// then do the same as in the "Tracelines" class...
        linesForATime();
        magForATime();
    }
    else if (task_type == Tracer::MOVING_POINTS)
    {
        // ... if it is moving points, then include only
        //     ptasks that are still "alive".
        std::vector<float> point_x;
        std::vector<float> point_y;
        std::vector<float> point_z;
        std::vector<float> mag;
        std::vector<float> mag2; // for y component of velocity vector
        std::vector<float> mag3; // for z component of velocity vector
        int i;
        std::string name;
        vector<const coDistributedObject *> sfield;
        loadField(sfield);
        for (i = 0; i < no_ptasks_; ++i)
        {
            PPathline *p_pathline = (PPathline *)(ptasks_[i]);
            if (p_pathline->getKeepStatus() != PTask::NOT_SERVICED)
            {
                continue;
            }
            float *x_in = p_pathline->x_c();
            float *y_in = p_pathline->y_c();
            float *z_in = p_pathline->z_c();
            float *m_in = NULL;
            if (sfield.size() == 0)
            {
                m_in = p_pathline->m_c();
            }
            else
            {
                m_in = p_pathline->m_c_interpolate(sfield, sfield);
            }
            point_x.push_back(x_in[p_pathline->num_points() - 1]);
            point_y.push_back(y_in[p_pathline->num_points() - 1]);
            point_z.push_back(z_in[p_pathline->num_points() - 1]);
            if (Whatout != PTask::V_VEC)
            {
                mag.push_back(m_in[p_pathline->num_points() - 1]);
            }
            else
            {
                mag.push_back(m_in[(p_pathline->num_points() - 1) * 3 + 0]);
                mag2.push_back(m_in[(p_pathline->num_points() - 1) * 3 + 1]);
                mag3.push_back(m_in[(p_pathline->num_points() - 1) * 3 + 2]);
            }

            fprintf(stderr, "Tracer: finished %5d of %d particles\r", i + 1, no_ptasks_);
        }
        fprintf(stderr, "\n");
        name = name_line_;
        char buf[64];
        sprintf(buf, "_%d", covise_time_);
        name += buf;
        if (lines_.size() <= covise_time_)
            lines_.resize(covise_time_ + 1);
        lines_[covise_time_] = new coDoPoints(name, (int)point_x.size(),
                                              &point_x[0],
                                              &point_y[0],
                                              &point_z[0]);
        if (verschiebung[0] != 0.0
            || verschiebung[1] != 0.0
            || verschiebung[2] != 0.0)
        {
            float *x_line, *y_line, *z_line;
            ((coDoPoints *)(lines_[covise_time_]))->getAddresses(&x_line, &y_line, &z_line);
            int no_points = ((coDoPoints *)(lines_[covise_time_]))->getNumPoints();
            Verschieben(verschiebung, no_points, x_line, y_line, z_line);
        }

        if (magnitude_.size() <= covise_time_)
            magnitude_.resize(covise_time_ + 1);
        name = name_magnitude_;
        sprintf(buf, "_%d", covise_time_);
        name += buf;
        if (Whatout != PTask::V_VEC)
        {
            magnitude_[covise_time_] = new coDoFloat(name, (int)mag.size(), &mag[0]);
        }
        else
        {
            magnitude_[covise_time_] = new coDoVec3(name, (int)mag.size(), &mag[0], &mag2[0], &mag3[0]);
        }
        if (speciesAttr.length())
            magnitude_[covise_time_]->addAttribute("SPECIES", speciesAttr.c_str());
    }
}

extern float timeNewParticles;

extern int newParticles;
extern int skip_initial_steps;
extern int cycles;

// Called every time step to create new PTasks if necessary.
void
Pathlines::createPTasks()
{
    int i;
    setTime();
    if (covise_time_ < skip_initial_steps)
    {
        if (covise_time_ == 0)
        {
            no_ptasks_keep_ = no_ptasks_;
            no_ptasks_ = 0;
        }
    }
    else if (covise_time_ == skip_initial_steps)
    {
        if (no_ptasks_keep_ > 0)
        {
            no_ptasks_ = no_ptasks_keep_;
            realTime_0_ = realTime_1_;
            no_ptasks_keep_ = 0;
        }
        ptasks_ = new PTask *[no_ptasks_];
        for (i = 0; i < no_ptasks_; ++i)
        {
            ptasks_[i] = new PPathline(x_ini_[i], y_ini_[i], z_ini_[i],
                                       grid_tstep0_, velo_tstep0_,
                                       grid_tstep1_, velo_tstep1_, realTime_0_, realTime_1_, i);
        }
    }
    else
    {
        // reset status for previous PTasks, and change their
        // grid and velocity lists for the new time step.
        for (i = 0; i < no_ptasks_ - no_new_ptasks_; ++i)
        {
            PPathline *p_pathline = (PPathline *)(ptasks_[i]);
            p_pathline->setNewTimeStep(grid_tstep1_, velo_tstep1_, realTime_1_);
            p_pathline->set_status(PTask::NOT_SERVICED);
        }
        // now make room for new PTasks, if any.
        if (no_new_ptasks_)
        {
            // first keep the old ones...
            PTask **tmp = new PTask *[no_ptasks_];
            for (i = 0; i < no_ptasks_ - no_new_ptasks_; ++i)
            {
                tmp[i] = ptasks_[i];
            }
            delete[] ptasks_;
            ptasks_ = tmp;
            //... and now add the new ones
            // but first prepare an array for initial times.
            std::vector<float> initialTimes;
            std::vector<int> remapPosition; // maps new PTasks to the index for initial position
            if (!ini_p_
                || (covise_time_ > 0 // redundant!?
                    && (!ini_p_->isType("SETELE")
                        || !ini_p_->getAttribute("TIMESTEP"))))
            {
                if (newParticles > 0)
                {
                    if (timeNewParticles > 0.0)
                    {
                        int totalNoReleases = no_ptasks_ / number_per_tstep_;
                        int newNoReleases = no_new_ptasks_ / number_per_tstep_;
                        for (i = totalNoReleases - newNoReleases; i < totalNoReleases; ++i)
                        {
                            int particle;
                            for (particle = 0; particle < number_per_tstep_; ++particle)
                            {
                                initialTimes.push_back(
                                    realTime(NULL, NULL, skip_initial_steps) + i * timeNewParticles);
                                remapPosition.push_back(particle);
                            }
                        }
                    }
                    else
                    {
                        for (i = 0; i < no_new_ptasks_; ++i)
                        {
                            initialTimes.push_back(realTime_1_);
                            remapPosition.push_back(i);
                        }
                    }
                }
            }
            else
            {
                for (i = 0; i < no_new_ptasks_; ++i)
                {
                    initialTimes.push_back(realTime_1_);
                    remapPosition.push_back(i);
                }
            }

            for (i = 0; i < no_new_ptasks_; ++i)
            {
                int position = remapPosition[i];
                ptasks_[i + no_ptasks_ - no_new_ptasks_] = new PPathline(x_ini_[position], y_ini_[position],
                                                                         z_ini_[position],
                                                                         grid_tstep0_, velo_tstep0_,
                                                                         grid_tstep1_, velo_tstep1_,
                                                                         realTime_0_,
                                                                         initialTimes[i], i);
                PPathline *ppathline = (PPathline *)(ptasks_[i + no_ptasks_ - no_new_ptasks_]);
                ppathline->setNewTimeStep(grid_tstep1_, velo_tstep1_, realTime_1_);
                ppathline->set_status(PTask::NOT_SERVICED);
            }
        } // end if(no_new_ptasks_)
    } // else ... covise_time > 0
}

// return 1 if all PTasks have been finished, 0 otherwise (for a given time step)
int
Pathlines::allPFinished()
{
    int ret = 1;
    if (covise_time_ > 0)
    {
        ret = (no_finished_ == no_ptasks_);
    }
#ifdef _DEBUG_
    fprintf(stderr, "allPFinished, covise_time_: %d, no_finished_: %d, no_ptasks_:%d\n", covise_time_, no_finished_, no_ptasks_);
#endif
    return ret;
}

// sets realTime_?_, grid_tstep?_ and velo_tstep?_
void
Pathlines::setTime()
{
    int i, count_p, point;
    float *x_c, *y_c, *z_c;
    coDoPoints *p_points;
    // initialise counters
    no_finished_ = 0;
    serviced_ = 0;
    ++covise_time_;
    // set objects for inputs of the two involved time steps
    if (covise_time_ == 0)
    {
        skip_initial_steps = skip_initial_steps % (no_steps_ * cycles);
    }

    if (covise_time_ > skip_initial_steps)
    {
        ExpandGridObjects(covise_time_ - 1, grid_tstep0_);
        ExpandVeloObjects(covise_time_ - 1, velo_tstep0_);
        realTime_0_ = realTime_1_;
    }
    else
    {
        ExpandGridObjects(covise_time_, grid_tstep0_);
        ExpandVeloObjects(covise_time_, velo_tstep0_);
        realTime_0_ = realTime(grid_, velo_, covise_time_);
    }
    realTime_1_ = realTime(grid_, velo_, covise_time_);
    ExpandGridObjects(covise_time_, grid_tstep1_);
    ExpandVeloObjects(covise_time_, velo_tstep1_);

    if (!ini_p_)
    {
        if (newParticles > 0)
        {
            if (covise_time_ > skip_initial_steps)
            {
                if (timeNewParticles > 0.0)
                {
                    int testExactNReleases = int((realTime_1_ - realTime(NULL, NULL, skip_initial_steps)) / timeNewParticles);
                    int nReleases = testExactNReleases;
                    // consider what happens when timeNewParticles divides
                    // exactly the elepsed real time!!!!
                    if (nReleases == testExactNReleases)
                        ++nReleases;

                    no_new_ptasks_ = nReleases * number_per_tstep_ - no_ptasks_;
                }
                else
                {
                    no_new_ptasks_ = no_ptasks_ / (covise_time_ - skip_initial_steps);
                }
            }
            else
            {
                no_new_ptasks_ = 0;
            }
            no_ptasks_ += no_new_ptasks_;
        }
    }
    // if there are coDoPoints and (this is the first time step or
    // the the coDoPoints are dynamic or ----).
    else if (covise_time_ == 0
             || (ini_p_->isType("SETELE")
                 && ini_p_->getAttribute("TIMESTEP"))
             || (newParticles > 0
                 && timeNewParticles == 0.0))
    {
        ExpandInipObjects(covise_time_, inip_tstep0_);
        if (covise_time_ == 0)
            no_ptasks_ = 0;
        // count number of new PTasks.
        no_new_ptasks_ = 0;
        for (i = 0; i < inip_tstep0_.size(); ++i)
        {
            if (!inip_tstep0_[i]->isType("POINTS"))
                continue;
            p_points = (coDoPoints *)(inip_tstep0_[i]);
            no_new_ptasks_ += p_points->getNumPoints();
        }
        delete[] x_ini_;
        delete[] y_ini_;
        delete[] z_ini_;
        x_ini_ = new float[no_new_ptasks_];
        y_ini_ = new float[no_new_ptasks_];
        z_ini_ = new float[no_new_ptasks_];
        for (i = 0, count_p = 0; i < inip_tstep0_.size(); ++i)
        {
            if (!inip_tstep0_[i]->isType("POINTS"))
                continue;
            p_points = (coDoPoints *)(inip_tstep0_[i]);
            p_points->getAddresses(&x_c, &y_c, &z_c);
            for (point = 0; point < p_points->getNumPoints(); ++point, ++count_p)
            {
                x_ini_[count_p] = x_c[point];
                y_ini_[count_p] = y_c[point];
                z_ini_[count_p] = z_c[point];
            }
        }
        no_ptasks_ += no_new_ptasks_;
        // if coDoPoints is static
        if (!ini_p_->isType("SETELE") || !ini_p_->getAttribute("TIMESTEP"))
        {
            number_per_tstep_ = count_p;
        }
    }
    else if (newParticles) // if there are coDoPoints and this
    {
        // is not the first time step
        // and the coDoPoints is static (grid is necessarily dynamic)
        // inip_tstep0_ is already OK, but we should calculate no_new_ptasks_
        int testExactNReleases = int((realTime_1_ - realTime(NULL, NULL, skip_initial_steps)) / timeNewParticles);
        int nReleases = testExactNReleases;
        // consider what happens when timeNewParticles divides
        // exactly the elepsed real time!!!!
        if (nReleases == testExactNReleases)
            ++nReleases;
        no_new_ptasks_ = nReleases * number_per_tstep_ - no_ptasks_;
        no_ptasks_ += no_new_ptasks_;
    }
    else
    {
        no_new_ptasks_ = 0;
    }
}

int
Pathlines::Finished()
{
    return (covise_time_ >= cycles * no_steps_ - 1);
}

// see header
Pathlines::Pathlines(
    const coModule *mod,
    const char *name_line, const char *name_magnitude,
    const coDistributedObject *grid,
    const coDistributedObject *velo,
    const coDistributedObject *ini_p,
    int number_per_tstep,
    real *x_ini,
    real *y_ini,
    real *z_ini,
    const coDistributedObject *sfield)
    : Tracelines(mod, name_line, name_magnitude, grid, velo, ini_p,
                 number_per_tstep, x_ini, y_ini, z_ini, sfield)
{
    no_new_ptasks_ = 0;
    no_ptasks_keep_ = 0;
}

void
Pathlines::loadField(vector<const coDistributedObject *> &sfield)
{
    if (!sfield_)
        return;
    vector<const coDistributedObject *> l_sfield;
    // assume the dataset is dynamic
    int no_elems;
    const coDistributedObject *const *field_objects = NULL;
    field_objects = ((const coDoSet *)(sfield_))->getAllElements(&no_elems);
    const coDistributedObject *this_field0 = NULL;
    if (covise_time_ > 0)
    {
        this_field0 = field_objects[covise_time_ % no_elems - 1];
    }
    const coDistributedObject *this_field1 = field_objects[covise_time_ % no_elems];
    // other objs in field_objects may be deleted
    set<const coDistributedObject *> toBeDeleted;
    int i;
    for (i = 0; i < no_elems; ++i)
    {
        if (field_objects[i] != this_field0 && field_objects[i] != this_field1)
            toBeDeleted.insert(field_objects[i]);
    }
    set<const coDistributedObject *>::iterator it = toBeDeleted.begin();
    set<const coDistributedObject *>::iterator it_end = toBeDeleted.end();
    for (; it != it_end; ++it)
    {
        delete (*it);
    }
    delete[] field_objects;

    if (this_field0)
    {
        if (this_field0->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *field_objects = ((const coDoSet *)(this_field0))->getAllElements(&no_elems);
            std::vector<const coDistributedObject *> field_tstep;
            ExpandSetList(field_objects, no_elems, field_tstep);
            for (int i = 0; i < field_tstep.size(); ++i)
            {
                l_sfield.push_back(field_tstep[i]);
            }
        }
        else
        {
            l_sfield.push_back(this_field0);
        }
    }
    l_sfield.push_back(NULL);
    if (this_field1->isType("SETELE"))
    {
        int no_elems;
        const coDistributedObject *const *field_objects = ((coDoSet *)(this_field1))->getAllElements(&no_elems);
        std::vector<const coDistributedObject *> field_tstep;
        ExpandSetList(field_objects, no_elems, field_tstep);
        for (int i = 0; i < field_tstep.size(); ++i)
        {
            l_sfield.push_back(field_tstep[i]);
        }
    }
    else
    {
        l_sfield.push_back(this_field1);
    }

    l_sfield.swap(sfield);
}
