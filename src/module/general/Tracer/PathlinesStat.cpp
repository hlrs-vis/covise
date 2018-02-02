/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PathlinesStat.h"
#include "PPathlineStatNoControl.h"
#include "PPathlineStatInterpolation.h"
#include "Tracer.h"
#include <limits.h>
#include <float.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

extern PTask::whatout Whatout;
extern int control;
extern string speciesAttr;

extern float timeNewParticles;
extern int newParticles;

void
PathlinesStat::createPTasks()
{
    // createPTasks is called only once (we have static data)
    int i;
    setTime();

    if (control == 1)
    {
        for (i = 0; i < no_ptasks_; ++i)
        {
            ptasks_[i] = new PPathlineStat(x_ini_[i], y_ini_[i], z_ini_[i],
                                           grid_tstep0_, velo_tstep0_, i);
        }
    }
    else
    {
        for (i = 0; i < no_ptasks_; ++i)
        {
            ptasks_[i] = new PPathlineStatInterpolation(x_ini_[i],
                                                        y_ini_[i], z_ini_[i],
                                                        grid_tstep0_, velo_tstep0_, i);
        }
    }
    // Use of PPathlineStatNoControl is not advisable
    // else{
    //    for(i=0;i<no_ptasks_;++i){
    //       ptasks_[i] = new PPathlineStatNoControl(x_ini_[i],y_ini_[i],z_ini_[i],
    //                                   grid_tstep0_,velo_tstep0_,i);
    //    }
    // }
}

PathlinesStat::~PathlinesStat()
{
}

void
PathlinesStat::gatherTimeStep()
{
}

PathlinesStat::PathlinesStat(const coModule *mod,
                             const char *name_l, // name for line object
                             const char *name_m, // name for magnitude object
                             const coDistributedObject *grid, // grid object
                             const coDistributedObject *velo, // velocity object
                             const coDistributedObject *ini_p, // initial points (coDoPoints type)
                             int number_per_tstep, // number of pathlines for a time step (length of float arrays below)
                             float *x_ini, // list of initial points X-coordinates
                             float *y_ini, // list of initial points Y-coordinates
                             float *z_ini,
                             const coDistributedObject *sfield, // list of initial points Z-coordinates
                             int number)
    : // number of streamline
    Streamlines(mod, name_l, name_m, grid, velo, ini_p,
                number_per_tstep, x_ini, y_ini, z_ini,
                INT_MAX, FLT_MAX, FORWARDS, sfield, number)
{
}

extern int numOfAnimationSteps;
extern int task_type;
extern float verschiebung[3];
extern bool randomOffset;

// produce final output
void
PathlinesStat::gatherAll(coOutputPort *p_line, // pointer to line output port
                         coOutputPort *p_mag) // pointer to magnitude output port
{
    // perhaps we have to map an external field...
    vector<const coDistributedObject *> sfield;
    loadField(sfield);
    srand(1234);

    int *offsets = NULL;
    if (randomOffset)
    {
        offsets = new int[no_ptasks_];
        for (int traj = 0; traj < no_ptasks_; ++traj)
        {
            offsets[traj] = (int)((double)rand() / ((float)RAND_MAX + 1.0) * (timeNewParticles));
        }
    }
    // we basically want to create a dynamic set of line sets.
    std::vector<int> *interpolation = new std::vector<int>[no_ptasks_];
    // first determine number of lines per time step,
    int no_lines_per_t = 0;
    // this number shall be time-independent even if a trajectory
    // gets frozen.
    int traj;
    for (traj = 0; traj < no_ptasks_; ++traj)
    {
        PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
        if (sfield.size() != 0)
        {
            p_task->m_c_interpolate(sfield, sfield);
        }
        if (p_task->num_points() > 0)
        {
            ++no_lines_per_t;
            // find here the intervals at which we will want
            // to make our interpolation
            p_task->fillInterpolation(interpolation[traj]);
        }
    }
    int tick;
    //(void)num_of_releases;
    coDistributedObject **l_time_steps = new coDistributedObject *[numOfAnimationSteps + 1];
    coDistributedObject **m_time_steps = new coDistributedObject *[numOfAnimationSteps + 1];
    l_time_steps[numOfAnimationSteps] = NULL;
    m_time_steps[numOfAnimationSteps] = NULL;

    std::string line_name = p_line->getObjName();
    std::string mag_name = p_mag->getObjName();
    int num_of_releases = 0;
    // loop over time steps
    for (tick = 0; tick < numOfAnimationSteps; ++tick)
    {
        char tick_trail[32];
        sprintf(tick_trail, "_%d", tick);
        std::string line_name_time(line_name);
        std::string mag_name_time(mag_name);
        line_name_time += tick_trail;
        mag_name_time += tick_trail;
        if (task_type == Tracer::GROWING_LINES) // growing lines
        {
#ifdef _ONE_LINE_PER_DO_LINES_
            coDistributedObject **lines = new coDistributedObject *[no_lines_per_t + 1];
            lines[no_lines_per_t] = NULL;
            coDistributedObject **mag = new coDistributedObject *[no_lines_per_t + 1];
            mag[no_lines_per_t] = NULL;
            int line_ind = 0;
            // loop over trajectories
            for (traj = 0; traj < no_ptasks_; ++traj)
            {
                char traj_trail[32];
                sprintf(traj_trail, "_%d", traj);
                std::string line_name_time_traj(line_name_time);
                std::string mag_name_time_traj(mag_name_time);
                line_name_time_traj += traj_trail;
                mag_name_time_traj += traj_trail;

                PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
                if (interpolation[traj].size() == 0)
                    continue;
                p_task->pathUpToTime(p_line,
                                     tick, interpolation[traj],
                                     &lines[line_ind], &mag[line_ind]);

                ++line_ind;
            } // loop over trajectories
            l_time_steps[tick] = new coDoSet(line_name_time, lines);
            m_time_steps[tick] = new coDoSet(mag_name_time, mag);
            // delete objects in lines & mag
            for (traj = 0; traj < no_lines_per_t; ++traj)
            {
                delete lines[traj];
                delete mag[traj];
            }
            delete[] lines;
            delete[] mag;
#else
            // make a single coDoLines with all trajectories
            // first calculate number of vertices
            int no_vertices = 0;
            for (traj = 0; traj < no_ptasks_; ++traj)
            {
                PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
                no_vertices += p_task->numVertices(tick, interpolation[traj]);
            }
            l_time_steps[tick] = new coDoLines(line_name_time, no_vertices,
                                               no_vertices, no_lines_per_t);
            if (Whatout == PTask::V_VEC)
                m_time_steps[tick] = new coDoVec3(mag_name_time, no_vertices);
            else
                m_time_steps[tick] = new coDoFloat(mag_name_time, no_vertices);
            if (speciesAttr.length() > 0)
            {
                m_time_steps[tick]->addAttribute("SPECIES", speciesAttr.c_str());
            }
            // fill arrays
            int *line_list, *vertex_list;
            float *x_line, *y_line, *z_line;
            float *m_array[3] = { NULL, NULL, NULL };
            ((coDoLines *)(l_time_steps[tick]))->getAddresses(&x_line, &y_line, &z_line, &vertex_list, &line_list);
            if (Whatout != PTask::V_VEC)
                ((coDoFloat *)(m_time_steps[tick]))->getAddress(&m_array[0]);
            else
                ((coDoVec3 *)(m_time_steps[tick]))->getAddresses(&m_array[0], &m_array[1], &m_array[2]);

            int accum_vertex = 0;
            int line_ind;
            for (traj = 0, line_ind = 0; traj < no_ptasks_; ++traj)
            {
                PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
                if (interpolation[traj].size() == 0)
                    continue;
                // line list
                line_list[line_ind] = accum_vertex;
                // coordinates and magnitude
                if (Whatout != PTask::V_VEC)
                    p_task->fillLineResultsAtTime(x_line, y_line, z_line, m_array[0], accum_vertex, tick, interpolation[traj]);
                else
                    p_task->fillLineResultsAtTime(x_line, y_line, z_line, m_array[0], m_array[1], m_array[2], accum_vertex, tick, interpolation[traj]);
                accum_vertex += p_task->numVertices(tick, interpolation[traj]);
                ++line_ind;
            }
            // possible optimisation here!!!
            for (accum_vertex = 0; accum_vertex < no_vertices; ++accum_vertex)
            {
                vertex_list[accum_vertex] = accum_vertex;
            }
#endif
        } // 2 -> points
        else
        {
            std::vector<float> points_x;
            std::vector<float> points_y;
            std::vector<float> points_z;
            std::vector<float> mag;
            for (traj = 0; traj < no_ptasks_; ++traj)
            {
                PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
                p_task->fillPointAtTime(points_x, points_y, points_z, mag,
                                        tick, interpolation[traj]);
            }
            int release_tick;
            int tick_newParticles = (int)timeNewParticles;
            if (newParticles) //test if ThrowNewParticles is activated
            {
                if (timeNewParticles <= 0)
                    Covise::sendWarning("ParticleReleaseRate should be positive!");
                else if ((float)tick_newParticles != timeNewParticles)
                {
                    Covise::sendError("ParticleReleaseRate should be an integer value because you use static data. It defines the number of steps until new particles are released again.");
                }
                else
                {
                    //First points were drawn above, so we start with the second ones
                    if (tick >= tick_newParticles)
                    {
                        //draw new particles, if tick reaches ParticleReleaseRate
                        if ((tick % tick_newParticles) == 0)
                            num_of_releases++;
                        //begin with the new points at start position, don't forget to draw the old ones
                        for (int release_rate = 1; release_rate <= num_of_releases; ++release_rate)
                        {
                            //Get release_tick for every starting plane
                            //necessary because of the different ages of the particles
                            release_tick = tick - tick_newParticles * (release_rate);
                            for (traj = 0; traj < no_ptasks_; ++traj)
                            {
                                PPathlineStat *p_task = (PPathlineStat *)(ptasks_[traj]);
                                // if outputting a vector, mag contains a list of 3 consecutive components
                                if (randomOffset)
                                    p_task->fillPointAtTime(points_x, points_y, points_z, mag, release_tick + offsets[traj], interpolation[traj]);
                                else
                                    p_task->fillPointAtTime(points_x, points_y, points_z, mag, release_tick, interpolation[traj]);
                            }
                        }
                    }
                }
            }

            if (points_x.size())
            {
                l_time_steps[tick] = new coDoPoints(line_name_time, (int)points_x.size(),
                                                    &points_x[0], &points_y[0],
                                                    &points_z[0]);
                if (Whatout == PTask::V_VEC)
                {
                    int mag_size = (int)(mag.size() / 3); // mag has triple size, since all 3 components are stored consecutively
                    m_time_steps[tick] = new coDoVec3(mag_name_time, mag_size);
                    float *v_x, *v_y, *v_z;
                    ((coDoVec3 *)m_time_steps[tick])->getAddresses(&v_x, &v_y, &v_z);
                    for (int j = 0; j < mag_size; j++)
                    {
                        v_x[j] = mag[3 * j + 0];
                        v_y[j] = mag[3 * j + 1];
                        v_z[j] = mag[3 * j + 2];
                    }
                }
                else
                    m_time_steps[tick] = new coDoFloat(mag_name_time, (int)mag.size(), &mag[0]);
            }
            else
            {
                l_time_steps[tick] = new coDoPoints(line_name_time, (int)points_x.size(),
                                                    NULL, NULL,
                                                    NULL);
                m_time_steps[tick] = new coDoFloat(mag_name_time, (int)mag.size(), NULL);
            }
            if (speciesAttr.length() > 0)
                m_time_steps[tick]->addAttribute("SPECIES", speciesAttr.c_str());
        }
    } // loop over ticks

    if (verschiebung[0] != 0.0 // shift if required...
        || verschiebung[1] != 0.0
        || verschiebung[2] != 0.0)
    {
        for (tick = 0; tick < numOfAnimationSteps; ++tick)
        {
            float *x_line, *y_line, *z_line;
            int no_points;
            if (task_type == Tracer::GROWING_LINES) // growing lines
            {
                // fill arrays
                int *line_list, *vertex_list;
                ((coDoLines *)(l_time_steps[tick]))->getAddresses(&x_line, &y_line, &z_line, &vertex_list, &line_list);
                no_points = ((coDoLines *)(l_time_steps[tick]))->getNumPoints();
            }
            else // Points
            {
                ((coDoPoints *)(l_time_steps[tick]))->getAddresses(&x_line, &y_line, &z_line);
                no_points = ((coDoPoints *)(l_time_steps[tick]))->getNumPoints();
            }
            Verschieben(verschiebung, no_points,
                        x_line, y_line, z_line);
        }
    }

    coDoSet *output_lines = new coDoSet(line_name, l_time_steps);
    coDoSet *output_mag = new coDoSet(mag_name, m_time_steps);
    // set TIMESTEP attribute
    char dyna_attr[32];
    sprintf(dyna_attr, "1 %d", numOfAnimationSteps);
    output_lines->addAttribute("TIMESTEP", dyna_attr);
    output_mag->addAttribute("TIMESTEP", dyna_attr);
    // delete objects in l_time_steps & m_time_steps
    for (tick = 0; tick < numOfAnimationSteps; ++tick)
    {
        delete l_time_steps[tick];
        delete m_time_steps[tick];
    }
    delete[] l_time_steps;
    delete[] m_time_steps;
    // assign output
    p_line->setCurrentObject(output_lines);
    p_mag->setCurrentObject(output_mag);
    // clean up
    delete[] interpolation;
}
