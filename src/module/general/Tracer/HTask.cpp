/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include "HTask.h"

// expand a set into a list (out_array)
int
ExpandSetList(const coDistributedObject *const *objects,
              int h_many,
              std::vector<const coDistributedObject *> &out_array)
{
    out_array.clear();
    for (int i = 0; i < h_many; ++i)
    {
        const coDoSet *set = dynamic_cast<const coDoSet *>(objects[i]);
        if (!set)
        {
            out_array.push_back(objects[i]);
            continue;
        }

        // Set
        int in_set_len;
        const coDistributedObject *const *in_set_elements = set->getAllElements(&in_set_len);
        std::vector<const coDistributedObject *> partial;
        int add = ExpandSetList(in_set_elements, in_set_len, partial);
        for (int j = 0; j < add; ++j)
        {
            out_array.push_back(partial[j]);
        }
    }
#ifdef _CLEAN_UP_
    delete[] objects;
#endif
    return (int)out_array.size();
}

// get the grid expanded lists
void
HTask::ExpandGridObjects(int covise_time,
                         std::vector<const coDistributedObject *> &grid_tstep0)
{
    grid_tstep0.clear();
    covise_time = covise_time % grid_tstep_opt_.size();
    for (int i = 0; i < grid_tstep_opt_[covise_time].size(); ++i)
    {
        grid_tstep0.push_back(grid_tstep_opt_[covise_time][i]);
    }
}

// get the velocity expanded lists
void
HTask::ExpandVeloObjects(int covise_time,
                         std::vector<const coDistributedObject *> &velo_tstep0)
{
    velo_tstep0.clear();
    covise_time = covise_time % num_velo_steps_;
    int i;
    for (i = 0; i < velo_tstep_opt_[covise_time].size(); ++i)
    {
        velo_tstep0.push_back(velo_tstep_opt_[covise_time][i]);
    }
}

// get the initial points expanded lists
void
HTask::ExpandInipObjects(int covise_time,
                         std::vector<const coDistributedObject *> &inip_tstep0)
{
    inip_tstep0.clear();
    covise_time = covise_time % num_inip_steps_;
    int i;
    for (i = 0; i < inip_tstep_opt_[covise_time].size(); ++i)
    {
        inip_tstep0.push_back(inip_tstep_opt_[covise_time][i]);
    }
}

// Make set expansions for input objects
void
HTask::MakeIAsForExpand()
{
    int num_elems;
    const coDistributedObject *const *set_list;
    if (dynamic_cast<const coDoSet *>(grid_))
    {
        set_list = ((const coDoSet *)(grid_))->getAllElements(&num_elems);
        if (grid_->getAttribute("TIMESTEP"))
        {
            grid_tstep_opt_.resize(num_elems);
            for (int time = 0; time < num_elems; ++time)
            {
                if (set_list[time]->isType("SETELE"))
                {
                    int no_obj_in_set;
                    const coDistributedObject *const *set_list_int = ((const coDoSet *)(set_list[time]))->getAllElements(&no_obj_in_set);
                    ExpandSetList(set_list_int, no_obj_in_set, grid_tstep_opt_[time]);
                }
                else
                {
                    grid_tstep_opt_[time].push_back(set_list[time]);
                }
            }
        }
        else
        {
            // If grid is a static set.
            grid_tstep_opt_.resize(1);
            ExpandSetList(set_list, num_elems, grid_tstep_opt_[0]);
        }
    }
    else // simple object
    {
        grid_tstep_opt_.resize(1);
        grid_tstep_opt_[0].push_back(grid_);
    }
    // Velocity
    if (dynamic_cast<const coDoSet *>(velo_))
    {
        set_list = ((const coDoSet *)(velo_))->getAllElements(&num_elems);
        if (velo_->getAttribute("TIMESTEP"))
        {
            num_velo_steps_ = num_elems;
            velo_tstep_opt_ = new std::vector<const coDistributedObject *>[num_elems];
            int time;
            for (time = 0; time < num_elems; ++time)
            {
                if (set_list[time]->isType("SETELE"))
                {
                    int no_obj_in_set;
                    const coDistributedObject *const *set_list_int = ((coDoSet *)(set_list[time]))->getAllElements(&no_obj_in_set);
                    ExpandSetList(set_list_int, no_obj_in_set, velo_tstep_opt_[time]);
                }
                else
                {
                    velo_tstep_opt_[time].push_back(set_list[time]);
                }
            }
        }
        else // If grid is a static set.
        {
            num_velo_steps_ = 1;
            velo_tstep_opt_ = new std::vector<const coDistributedObject *>[1];
            ExpandSetList(set_list, num_elems, velo_tstep_opt_[0]);
        }
    }
    else // simple object
    {
        num_velo_steps_ = 1;
        velo_tstep_opt_ = new std::vector<const coDistributedObject *>[1];
        velo_tstep_opt_[0].push_back(velo_);
    }
    // Initial points
    if (!ini_p_)
        return;
    if (dynamic_cast<const coDoSet *>(ini_p_))
    {
        set_list = ((coDoSet *)(ini_p_))->getAllElements(&num_elems);
        if (ini_p_->getAttribute("TIMESTEP"))
        {
            num_inip_steps_ = num_elems;
            inip_tstep_opt_ = new std::vector<const coDistributedObject *>[num_elems];
            int time;
            for (time = 0; time < num_elems; ++time)
            {
                if (set_list[time]->isType("SETELE"))
                {
                    int no_obj_in_set;
                    const coDistributedObject *const *set_list_int = ((coDoSet *)(set_list[time]))->getAllElements(&no_obj_in_set);
                    ExpandSetList(set_list_int, no_obj_in_set, inip_tstep_opt_[time]);
                }
                else
                {
                    inip_tstep_opt_[time].push_back(set_list[time]);
                }
            }
        }
        else // If grid is a static set.
        {
            num_inip_steps_ = 1;
            inip_tstep_opt_ = new std::vector<const coDistributedObject *>[1];
            ExpandSetList(set_list, num_elems, inip_tstep_opt_[0]);
        }
    }
    else // simple object
    {
        num_inip_steps_ = 1;
        inip_tstep_opt_ = new std::vector<const coDistributedObject *>[1];
        inip_tstep_opt_[0].push_back(ini_p_);
    }
}

// see header
HTask::HTask(const coModule *mod,
             const char *name_line, const char *name_magnitude,
             const coDistributedObject *grid,
             const coDistributedObject *velo,
             const coDistributedObject *ini_p)
    : module_(mod)
    , name_line_(name_line)
    , name_magnitude_(name_magnitude)
    , velo_tstep_opt_(NULL)
    , num_velo_steps_(0)
    , inip_tstep_opt_(NULL)
    , num_inip_steps_(0)
{
    no_steps_ = 1;
    grid_ = grid;
    velo_ = velo;

    if (grid->isType("SETELE") && grid->getAttribute("TIMESTEP"))
    {
        // FIXME
        no_steps_ = ((coDoSet *)(grid))->getNumElements();
    }

    ini_p_ = ini_p;
    ptasks_ = 0;
    no_ptasks_ = 0;
    no_finished_ = serviced_ = 0;

    // Create arrays of std::vector's to optimise ExpandObjects.
    MakeIAsForExpand();
    fillRealTime();
}

// maqke the assignation of a PTask for a worker thread
void
HTask::assignTask(PTask **pp_task,
                  int label)
{
    // shortly it is serviced
    ptasks_[serviced_]->set_status(PTask::SERVICED);
    ptasks_[serviced_]->set_label(label);
    *pp_task = ptasks_[serviced_++]; // assign and increase serviced_ counter
}

// Gather results from a PTask
void
HTask::gatherPTask(PTask **pp_task)
{ // a thread has finished a PTask
    ++no_finished_;
    *pp_task = 0;
}

// Solve sequentially all PTasks (used without pthreads)
void
HTask::Solve(float epsilon,
             float epsilon_abs)
{
    int i;
    for (i = 0; i < no_ptasks_; ++i)
    {
        ptasks_[i]->Solve(epsilon, epsilon_abs);
        ++no_finished_;
        ++serviced_;
    }
}

void
HTask::cleanPTasks()
{
    int i;
    if (ptasks_)
    {
        for (i = 0; i < no_ptasks_; ++i)
        {
            delete ptasks_[i];
        }
        delete[] ptasks_;
    }
}

int
HTask::hasTimeSteps()
{
    if (grid_->isType("SETELE") && grid_->getAttribute("TIMESTEP"))
        return 1;
    return 0;
}

extern float stepDuration;
extern int cycles;

// create lists of real time steps for all dicrete times
void
HTask::fillRealTime()
{
    int i;
    if (!grid_tstep_opt_[0][0]->getAttribute("REALTIME"))
    {
        for (i = 0; i < no_steps_; ++i)
        {
            realTimes_.push_back(i * stepDuration);
        }
        return;
    }

    if (no_steps_ > 1
        && grid_tstep_opt_[1][0]->getAttribute("REALTIME")
        && realTimeInt(grid_, velo_, 0) == realTimeInt(grid_, velo_, 1))
    {
        for (i = 0; i < no_steps_; ++i)
        {
            realTimes_.push_back(i * stepDuration);
        }
        return;
    }

    for (i = 0; i < no_steps_; ++i)
    {
        realTimes_.push_back(realTimeInt(grid_, velo_, i));
    }
}

// get real time for a discrete time
float
HTask::realTime(const coDistributedObject *, const coDistributedObject *, int step)
{
    int data_step = step % no_steps_;
    int no_cycle = step / no_steps_;

    float ret = realTimes_[data_step];

    float cycle_time = stepDuration + realTimes_[no_steps_ - 1];
    ret += no_cycle * cycle_time;

    return ret;
}

// returns real time: slow function, keep results in an array and afterwards use only the array
float
HTask::realTimeInt(const coDistributedObject *grid, const coDistributedObject *velo, int step)
{
    (void)grid;
    (void)velo;

    float ret;
    const char *time_g = grid_tstep_opt_[step][0]->getAttribute("REALTIME");
    if (time_g)
    {
        // static sets or static objects with realtime
        sscanf(time_g, "%f", &ret);
    }
    else
    {
        ret = step * stepDuration;
    }
    return ret;
}

// returns 0 if no problem is found with the data, -1 otherwise
extern float grid_tolerance;
extern float epsilon;
extern float epsilon_abs;
extern float divide_cell;
extern float max_out_of_cell;

int
HTask::DiagnosePoints()
{
    return 0;
}

int
HTask::Diagnose()
{
    int no_elems;
    std::vector<const coDistributedObject *> grid_tstep;
    std::vector<const coDistributedObject *> velo_tstep;
    std::vector<const coDistributedObject *> inip_tstep;

    if (hasTimeSteps()) // check that velo and ini_points are also dynamic
    {
        if (!velo_->isType("SETELE") || !velo_->getAttribute("TIMESTEP"))
        {
            module_->sendError("The grid is dynamic but the velocity is static.");
            return -1;
        }
        // ((coDoSet *)(velo_))->getAllElements(&no_elems);
        // FIXME
        //no_elems = ((coDoSet *)(velo_))->getNumElements();
        no_elems = ((coDoSet *)(velo_))->getNumElements();
        if (no_steps_ != no_elems)
        {
            module_->sendError("The grid and velocity objects have a different number of time steps.");
            return -1;
        }
        if (ini_p_
            && ini_p_->isType("SETELE")
            && ini_p_->getAttribute("TIMESTEP"))
        {
            //((coDoSet *)(ini_p_))->getAllElements(&no_elems);
            // FIXME
            no_elems = ((coDoSet *)(ini_p_))->getNumElements();
            if (no_steps_ != no_elems)
            {
                module_->sendError("The grid and initial-points objects have a different number of time steps.");
                return -1;
            }
        }
    } // grid is static
    else
    {
        if (velo_->isType("SETELE") && velo_->getAttribute("TIMESTEP"))
        {
            module_->sendError("The grid is static, but the velocity object is dynamic.");
            return -1;
        }
        if (ini_p_ && ini_p_->isType("SETELE") && ini_p_->getAttribute("TIMESTEP"))
        {
            module_->sendError("The grid is static, but the initial points object is dynamic.");
            return -1;
        }
    }

    // now check how many objects are found per time step
    int i;
    for (i = 0; i < no_steps_; ++i)
    {
        ExpandGridObjects(i, grid_tstep);
        ExpandVeloObjects(i, velo_tstep);
        no_elems = (int)grid_tstep.size();
        if (no_elems != (int)velo_tstep.size())
        {
            module_->sendError("Different number of grids and velocity objects at time %d.", i);
            return -1;
        }
        /* no error
            if(ini_p_){
               ExpandInipObjects(i,inip_tstep);
               if(no_elems != inip_tstep.size()){
                  sprintf(buf,"Different number of grids and initial-points objects at time %d.",i);
                  Covise::sendError(buf);
                  return -1;
               }
            }
      */
        // now check that UNSGRD are matched with USTVDT and so on...
        int obj_at_time;
        for (obj_at_time = 0; obj_at_time < no_elems; ++obj_at_time)
        {
            if (grid_tstep[obj_at_time]->isType("UNSGRD"))
            {
                if (!(velo_tstep[obj_at_time]->isType("USTVDT")))
                {
                    module_->sendError("An unstructured grid is not matched with an unstructured vector data object");
                    return -1;
                }
                else // check dimensions
                {
                    int e, c, p_g;
                    ((coDoUnstructuredGrid *)(grid_tstep[obj_at_time]))->
                        getGridSize(&e, &c, &p_g);
                    int p_v = ((coDoVec3 *)(velo_tstep[obj_at_time]))->getNumPoints();
                    if (p_v != 0 && p_v != p_g)
                    {
                        module_->sendError("Vector data is not vertex-based");
                        return -1;
                    }
                    continue;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("POLYGN"))
            {
                if (!(velo_tstep[obj_at_time]->isType("USTVDT")))
                {
                    module_->sendError("An polygonal grid is not matched with an unstructured vector data object");
                    return -1;
                }
                else // check dimensions
                {
                    int p_g;
                    p_g = dynamic_cast<const coDoPolygons *>(grid_tstep[obj_at_time])->getNumPoints();
                    int p_v = dynamic_cast<const coDoVec3 *>(velo_tstep[obj_at_time])->getNumPoints();
                    if (p_v != 0 && p_v != p_g)
                    {
                        module_->sendError("Vector data is not vertex-based");
                        return -1;
                    }
                    continue;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("UNIGRD"))
            {
                if (!(velo_tstep[obj_at_time]->isType("USTVDT")))
                {
                    module_->sendError("A uniform grid is not matched with a structured vector data object.");
                    return -1;
                }
                else // check dimensions
                {
                    int x_s, y_s, z_s;
                    ((coDoUniformGrid *)(grid_tstep[obj_at_time]))->
                        getGridSize(&x_s, &y_s, &z_s);
                    int nelem = ((coDoVec3 *)(velo_tstep[obj_at_time]))->getNumPoints();
                    if (nelem != 0 /* ??? */ && x_s * y_s * z_s != nelem)
                    {
                        module_->sendError("Vector data is not vertex-based");
                        return -1;
                    }
                    continue;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("RCTGRD"))
            {
                if (!(velo_tstep[obj_at_time]->isType("USTVDT")))
                {
                    module_->sendError("A rectangular grid is not matched with a structured vector data object.");
                    return -1;
                }
                else
                {
                    int x_s, y_s, z_s;
                    ((coDoRectilinearGrid *)(grid_tstep[obj_at_time]))->getGridSize(&x_s, &y_s, &z_s);
                    int nelem = ((coDoVec3 *)(velo_tstep[obj_at_time]))->getNumPoints();

                    if (nelem != 0 /* ??? */ && x_s * y_s * z_s != nelem)
                    {
                        module_->sendError("Vector data is not vertex-based");
                        return -1;
                    }
                    continue;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("STRGRD"))
            {
                if (!(velo_tstep[obj_at_time]->isType("USTVDT")))
                {
                    module_->sendError("A structured grid is not matched with a structured vector data object.");
                    return -1;
                }
                else
                {
                    int x_s, y_s, z_s;
                    ((coDoStructuredGrid *)(grid_tstep[obj_at_time]))->getGridSize(&x_s, &y_s, &z_s);
                    int nelem = ((coDoVec3 *)(velo_tstep[obj_at_time]))->getNumPoints();

                    if (nelem != 0 /* ??? */ && x_s * y_s * z_s != nelem)
                    {
                        module_->sendError("Vector data is not vertex-based");
                        return -1;
                    }
                    continue;
                }
            }
            else
            {
                module_->sendError("One grid object has a non-grid type.");
                return -1;
            }
        }
    }

    return 0;
}

// returns whether all PTasks finished for a time step
int
HTask::allPFinished()
{
    return (no_finished_ == no_ptasks_);
}

// returns whether there are still unserviced ptasks
int
HTask::unserviced()
{
    return serviced_ != no_ptasks_;
}

HTask::~HTask()
{
    cleanPTasks();
    delete[] velo_tstep_opt_;
    delete[] inip_tstep_opt_;
}

#include "BBoxAdmin.h"

void
HTask::AssignOctTrees(const BBoxAdmin *bboxAdmin)
{
    bboxAdmin->assignOctTrees(grid_tstep_opt_);
}
