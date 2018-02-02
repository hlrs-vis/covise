/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Tracer.h"
#include "Tracelines.h"
#include "PTraceline.h"
#include "HTask.h"
#include <set>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>

extern string speciesAttr;
extern PTask::whatout Whatout;

// self explanatory
void
Tracelines::WarnIfOutOfDomain()
{
    if (warnOutOfDomain_ == 0)
    {
        bool someInsideDomain = false;
        bool finished = true;
        for (int task = 0; task < no_ptasks_; ++task)
        {
            PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
            if (p_traceline->get_status() == PTask::FINISHED_DOMAIN)
            {
                if (p_traceline->num_points() == 0)
                {
                    warnOutOfDomain_ = 1;
                    if (someInsideDomain)
                        break;
                }
                else
                {
                    someInsideDomain = true;
                }
            }
            else
            {
                finished = false;
            }
        }
        if (finished)
        {
            if (warnOutOfDomain_)
            {
                if (someInsideDomain)
                    Covise::sendWarning("Some initial point(s) are out of the domain");
                else
                    Covise::sendWarning("All initial points are out of the domain");
            }
        }
        else
        {
            warnOutOfDomain_ = 0;
        }
    }
}

// used in the construction of lines (corner list)
int *
Tracelines::cornerList()
{
    int *corner_list;
    int max_num_of_corners = 0;
    int i;

    // find the maximum length of all lines for this time step
    // (used when creating the corner list for DO_LINES)
    for (i = 0; i < no_ptasks_; ++i)
    {
        PTraceline *p_traceline = (PTraceline *)(ptasks_[i]);
        if (max_num_of_corners < p_traceline->num_points())
            max_num_of_corners = p_traceline->num_points();
    }
    corner_list = new int[max_num_of_corners];
    for (i = 0; i < max_num_of_corners; ++i)
    {
        corner_list[i] = i;
    }

    return corner_list;
}

extern int startStyle; // lines or plane
extern float verschiebung[3];

// makes lines for a time (used in gatherTimeStep)
#ifdef _ONE_LINE_PER_DO_LINES_
void
Tracelines::linesForATime()
{
    int i;
    char buf[64];
    int zero = 0;
    std::string name;

    int *corner_list = cornerList();
    coDistributedObject **set_list = new coDistributedObject *[no_ptasks_ + 1];
    set_list[no_ptasks_] = 0;

    for (i = 0; i < no_ptasks_; ++i)
    {
        PTraceline *p_traceline = (PTraceline *)(ptasks_[i]);
        name = name_line_;
        if (hasTimeSteps())
        {
            sprintf(buf, "_%d", covise_time_);
            name += buf;
        }
        sprintf(buf, "_%d", i);
        name += buf;
        set_list[i] = new coDoLines(name, p_traceline->num_points(),
                                    p_traceline->x_c(), p_traceline->y_c(), p_traceline->z_c(),
                                    p_traceline->num_points(), corner_list, 1, &zero);

        float *x_line, *y_line, *z_line;
        int *corner_list, *line_list;
        ((coDoLines *)(set_list[i]))->getAddresses(&x_line, &y_line, &z_line, &corner_list, &line_list);
        if (verschiebung[0] != 0.0
            || verschiebung[1] != 0.0
            || verschiebung[2] != 0.0)
        {
            Verschieben(verschiebung, p_traceline->num_points(),
                        x_line, y_line, z_line);
        }
    }
    name = name_line_;
    if (hasTimeSteps())
    {
        sprintf(buf, "_%d", covise_time_);
        name += buf;
    }
    if (lines_.size() <= covise_time_)
        lines_.resize(covise_time_ + 1);
    lines_[covise_time_] = new coDoSet(name, set_list);
    for (i = 0; i < no_ptasks_; ++i)
    {
        delete set_list[i];
    }
    delete[] set_list;
    delete[] corner_list;
}

// makes magnitude objects for a time (used in gatherTimeStep)
void
Tracelines::magForATime()
{
    int i;
    char buf[64];
    std::string name;

    coDistributedObject **set_list = new coDistributedObject *[no_ptasks_ + 1];
    set_list[no_ptasks_] = 0;
    for (i = 0; i < no_ptasks_; ++i)
    {
        PTraceline *p_traceline = (PTraceline *)(ptasks_[i]);
        name = name_magnitude_;
        if (hasTimeSteps())
        {
            sprintf(buf, "_%d", covise_time_);
            name += buf;
        }
        sprintf(buf, "_%d", i);
        name += buf;
        set_list[i] = new coDoFloat(name,
                                    p_traceline->num_points(), p_traceline->m_c());
    }
    name = name_magnitude_;
    if (hasTimeSteps())
    {
        sprintf(buf, "_%d", covise_time_);
        name += buf;
    }
    if (magnitude_.size() <= covise_time_)
        magnitude_.resize(covise_time_ + 1);
    magnitude_[covise_time_] = new coDoSet(name, set_list);
    for (i = 0; i < no_ptasks_; ++i)
    {
        delete set_list[i];
    }
    delete[] set_list;
}

// makes lines for a time (used in gatherTimeStep)
#else
void
Tracelines::linesForATime()
{
    int task;
    std::string name;

    // first we have to calculate the no of points, corners & lines...
    int no_of_points = 0;
    // no of corners is = no_of_points
    // no_ptasks_ is the number of lines
    int numEmpty = 0;
    for (task = 0; task < no_ptasks_; ++task)
    {
        PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
        no_of_points += p_traceline->num_points();

        if (p_traceline->num_points() == 0)
        {
            numEmpty++;
        }
    }

    if (lines_.size() <= covise_time_)
        lines_.resize(covise_time_ + 1);
    name = name_line_;
    if (hasTimeSteps())
    {
        char buf[64];
        sprintf(buf, "_%d", covise_time_);
        name += buf;
    }
    lines_[covise_time_] = new coDoLines(name, no_of_points, no_of_points, no_ptasks_ - numEmpty);
    if (no_of_points == 0)
    {
        // all lines are empty - probably all starting points outside
        return;
    }

    // this is the line list before we get rid of the empty lines
    int *orig_line_list = new int[no_ptasks_];

    // now fill in the arrays
    float *x_line, *y_line, *z_line;
    int *corner_list, *line_list;
    ((coDoLines *)(lines_[covise_time_]))->getAddresses(&x_line, &y_line, &z_line, &corner_list, &line_list);
    int point = 0;
    int numOutside = 0;
    bool lastline = false;

    for (task = 0; task < no_ptasks_; ++task)
    {
        if (point < no_of_points)
        {
            orig_line_list[task] = point;
        }
        else
        {
            orig_line_list[task] = no_of_points - 1;
        }

        //if (point < no_of_points)
        if (task == 0)
        {
            // at least one line has to be generated (so that Cover interactor can work), this might be an empty line
            line_list[task] = point;
        }
        else
        {
            if (point != orig_line_list[task - 1])
            {
                if (point < no_of_points)
                {
                    // this is a line with a starting point inside!
                    line_list[task - numOutside] = point;
                }
                else if (!lastline) // avoid problems when last starting points are outside
                {
                    // this is the last line ...
                    line_list[task - numOutside] = no_of_points - 1;
                    lastline = true;
                }
            }
            else
            {
                numOutside++;
            }
        }

        PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
        int corner;
        memcpy(x_line + point, p_traceline->x_c(),
               sizeof(float) * p_traceline->num_points());
        memcpy(y_line + point, p_traceline->y_c(),
               sizeof(float) * p_traceline->num_points());
        memcpy(z_line + point, p_traceline->z_c(),
               sizeof(float) * p_traceline->num_points());
        for (corner = 0; corner < p_traceline->num_points(); ++corner, ++point)
        {
            corner_list[point] = point;
        }
    }

    delete[] orig_line_list;

    if (verschiebung[0] != 0.0
        || verschiebung[1] != 0.0
        || verschiebung[2] != 0.0)
    {
        Verschieben(verschiebung, no_of_points, x_line, y_line, z_line);
    }
}

void
Tracelines::Verschieben(const float *shift, int no_p,
                        float *x, float *y, float *z)
{
    int point;
    for (point = 0; point < no_p; ++point)
    {
        x[point] += shift[0];
        y[point] += shift[1];
        z[point] += shift[2];
    }
}

// makes magnitude objects for a time (used in gatherTimeStep)
void
Tracelines::magForATime()
{
    int task;
    std::string name;

    // first we have to calculate the no of points
    int no_of_points = 0;
    for (task = 0; task < no_ptasks_; ++task)
    {
        PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
        no_of_points += p_traceline->num_points();
    }

    vector<const coDistributedObject *> sfield;
    loadField(sfield); // loads field

    if (magnitude_.size() <= covise_time_)
        magnitude_.resize(covise_time_ + 1);
    name = name_magnitude_;
    if (hasTimeSteps())
    {
        char buf[64];
        sprintf(buf, "_%d", covise_time_);
        name += buf;
    }
    if (Whatout != PTask::V_VEC)
        magnitude_[covise_time_] = new coDoFloat(name, no_of_points);
    else
        magnitude_[covise_time_] = new coDoVec3(name, no_of_points);
    if (speciesAttr.length() > 0)
        magnitude_[covise_time_]->addAttribute("SPECIES", speciesAttr.c_str());
    float *mag_array = NULL;
    float *mag_array2 = NULL;
    float *mag_array3 = NULL;
    if (Whatout != PTask::V_VEC)
        ((coDoFloat *)(magnitude_[covise_time_]))->getAddress(&mag_array);
    else
        ((coDoVec3 *)(magnitude_[covise_time_]))->getAddresses(&mag_array, &mag_array2, &mag_array3);

    if (sfield.size() == 0)
    {
        if (Whatout != PTask::V_VEC)
        {
            int point = 0;
            for (task = 0; task < no_ptasks_; ++task)
            {
                PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
                memcpy(mag_array + point, p_traceline->m_c(),
                       sizeof(float) * p_traceline->num_points());
                point += p_traceline->num_points();
            }
        }
        else
        {
            int point = 0;
            for (task = 0; task < no_ptasks_; ++task)
            {
                PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
                memcpy(mag_array + point, p_traceline->u_c(), sizeof(float) * p_traceline->num_points());
                memcpy(mag_array2 + point, p_traceline->v_c(), sizeof(float) * p_traceline->num_points());
                memcpy(mag_array3 + point, p_traceline->w_c(), sizeof(float) * p_traceline->num_points());
                point += p_traceline->num_points();
            }
        }
    }
    else
    {
        int point = 0;
        for (task = 0; task < no_ptasks_; ++task)
        {
            PTraceline *p_traceline = (PTraceline *)(ptasks_[task]);
            memcpy(mag_array + point, p_traceline->m_c_interpolate(sfield, sfield),
                   sizeof(float) * p_traceline->num_points());
            point += p_traceline->num_points();
        }
    }
}
#endif

// make a static set of lines with the finished PStreamline objects
// also a static set with the chosen magnitude
void
Tracelines::gatherTimeStep()
{
    linesForATime();
    magForATime();
}

// Gather results from all all time steps.
void
Tracelines::gatherAll(coOutputPort *p_line,
                      coOutputPort *p_mag)
{
    int i;
    coDoSet *set_obj;
    char buf[16];
    // use the std::vector's lines_ and magnitude_...
    if (hasTimeSteps())
    {
        sprintf(buf, "1 %lu", (unsigned long)lines_.size());
        coDistributedObject **set_list;
        set_list = new coDistributedObject *[lines_.size() + 1];
        set_list[lines_.size()] = 0;
        // lines
        for (i = 0; i < lines_.size(); ++i)
        {
            set_list[i] = lines_[i];
        }
        set_obj = new coDoSet(name_line_, set_list);
        set_obj->addAttribute("TIMESTEP", buf);
        p_line->setCurrentObject(set_obj);
        for (i = 0; i < lines_.size(); ++i)
            delete set_list[i];
        // magnitude
        for (i = 0; i < lines_.size(); ++i)
        {
            set_list[i] = magnitude_[i];
        }
        set_obj = new coDoSet(name_magnitude_, set_list);
        set_obj->addAttribute("TIMESTEP", buf);
        p_mag->setCurrentObject(set_obj);
        for (i = 0; i < lines_.size(); ++i)
            delete set_list[i];
        delete[] set_list;
    }
    else
    {
        p_line->setCurrentObject(lines_[0]);
        p_mag->setCurrentObject(magnitude_[0]);
    }
}

// see header
Tracelines::Tracelines(
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
    : HTask(mod, name_line, name_magnitude, grid, velo, ini_p)
    , sfield_(sfield)
{
    warnOutOfDomain_ = 0;
    covise_time_ = -1;
    no_ptasks_ = number_per_tstep;
    number_per_tstep_ = number_per_tstep;
    x_ini_ = x_ini; // delete in destructor
    y_ini_ = y_ini;
    z_ini_ = z_ini;
    // fixme what's this?
    char interaction[1000];

    if (startStyle == Tracer::SQUARE)
    {
        sprintf(interaction, "P%s\n%s\n%s\n", Covise::get_module(),
                Covise::get_instance(), Covise::get_host());
    }
    else if (startStyle == Tracer::LINE)
    {
        sprintf(interaction, "T%s\n%s\n%s\n", Covise::get_module(),
                Covise::get_instance(), Covise::get_host());
    }
    else
    {
        interaction[0] = '\0';
    }
    interaction_ = interaction;
}

extern int cycles;

// Are all time steps processed?
int
Tracelines::Finished()
{
    return (covise_time_ >= no_steps_ - 1);
}

void
Tracelines::loadField(vector<const coDistributedObject *> &sfield)
{
    if (!sfield_)
        return;
    vector<const coDistributedObject *> l_sfield;
    std::vector<const coDistributedObject *> field_tstep;

    const coDistributedObject *this_field = sfield_;
    if (hasTimeSteps())
    {
        int no_elems;
        const coDistributedObject *const *field_objects = NULL;
        field_objects = ((coDoSet *)(sfield_))->getAllElements(&no_elems);
        this_field = field_objects[covise_time_];
        // other objs in field_objects may be deleted
        set<const coDistributedObject *> toBeDeleted;
        int i;
        for (i = 0; i < no_elems; ++i)
        {
            if (field_objects[i] != this_field)
                toBeDeleted.insert(field_objects[i]);
        }
        set<const coDistributedObject *>::iterator it = toBeDeleted.begin();
        set<const coDistributedObject *>::iterator it_end = toBeDeleted.end();
        for (; it != it_end; ++it)
        {
            delete (*it);
        }
        delete[] field_objects;
    }
    if (this_field->isType("SETELE"))
    {
        int no_elems;
        const coDistributedObject *const *field_objects = ((coDoSet *)(this_field))->getAllElements(&no_elems);

        ExpandSetList(field_objects, no_elems, field_tstep);
        for (int i = 0; i < field_tstep.size(); ++i)
        {
            l_sfield.push_back(field_tstep[i]);
        }
    }
    else
    {
        l_sfield.push_back(this_field);
    }

    l_sfield.swap(sfield);
}

int
Tracelines::Diagnose()
{
    int ret = HTask::Diagnose();
    if (ret != 0)
        return ret;
    // check sfield_
    if (!sfield_)
        return 0;
    if (hasTimeSteps())
    {
        if (!sfield_->isType("SETELE") || !sfield_->getAttribute("TIMESTEP"))
        {
            Covise::sendError("External scalar field to be mapped is static.");
            return -1;
        }
        int no_elems = ((coDoSet *)(sfield_))->getNumElements();
        if (no_steps_ != no_elems)
        {
            Covise::sendError("The number of time steps of the field to be mapped is not correct.");
            return -1;
        }
    }
    else
    {
        if (sfield_->isType("SETELE") && sfield_->getAttribute("TIMESTEP"))
        {
            Covise::sendError("The grid is static, but the field to be mapped is dynamic.");
            return -1;
        }
    }

    // now check how many objects are found per time step
    int i;
    const coDistributedObject *const *field_objects = NULL;
    if (hasTimeSteps())
    {
        int no_elems;
        field_objects = ((coDoSet *)(sfield_))->getAllElements(&no_elems);
    }
    for (i = 0; i < no_steps_; ++i)
    {
        std::vector<const coDistributedObject *> grid_tstep;
        std::vector<const coDistributedObject *> field_tstep;
        ExpandGridObjects(i, grid_tstep);
        const coDistributedObject *thisSField = sfield_;
        if (field_objects)
        {
            thisSField = field_objects[i];
        }
        if (thisSField->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *theseSFields = ((coDoSet *)(thisSField))->getAllElements(&no_elems);
            ExpandSetList(theseSFields, no_elems, field_tstep);
        }
        else
        {
            field_tstep.push_back(thisSField);
        }
        size_t no_elems = grid_tstep.size();
        if (no_elems != field_tstep.size())
        {
            Covise::sendError("Different number of grids and field objects at time %d.", i);
            return -1;
        }
        // match grid objects and fields
        size_t obj_at_time;
        for (obj_at_time = 0; obj_at_time < no_elems; ++obj_at_time)
        {
            // get field size
            int size = -1;
            if (field_tstep[obj_at_time]->isType("USTSDT"))
            {
                size = ((coDoFloat *)field_tstep[obj_at_time])->getNumPoints();
            }
            else
            {
                Covise::sendError("Input field is not scalar.");
                return -1;
            }
            // get grid size
            if (grid_tstep[obj_at_time]->isType("UNSGRD"))
            {
                int e, c, p_g;
                ((coDoUnstructuredGrid *)(grid_tstep[obj_at_time]))->getGridSize(&e, &c, &p_g);
                if (p_g != size)
                {
                    Covise::sendError("Non-matching grid and input field objects");
                    return -1;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("POLYGN"))
            {
                int p_g = ((coDoPolygons *)grid_tstep[obj_at_time])->getNumPoints();
                if (p_g != size)
                {
                    Covise::sendError("Non-matching grid and input field objects");
                    return -1;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("UNIGRD"))
            {
                int x_s, y_s, z_s;
                ((coDoUniformGrid *)(grid_tstep[obj_at_time]))->getGridSize(&x_s, &y_s, &z_s);
                if (x_s * y_s * z_s != size)
                {
                    Covise::sendError("Non-matching grid and input field objects");
                    return -1;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("RCTGRD"))
            {
                int x_s, y_s, z_s;
                ((coDoRectilinearGrid *)(grid_tstep[obj_at_time]))->getGridSize(&x_s, &y_s, &z_s);
                if (x_s * y_s * z_s != size)
                {
                    Covise::sendError("Non-matching grid and input field objects");
                    return -1;
                }
            }
            else if (grid_tstep[obj_at_time]->isType("STRGRD"))
            {
                int x_s, y_s, z_s;
                ((coDoStructuredGrid *)(grid_tstep[obj_at_time]))->getGridSize(&x_s, &y_s, &z_s);
                if (x_s * y_s * z_s != size)
                {
                    Covise::sendError("Non-matching grid and input field objects");
                    return -1;
                }
            }
            else
            {
                Covise::sendError("Incorrect grid type: %s", grid_tstep[obj_at_time]->getType());
                return -1;
            }
        }
    }
    if (field_objects)
    {
        set<const coDistributedObject *> toBeDeleted;
        for (i = 0; i < no_steps_; ++i)
        {
            toBeDeleted.insert(field_objects[i]);
        }
        set<const coDistributedObject *>::iterator it = toBeDeleted.begin();
        set<const coDistributedObject *>::iterator it_end = toBeDeleted.end();
        for (; it != it_end; ++it)
        {
            delete (*it);
        }
        delete[] field_objects;
    }
    return 0;
}
