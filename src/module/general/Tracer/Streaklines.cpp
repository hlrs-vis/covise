/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Streaklines.h"
#include "PPathline.h"
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

extern string speciesAttr;

Streaklines::Streaklines(
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
    : Pathlines(mod, name_line, name_magnitude, grid, velo, ini_p,
                number_per_tstep, x_ini, y_ini, z_ini, sfield)
{
    // correct number_per_tstep_ and no_ptasks_ if ini_p is not NULL
    // get number of elements
    if (ini_p_)
    {
        int num_elems = ((coDoSet *)(ini_p_))->getNumElements();

        if (num_elems <= 0)
        {
            number_per_tstep_ = 0;
            no_ptasks_ = 0;
        }
        else
        {
            // get the number of starting points at time 0 -> variable point_counter
            std::vector<const coDistributedObject *> inip_tstep;
            ExpandInipObjects(0, inip_tstep);
            int i; //object counter
            int point_counter = 0;
            for (i = 0; i < inip_tstep.size(); ++i)
            {
                if (!inip_tstep[i]->isType("POINTS"))
                    continue;
                coDoPoints *p_points = (coDoPoints *)(inip_tstep[i]);
                point_counter += p_points->getNumPoints();
            }
            number_per_tstep_ = point_counter;
            no_ptasks_ = point_counter;
        }
    }
}

extern float verschiebung[3];

void
Streaklines::gatherTimeStep()
{
    std::vector<float> x_coord;
    std::vector<float> y_coord;
    std::vector<float> z_coord;
    std::vector<float> mag;
    std::vector<int> vertices;
    std::vector<int> lineList;
    // number_per_tstep_
    int release;
    int line_label;
    int no_of_releases = no_ptasks_ / number_per_tstep_;
    // count number of lines, and vertices
    int numLines = 0;
    int numVertices = 0;

    vector<const coDistributedObject *> sfield;
    loadField(sfield);

    for (line_label = 0; line_label < number_per_tstep_; ++line_label)
    {
        char startLineFlag = 'y';
        for (release = 0; release < no_of_releases; ++release)
        {
            // get the last point of pathline with task order ==
            // release * number_per_tstep_ + line_label
            int task = release * number_per_tstep_ + line_label;
            PPathline *p_pathline = (PPathline *)(ptasks_[task]);
            // check if this PTask is still alive or not
            if (p_pathline->getKeepStatus() != PTask::NOT_SERVICED)
            {
                // terminated
                startLineFlag = 'y';
            }
            else
            {
                // still alive
                vertices.push_back(numVertices);
                float *x_c = p_pathline->x_c();
                float *y_c = p_pathline->y_c();
                float *z_c = p_pathline->z_c();
                float *m_c = p_pathline->m_c();
                if (sfield.size() > 0)
                {
                    m_c = p_pathline->m_c_interpolate(sfield, sfield);
                }
                int no_points = p_pathline->num_points();
                x_coord.push_back(x_c[no_points - 1]);
                y_coord.push_back(y_c[no_points - 1]);
                z_coord.push_back(z_c[no_points - 1]);
                mag.push_back(m_c[no_points - 1]);
                if (startLineFlag == 'y')
                {
                    startLineFlag = 'n';
                    lineList.push_back(numVertices);
                    ++numLines;
                }
                ++numVertices;
            }
        }
    }
    if (lines_.size() <= covise_time_)
        lines_.resize(covise_time_ + 1);
    std::string name = name_line_;
    char buf[64];
    sprintf(buf, "_%d", covise_time_);
    name += buf;

    lines_[covise_time_] = new coDoLines(name, numVertices,
                                         &x_coord[0],
                                         &y_coord[0],
                                         &z_coord[0],
                                         numVertices, &vertices[0],
                                         numLines, &lineList[0]);

    float *x_line, *y_line, *z_line;
    int *corner_list, *line_list;
    ((coDoLines *)(lines_[covise_time_]))->getAddresses(&x_line, &y_line, &z_line, &corner_list, &line_list);
    if (verschiebung[0] != 0.0
        || verschiebung[1] != 0.0
        || verschiebung[2] != 0.0)
    {
        Verschieben(verschiebung, numVertices,
                    x_line, y_line, z_line);
    }

    if (magnitude_.size() <= covise_time_)
        magnitude_.resize(covise_time_ + 1);
    // magnitude
    name = name_magnitude_;
    sprintf(buf, "_%d", covise_time_);
    name += buf;

    magnitude_[covise_time_] = new coDoFloat(name, numVertices, &mag[0]);
    if (speciesAttr.length() > 0)
        magnitude_[covise_time_]->addAttribute("SPECIES", speciesAttr.c_str());
}

// If we get dynamic coDoPoints, check that the number
// of starting positions is time-invariant
int
Streaklines::DiagnosePoints()
{
    if (ini_p_
        && ini_p_->isType("SETELE") // this is redundant...
        && ini_p_->getAttribute("TIMESTEP"))
    {
        // get number of elements
        int num_elems = ((coDoSet *)(ini_p_))->getNumElements();

        if (num_elems <= 0)
            return 0;

        // get the number of starting points at time 0 -> variable point_counter
        std::vector<const coDistributedObject *> inip_tstep;
        ExpandInipObjects(0, inip_tstep);
        int i; //object counter
        int point_counter = 0;
        for (i = 0; i < inip_tstep.size(); ++i)
        {
            if (!inip_tstep[i]->isType("POINTS"))
                continue;
            coDoPoints *p_points = (coDoPoints *)(inip_tstep[i]);
            point_counter += p_points->getNumPoints();
        }

        // check other time steps
        int covise_time;
        for (covise_time = 1; covise_time < num_elems; ++covise_time)
        {
            ExpandInipObjects(covise_time, inip_tstep);
            int point_counter_int = 0;
            for (i = 0; i < inip_tstep.size(); ++i)
            {
                if (!inip_tstep[i]->isType("POINTS"))
                    continue;
                coDoPoints *p_points = (coDoPoints *)(inip_tstep[i]);
                point_counter_int += p_points->getNumPoints();
            }
            if (point_counter_int != point_counter)
            {
                Covise::sendError("When creating streaklines, if the coDoPoints input object is dynamic, the number of initial points may not time-dependent");
                return -1;
            }
        }
    }
    return 0;
}
