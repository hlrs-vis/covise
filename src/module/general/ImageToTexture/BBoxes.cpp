/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <stdio.h>
#include <float.h>
#include "BBoxes.h"
#include <appl/ApplInterface.h>

void
BBoxes::clean()
{
    no_times_ = 0;
    delete[] bboxes_;
    bboxes_ = NULL;
}

const float *
BBoxes::getBBox(int time) const
{
    if (bboxes_)
    {
        return bboxes_[time];
    }
    return NULL;
}

void
BBoxes::prepare(int no_times)
{
    no_times_ = no_times;
    bboxes_ = new BBox[no_times];
    int time;
    for (time = 0; time < no_times; ++time)
    {
        float *bbox = bboxes_[time];
        bbox[MIN_X] = FLT_MAX, bbox[MAX_X] = -FLT_MAX;
        bbox[MIN_Y] = FLT_MAX, bbox[MAX_Y] = -FLT_MAX;
    }
}

BBoxes::BBoxes()
{
    no_times_ = 0;
    bboxes_ = NULL;
}

BBoxes::~BBoxes()
{
    clean();
}

void
BBoxes::FillBBox(const coDoPolygons *inObj,
                 const coDoVec3 *shiftObj,
                 int time)
{
    int point;
    float *x_c, *y_c, *z_c;
    int *v_l, *l_l;
    float *bbox = bboxes_[time];
    inObj->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    float *u_x = NULL, *u_y = NULL, *u_z = NULL;
    if (shiftObj
        && inObj->getNumPoints() != shiftObj->getNumPoints())
    {
        Covise::sendError("Number of points in geometry and shift object do not match, only the geometry is taken into account");
    }
    else if (shiftObj)
    {
        shiftObj->getAddresses(&u_x, &u_y, &u_z);
    }
    if (!u_x)
    {
        for (point = 0; point < inObj->getNumPoints(); ++point)
        {
            if (x_c[point] > bbox[MAX_X])
                bbox[MAX_X] = x_c[point];
            if (y_c[point] > bbox[MAX_Y])
                bbox[MAX_Y] = y_c[point];
            if (x_c[point] < bbox[MIN_X])
                bbox[MIN_X] = x_c[point];
            if (y_c[point] < bbox[MIN_Y])
                bbox[MIN_Y] = y_c[point];
        }
    }
    else
    {
        for (point = 0; point < inObj->getNumPoints(); ++point)
        {
            if (x_c[point] - u_x[point] > bbox[MAX_X])
                bbox[MAX_X] = x_c[point] - u_x[point];
            if (y_c[point] - u_y[point] > bbox[MAX_Y])
                bbox[MAX_Y] = y_c[point] - u_y[point];
            if (x_c[point] - u_x[point] < bbox[MIN_X])
                bbox[MIN_X] = x_c[point] - u_x[point];
            if (y_c[point] - u_y[point] < bbox[MIN_Y])
                bbox[MIN_Y] = y_c[point] - u_y[point];
        }
    }
}

ostream &
operator<<(ostream &outfile, const BBoxes &tree)
{
    int time;
    for (time = 0; time < tree.getNoTimes(); ++time)
    {
        const float *bbox = tree.getBBox(time);
        outfile << time << ": " << bbox[BBoxes::MIN_X] << ' ' << bbox[BBoxes::MAX_X] << ' '
                << bbox[BBoxes::MIN_Y] << ' ' << bbox[BBoxes::MAX_Y] << endl;
    }
    return outfile;
}

void
BBoxes::FillBBox(const coDoSet *in_set, const coDoSet *in_shift_set, int time)
{
    int no_set_elems;
    const coDistributedObject *const *grid_objs = in_set->getAllElements(&no_set_elems);
    const coDistributedObject *const *shift_objs = NULL;
    if (in_shift_set)
    {
        int no_set_elems_shift;
        shift_objs = in_shift_set->getAllElements(&no_set_elems_shift);
        if (no_set_elems_shift != no_set_elems)
        {
            Covise::sendError("BBoxes::FillBBox: Set structured do not match, only gemetry is taken into account");
            shift_objs = NULL;
        }
    }
    int elem;
    for (elem = 0; elem < no_set_elems; ++elem)
    {
        if (grid_objs[elem]->isType("SETELE"))
        {
            if (shift_objs && !shift_objs[elem]->isType("SETELE"))
            {
                Covise::sendError("BBoxes::FillBBox: Set structured do not match, only gemetry is taken into account");
                FillBBox(dynamic_cast<const coDoSet *>(grid_objs[elem]), NULL, time);
            }
            else if (shift_objs)
            {
                FillBBox(dynamic_cast<const coDoSet *>(grid_objs[elem]),
                         dynamic_cast<const coDoSet *>(shift_objs[elem]), time);
            }
            else
            {
                FillBBox(dynamic_cast<const coDoSet *>(grid_objs[elem]), NULL, time);
            }
        }
        else if (grid_objs[elem]->isType("POLYGN"))
        {
            if (shift_objs && !shift_objs[elem]->isType("USTVDT"))
            {
                Covise::sendError("BBoxes::FillBBox: A vector data is expected and not found, using only geometry");
                FillBBox(dynamic_cast<const coDoPolygons *>(grid_objs[elem]), NULL, time);
            }
            else if (shift_objs)
            {
                FillBBox(dynamic_cast<const coDoPolygons *>(grid_objs[elem]),
                         dynamic_cast<const coDoVec3 *>(shift_objs[elem]),
                         time);
            }
            else
            {
                FillBBox(dynamic_cast<const coDoPolygons *>(grid_objs[elem]), NULL, time);
            }
        }
    }
}
