/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <float.h>
#include "BBoxes.h"
#include <appl/ApplInterface.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoPolygons.h>

void
BBoxes::clean(int plane)
{
    orientation_[0] = 1.0;
    orientation_[1] = 1.0;
    orientation_[2] = 1.0;
    switch (plane)
    {
    case 0:
        orientation_[2] = 0.0;
        break;
    case 1:
        orientation_[0] = 0.0;
        break;
    case 2:
        orientation_[1] = 0.0;
        break;
    default:
        Covise::sendWarning("Forbidden option in BBoxes::clean, assuming XY plane");
        orientation_[2] = 0.0;
        break;
    }
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

// make room and initialse properly for computation
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
        bbox[MIN_Z] = FLT_MAX, bbox[MAX_Z] = -FLT_MAX;
        bbox[UXA] = 0.0;
        bbox[UYA] = 0.0;
        bbox[UZA] = 0.0;
        bbox[UXB] = 0.0;
        bbox[UYB] = 0.0;
        bbox[UZB] = 0.0;
    }
}

BBoxes::BBoxes()
{
    no_times_ = 0;
    bboxes_ = NULL;
}

BBoxes::~BBoxes()
{
    delete[] bboxes_;
}

// FillUBox is called when both displacements and geometry are availble
void
BBoxes::FillUBox(const float *x_c, const float *y_c, const float *z_c,
                 const float *u_x, const float *u_y, const float *u_z,
                 int no_points, int time)
{
    float *bbox = bboxes_[time];
    float AValue = orientation_[0] * bbox[MIN_X] + orientation_[1] * bbox[MIN_Y] + orientation_[2] * bbox[MIN_Z];
    float BValue = orientation_[0] * bbox[MAX_X] + orientation_[1] * bbox[MAX_Y] + orientation_[2] * bbox[MAX_Z];
    int point;
    for (point = 0; point < no_points; ++point)
    {
        float planeDistance = orientation_[0] * x_c[point] + orientation_[1] * y_c[point] + orientation_[2] * z_c[point];
        if (planeDistance < AValue)
        {
            AValue = planeDistance;
            bbox[MIN_X] = x_c[point];
            bbox[MIN_Y] = y_c[point];
            bbox[MIN_Z] = z_c[point];
            bbox[UXA] = u_x[point];
            bbox[UYA] = u_y[point];
            bbox[UZA] = u_z[point];
        }
        if (planeDistance > BValue)
        {
            BValue = planeDistance;
            bbox[MAX_X] = x_c[point];
            bbox[MAX_Y] = y_c[point];
            bbox[MAX_Z] = z_c[point];
            bbox[UXB] = u_x[point];
            bbox[UYB] = u_y[point];
            bbox[UZB] = u_z[point];
        }
    }
}

// only called when displacements are not availble
void
BBoxes::FillMBox(const float *x_c, const float *y_c, const float *z_c,
                 int no_points, int time)
{
    int point;
    float *bbox = bboxes_[time];
    for (point = 0; point < no_points; ++point)
    {
        if (x_c[point] > bbox[MAX_X])
            bbox[MAX_X] = x_c[point];
        if (y_c[point] > bbox[MAX_Y])
            bbox[MAX_Y] = y_c[point];
        if (z_c[point] > bbox[MAX_Z])
            bbox[MAX_Z] = z_c[point];
        if (x_c[point] < bbox[MIN_X])
            bbox[MIN_X] = x_c[point];
        if (y_c[point] < bbox[MIN_Y])
            bbox[MIN_Y] = y_c[point];
        if (z_c[point] < bbox[MIN_Z])
            bbox[MIN_Z] = z_c[point];
    }
}

// Structured, rectilinear and uniform grids
int
BBoxes::FillStrBBox(const coDistributedObject *inObj,
                    const coDoVec3 *shiftObj,
                    int time)
{
    float *x_c = NULL, *y_c = NULL, *z_c = NULL;
    float X_c[2];
    float Y_c[2];
    float Z_c[2];
    int xsize, ysize, zsize;
    if (inObj->isType("STRGRD"))
    {
        ((coDoStructuredGrid *)(inObj))->getAddresses(&x_c, &y_c, &z_c);
        ((coDoStructuredGrid *)(inObj))->getGridSize(&xsize, &ysize, &zsize);
    }
    else if (inObj->isType("UNIGRD"))
    {
        ((coDoUniformGrid *)(inObj))->getMinMax(&X_c[0], &X_c[1], &Y_c[0], &Y_c[1], &Z_c[0], &Z_c[1]);
        ((coDoUniformGrid *)(inObj))->getGridSize(&xsize, &ysize, &zsize);
    }
    else if (inObj->isType("RCTGRD"))
    {
        ((coDoRectilinearGrid *)(inObj))->getAddresses(&x_c, &y_c, &z_c);
        ((coDoRectilinearGrid *)(inObj))->getGridSize(&xsize, &ysize, &zsize);
        X_c[0] = x_c[0];
        X_c[1] = x_c[xsize - 1];
        Y_c[0] = y_c[0];
        Y_c[1] = y_c[ysize - 1];
        Z_c[0] = z_c[0];
        Z_c[1] = z_c[zsize - 1];
    }
    float *u_x = NULL, *u_y = NULL, *u_z = NULL;
    if (shiftObj)
    {
        int nelem = shiftObj->getNumPoints();
        if (xsize * ysize * zsize != nelem)
        {
            Covise::sendError("Number of points in geometry and shift object does not match");
            return -1;
        }
        shiftObj->getAddresses(&u_x, &u_y, &u_z);
        if (inObj->isType("STRGRD"))
        {
            FillUBox(x_c, y_c, z_c, u_x, u_y, u_z, xsize * ysize * zsize, time);
        }
        else
        {
            float U_x[2], U_y[2], U_z[2];
            int opNode = xsize * ysize * zsize - 1;
            U_x[0] = u_x[0];
            U_x[1] = u_x[opNode];
            U_y[0] = u_y[0];
            U_y[1] = u_y[opNode];
            U_z[0] = u_z[0];
            U_z[1] = u_z[opNode];
            FillUBox(X_c, Y_c, Z_c, U_x, U_y, U_z, 2, time);
        }
    }
    else
    {
        if (inObj->isType("STRGRD"))
        {
            FillMBox(x_c, y_c, z_c, xsize * ysize * zsize, time);
        }
        else
        {
            FillMBox(X_c, Y_c, Z_c, 2, time);
        }
    }
    return 0;
}

// Unstructured grids
int
BBoxes::FillUnstrBBox(const coDistributedObject *inObj,
                      const coDoVec3 *shiftObj,
                      int time)
{
    float *x_c = NULL, *y_c = NULL, *z_c = NULL;
    int *v_l = NULL, *l_l = NULL;
    int no_of_points = 0;

    if (inObj->isType("POLYGN"))
    {
        ((coDoPolygons *)(inObj))->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        no_of_points = ((coDoPolygons *)(inObj))->getNumPoints();
    }
    else if (inObj->isType("LINES"))
    {
        ((coDoLines *)(inObj))->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        no_of_points = ((coDoLines *)(inObj))->getNumPoints();
    }
    else if (inObj->isType("POINTS"))
    {
        ((coDoPoints *)(inObj))->getAddresses(&x_c, &y_c, &z_c);
        no_of_points = ((coDoPoints *)(inObj))->getNumPoints();
    }
    else if (inObj->isType("TRIANG"))
    {
        ((coDoTriangleStrips *)(inObj))->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        no_of_points = ((coDoTriangleStrips *)(inObj))->getNumPoints();
    }
    else if (inObj->isType("UNSGRD"))
    {
        ((coDoUnstructuredGrid *)(inObj))->getAddresses(&v_l, &l_l, &x_c, &y_c, &z_c);
        int e, c;
        ((coDoUnstructuredGrid *)(inObj))->getGridSize(&e, &c, &no_of_points);
    }
    else
    {
        fprintf(stderr, "BBoxes::fillUnstrBox: geom type not handled\n");
    }

    float *u_x = NULL, *u_y = NULL, *u_z = NULL;
    if (shiftObj
        && no_of_points != shiftObj->getNumPoints())
    {
        Covise::sendError("Number of points in geometry and shift object do not match, only the geometry is taken into account");
        return -1;
    }
    else if (shiftObj)
    {
        shiftObj->getAddresses(&u_x, &u_y, &u_z);
        FillUBox(x_c, y_c, z_c, u_x, u_y, u_z, no_of_points, time);
    }
    else
    {
        FillMBox(x_c, y_c, z_c, no_of_points, time);
    }
    return 0;
}

// interface with the module class
int
BBoxes::FillBBox(const coDistributedObject *geom, const coDistributedObject *disp, int time)
{
    if (geom->getAttribute("NO_BBOX"))
    {
        return 0;
    }
    // to be a set or not to be a set, that is the question
    if (geom->isType("SETELE"))
    {
        if (disp && !disp->isType("SETELE"))
        {
            Covise::sendError("A set in the geometry object has no valid displacement set");
            return -1;
        }
        int no_set_elems;
        const coDoSet *in_set_geom = (coDoSet *)(geom);
        const coDoSet *in_set_disp = (coDoSet *)(disp);
        const coDistributedObject *const *geom_objs = in_set_geom->getAllElements(&no_set_elems);
        const coDistributedObject *const *disp_objs = NULL;
        if (in_set_disp)
        {
            int no_set_elems_disp;
            disp_objs = in_set_disp->getAllElements(&no_set_elems_disp);
            if (no_set_elems_disp != no_set_elems)
            {
                Covise::sendError("BBoxes::FillBBox: Set structures does not match");
                return -1;
            }
        }
        int elem;
        for (elem = 0; elem < no_set_elems; ++elem)
        {
            int fail = 0;
            fail = FillBBox(geom_objs[elem], (in_set_disp) ? disp_objs[elem] : NULL,
                            time);
            if (fail != 0)
            {
                return -1;
            }
        }
    }
    else if (geom->isType("POLYGN")
             || geom->isType("LINES")
             || geom->isType("TRIANG")
             || geom->isType("POINTS")
             || geom->isType("UNSGRD"))
    {
        if (disp
            && !disp->isType("USTVDT"))
        {
            Covise::sendError("Incorrect data type in displacements, expected USTVDT");
            return -1;
        }
        return FillUnstrBBox(geom, (coDoVec3 *)(disp),
                             time);
    }
    else if (geom->isType("STRGRD")
             || geom->isType("RCTGRD")
             || geom->isType("UNIGRD"))
    {
        if (disp
            && !disp->isType("USTVDT"))
        {
            Covise::sendError("Incorrect data type in displacements, expected USTVDT");
            return -1;
        }
        return FillStrBBox(geom, (coDoVec3 *)(disp),
                           time);
    }
    return 0;
}

ostream &
operator<<(ostream &outfile, const BBoxes &tree)
{
    int time;
    for (time = 0; time < tree.getNoTimes(); ++time)
    {
        const float *bbox = tree.getBBox(time);
        outfile << time << ": " << bbox[BBoxes::MIN_X] << ' ' << bbox[BBoxes::MAX_X] << ' '
                << bbox[BBoxes::MIN_Y] << ' ' << bbox[BBoxes::MAX_Y] << bbox[BBoxes::UXA] << ' ' << bbox[BBoxes::UYA] << endl;
    }
    return outfile;
}
