/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Tracer.h"
#include <config/CoviseConfig.h>
#include <set>
#include <math.h>
#include <do/coDoPoints.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>

PTask::whatout Whatout;
string speciesAttr;

// find size of crew of worker threads
int
Tracer::findCrewSize()
{
#ifndef CO_hp1020
    int numNodes = coCoviseConfig::getInt("System.HostInfo.NumProcessors", 1);
    if ((numNodes < 1) || (numNodes > 200))
    {
        numNodes = 1;
    }
    return numNodes;
#else
    return 0;
#endif
}

// determine starting points (for a line of starting points)
void
Tracer::fillLine(float **x_ini, // array of output addresses
                 float **y_ini,
                 float **z_ini)
{
    int i;
    int no_p = p_no_startp->getValue();
    *x_ini = new float[no_p];
    *y_ini = new float[no_p];
    *z_ini = new float[no_p];
    float x1, y1, z1, x2, y2, z2, dx, dy, dz;
    x1 = p_startpoint1->getValue(0);
    y1 = p_startpoint1->getValue(1);
    z1 = p_startpoint1->getValue(2);
    (*x_ini)[0] = x1;
    (*y_ini)[0] = y1;
    (*z_ini)[0] = z1;
    if (no_p == 1)
        return;
    x2 = p_startpoint2->getValue(0);
    y2 = p_startpoint2->getValue(1);
    z2 = p_startpoint2->getValue(2);
    dx = (x2 - x1) / (no_p - 1.0f);
    dy = (y2 - y1) / (no_p - 1.0f);
    dz = (z2 - z1) / (no_p - 1.0f);
    for (i = 1; i < no_p; ++i)
    {
        (*x_ini)[i] = (*x_ini)[i - 1] + dx;
        (*y_ini)[i] = (*y_ini)[i - 1] + dy;
        (*z_ini)[i] = (*z_ini)[i - 1] + dz;
    }
    p_start->setCurrentObject(new coDoPoints(p_start->getObjName(), no_p, *x_ini, *y_ini, *z_ini));
}

// determine starting points (for a plane of starting points)
void
Tracer::fillSquare(float **x_ini, // array of output addresses
                   float **y_ini,
                   float **z_ini)
{
    int i, j;
    int n0, n1;
    int no_p = p_no_startp->getValue();
    float r, s;
    float s0, s1; // space between two divisions.
    float startDirection[3];
    // direction
    startDirection[0] = p_direction->getValue(0);
    startDirection[1] = p_direction->getValue(1);
    startDirection[2] = p_direction->getValue(2);
    // starting points and diff vector (v)
    float x1, y1, z1, x2, y2, z2;
    x1 = p_startpoint1->getValue(0);
    y1 = p_startpoint1->getValue(1);
    z1 = p_startpoint1->getValue(2);
    x2 = p_startpoint2->getValue(0);
    y2 = p_startpoint2->getValue(1);
    z2 = p_startpoint2->getValue(2);
    float v[3];
    v[0] = x2 - x1;
    v[1] = y2 - y1;
    v[2] = z2 - z1;
    // first normalise startDirection
    r = 1.0f / sqrt((startDirection[0] * startDirection[0]) + (startDirection[1] * startDirection[1]) + (startDirection[2] * startDirection[2]));
    startDirection[0] *= r;
    startDirection[1] *= r;
    startDirection[2] *= r;

    // work out projection of v onto direction
    r = v[0] * startDirection[0] + v[1] * startDirection[1] + v[2] * startDirection[2];

    // square sides
    float v0[3], v1[3];
    v0[0] = startDirection[0] * r;
    v0[1] = startDirection[1] * r;
    v0[2] = startDirection[2] * r;
    v1[0] = v[0] - v0[0];
    v1[1] = v[1] - v0[1];
    v1[2] = v[2] - v0[2];

    // side lengths
    r = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
    s = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    // and now number of points per side
    if (r > s)
    {
        n1 = int(sqrt(no_p * s / r)) + 1;
        if (n1 <= 1)
            n1 = 2;
        n0 = no_p / n1;
        if (n0 <= 1)
            n0 = 2;
    }
    else
    {
        n0 = int(sqrt(no_p * r / s)) + 1;
        if (n0 <= 1)
            n0 = 2;
        n1 = no_p / n0;
        if (n1 <= 1)
            n1 = 2;
    }
    no_p = n0 * n1;
    s0 = 1.0f / (n0 - 1);
    s1 = 1.0f / (n1 - 1);

    // now fill the grids
    *x_ini = new float[no_p];
    *y_ini = new float[no_p];
    *z_ini = new float[no_p];
    for (i = 0; i < n0; ++i)
    {
        for (j = 0; j < n1; ++j)
        {
            (*x_ini)[i * n1 + j] = x1 + v0[0] * s0 * i + v1[0] * s1 * j;
            (*y_ini)[i * n1 + j] = y1 + v0[1] * s0 * i + v1[1] * s1 * j;
            (*z_ini)[i * n1 + j] = z1 + v0[2] * s0 * i + v1[2] * s1 * j;
        }
    }
    p_no_startp->setValue(no_p);
    p_start->setCurrentObject(new coDoPoints(p_start->getObjName(), no_p, *x_ini, *y_ini, *z_ini));
}

void
Tracer::fillCylinder(float **x_ini, // array of output addresses
                     float **y_ini,
                     float **z_ini)
{
    int i, j;
    int n0, n1;
    int no_p = p_no_startp->getValue();
    float r, s;
    float s0, s1; // space between two divisions.
    float radius = p_cyl_radius->getValue();
    float height = p_cyl_height->getValue();

    // axis vector and point on bottom of cylinder
    float axis[3], axispoint[3];
    axis[0] = p_cyl_axis->getValue(0);
    axis[1] = p_cyl_axis->getValue(1);
    axis[2] = p_cyl_axis->getValue(2);
    axispoint[0] = p_cyl_axispoint->getValue(0);
    axispoint[1] = p_cyl_axispoint->getValue(1);
    axispoint[2] = p_cyl_axispoint->getValue(2);

    // first normalize axis vector
    r = 1.0f / sqrt((axis[0] * axis[0]) + (axis[1] * axis[1]) + (axis[2] * axis[2]));
    axis[0] *= r;
    axis[1] *= r;
    axis[2] *= r;

    // lengths
    s = (float)(2.0 * M_PI * radius);

    // and now number of points in axis direction and around cylinder
    if (height < 0.01 * r)
    {
        n0 = 1;
        s0 = 0.0;
        n1 = no_p;
    }
    else
    {
        n1 = int(sqrt(no_p * s / height)) + 1;
        if (n1 <= 1)
            n1 = 2;
        n0 = no_p / n1;
        if (n0 <= 1)
            n0 = 2;
        s0 = 1.0f / (n0 - 1);
    }

    no_p = n0 * n1;
    s1 = 1.0f / (n1 - 1);

    // we need two vectors at right angle to axis and to each other ...
    float vec1[3], vec2[3];
    float TOL = 0.0001f;
    // at first, determine one point p in plane
    int xsmall = 0;
    int ysmall = 0;
    //int zsmall=0;
    int nsmall = 0;
    if ((fabs(axis[0]) < TOL))
    {
        xsmall = 1;
        nsmall++;
    }
    if ((fabs(axis[1]) < TOL))
    {
        ysmall = 1;
        nsmall++;
    }
    if ((fabs(axis[2]) < TOL))
    {
        //zsmall=1;
        nsmall++;
    }
    if (nsmall == 0)
    {
        vec1[0] = 1.; // an arbitrary vector perpendicular to axis
        vec1[1] = 1.;
        vec1[2] = -(axis[0] + axis[1]) / axis[2];

        // 2nd vector is perpendicular to vec1 and to axis
        vec2[0] = axis[1] * vec1[2] - axis[2] * vec1[1];
        vec2[1] = axis[2] * vec1[0] - axis[0] * vec1[2];
        vec2[2] = axis[0] * vec1[1] - axis[1] * vec1[0];
    }
    else if (nsmall == 1)
    {
        if (xsmall)
        {
            vec1[0] = 1.;
            vec1[1] = 1.;
            vec1[2] = -(axis[0] + axis[1]) / axis[2];
        }
        else if (ysmall)
        {
            vec1[0] = -(axis[1] + axis[2]) / axis[0];
            vec1[1] = 1.;
            vec1[2] = 1.;
        }
        else // zsmall
        {
            vec1[0] = 1.;
            vec1[1] = -(axis[2] + axis[0]) / axis[1];
            vec1[2] = 1.;
        }
    }
    else if (nsmall == 2)
    {
        if (!xsmall)
        {
            vec1[0] = 0.;
            vec1[1] = 1.;
            vec1[2] = 0.;

            vec2[0] = 0.;
            vec2[1] = 0.;
            vec2[2] = 1.;
        }
        else if (!ysmall)
        {
            vec1[0] = 1.;
            vec1[1] = 0.;
            vec1[2] = 0.;

            vec2[0] = 0.;
            vec2[1] = 0.;
            vec2[2] = 1.;
        }
        else // !zsmall
        {
            vec1[0] = 1.;
            vec1[1] = 0.;
            vec1[2] = 0.;

            vec2[0] = 0.;
            vec2[1] = 1.;
            vec2[2] = 0.;
        }
    }
    else // nsmall==3
    {
        sendError("Your cylinder axis seems to be zero or close to zero. Please check.");
        return;
    }

    if (nsmall != 2)
    {
        // 2nd vector is perpendicular to vec1 and to axis
        vec2[0] = axis[1] * vec1[2] - axis[2] * vec1[1];
        vec2[1] = axis[2] * vec1[0] - axis[0] * vec1[2];
        vec2[2] = axis[0] * vec1[1] - axis[1] * vec1[0];

        // normalize vectors
        r = 1.0f / sqrt((vec1[0] * vec1[0]) + (vec1[1] * vec1[1]) + (vec1[2] * vec1[2]));
        vec1[0] *= r;
        vec1[1] *= r;
        vec1[2] *= r;
        r = 1.0f / sqrt((vec2[0] * vec2[0]) + (vec2[1] * vec2[1]) + (vec2[2] * vec2[2]));
        vec2[0] *= r;
        vec2[1] *= r;
        vec2[2] *= r;
    }

    // check vectors for rectangularity
    if ((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) > TOL)
    {
        sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
        return;
    }
    if ((vec1[0] * axis[0] + vec1[1] * axis[1] + vec1[2] * axis[2]) > TOL)
    {
        sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
        return;
    }
    if ((vec2[0] * axis[0] + vec2[1] * axis[1] + vec2[2] * axis[2]) > TOL)
    {
        sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
        return;
    }

    // now fill the grids
    *x_ini = new float[no_p];
    *y_ini = new float[no_p];
    *z_ini = new float[no_p];

    s0 *= height;
    float t = 0.;
    for (i = 0; i < n0; ++i)
    {
        for (j = 0; j < n1; ++j)
        {
            t = (float)(s1 * j * 2.0f * M_PI);
            (*x_ini)[i * n1 + j] = axispoint[0] + radius * vec1[0] * cos(t) + radius * vec2[0] * sin(t) + axis[0] * i * s0;
            (*y_ini)[i * n1 + j] = axispoint[1] + radius * vec1[1] * cos(t) + radius * vec2[1] * sin(t) + axis[1] * i * s0;
            (*z_ini)[i * n1 + j] = axispoint[2] + radius * vec1[2] * cos(t) + radius * vec2[2] * sin(t) + axis[2] * i * s0;
        }
    }
    p_no_startp->setValue(no_p);
    p_start->setCurrentObject(new coDoPoints(p_start->getObjName(), no_p, *x_ini, *y_ini, *z_ini));
}

// look for an attribute in an object (or recursively in children elements)
static string getAttribute(const coDistributedObject *obj, const char *attr_name)
{
    if (!obj)
        return "";
    const char *attr_val = obj->getAttribute(attr_name);
    if (attr_val) // attribute was found
    {
        return attr_val;
    }
    // there are no children to be inspected
    else if (!obj->isType("SETELE"))
    {
        return "";
    }
    // obj is a set
    int no_elems;
    const coDistributedObject *const *setList = ((coDoSet *)obj)->getAllElements(&no_elems);
    int i;
    string attrVal;
    for (i = 0; i < no_elems; ++i)
    {
        attrVal = getAttribute(setList[i], attr_name);
        if (attrVal.length() > 0)
            break;
    }
    // cleanup
    std::set<const coDistributedObject *> toBeDeleted(setList, setList + no_elems);
    std::set<const coDistributedObject *>::iterator it = toBeDeleted.begin();
    std::set<const coDistributedObject *>::iterator it_end = toBeDeleted.end();
    for (; it != it_end; ++it)
    {
        delete (*it);
    }
    delete[] setList;
    return attrVal;
}

// determine which magnitude we want to output
void
Tracer::fillWhatOut()
{
    if (p_field->getCurrentObject())
    {
        speciesAttr = getAttribute(p_field->getCurrentObject(), "SPECIES");
        return;
    }
    switch (p_whatout->getValue())
    {
    case PTask::V:
        Whatout = PTask::V;
        speciesAttr = "Velocity";
        break;
    case PTask::VX:
        Whatout = PTask::VX;
        speciesAttr = "Velocity-X";
        break;
    case PTask::VY:
        Whatout = PTask::VY;
        speciesAttr = "Velocity-Y";
        break;
    case PTask::VZ:
        Whatout = PTask::VZ;
        speciesAttr = "Velocity-Z";
        break;
    case PTask::TIME:
        Whatout = PTask::TIME;
        speciesAttr = "Time";
        break;
    case PTask::ID:
        Whatout = PTask::ID;
        speciesAttr = "Id";
        break;
    case PTask::V_VEC:
        Whatout = PTask::V_VEC;
        speciesAttr = "VelocityVec";
        break;
    default:
        Whatout = PTask::V;
        p_whatout->setValue(PTask::V);
        sendWarning("Setting whatout to velocity magnitude option");
        speciesAttr = "Velocity";
        break;
    }
}

coDoPoints *
Tracer::extractPoints(const coDistributedObject *obj)
{
    if (!obj)
    {
        return NULL;
    }

    if (obj->isType("POINTS"))
    {
        return (coDoPoints *)obj;
    }
    else
    {
        float *x = NULL, *y = NULL, *z = NULL;
        xValues = NULL;
        yValues = NULL;
        zValues = NULL;
        int num_points = 0;
        int *dummy, nn;
        numExtractedStartpoints = 0;
        if (obj->isType("UNSGRD"))
        {
            ((coDoUnstructuredGrid *)obj)->getAddresses(&dummy, &dummy, &x, &y, &z);
            ((coDoUnstructuredGrid *)obj)->getGridSize(&nn, &nn, &num_points);
        }
        else if (obj->isType("LINES"))
        {
            ((coDoLines *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
            num_points = ((coDoLines *)obj)->getNumPoints();
        }
        else if (obj->isType("POLYGN"))
        {
            ((coDoPolygons *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
            num_points = ((coDoPolygons *)obj)->getNumPoints();
        }
        else if (obj->isType("TRIANG"))
        {
            ((coDoTriangleStrips *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
            num_points = ((coDoTriangleStrips *)obj)->getNumPoints();
        }
        else if (obj->isType("USTVDT"))
        {
            ((coDoVec3 *)obj)->getAddresses(&x, &y, &z);
            num_points = ((coDoVec3 *)obj)->getNumPoints();
        }
        else if (obj->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *setList = ((coDoSet *)(obj))->getAllElements(&no_elems);
#ifdef VISENSO
            // backward compatibility to 6.0
            addPoints(setList[0]);
#else
            int elem;
            for (elem = 0; elem < no_elems; ++elem)
            {
                addPoints(setList[elem]);
            }
#endif
        }
        coDoPoints *ret = NULL;
        if (numExtractedStartpoints != 0)
        {
            ret = new coDoPoints(p_start->getObjName(), numExtractedStartpoints, xValues, yValues, zValues);
            delete[] xValues;
            delete[] yValues;
            delete[] zValues;
        }
        else if (x == NULL || y == NULL || z == NULL)
        {
            // Covise::sendWarning("Received unsupported object type at port pointsIn. Please check.");
            return NULL;
        }
        else
        {
            ret = new coDoPoints(p_start->getObjName(), num_points, x, y, z);
        }
        if (!ret->objectOk())
        {
            return NULL;
        }
        p_start->setCurrentObject(ret);
        return ret;
    }
}
void Tracer::addPoints(const coDistributedObject *obj)
{
    float *x = NULL, *y = NULL, *z = NULL;
    int num_points;
    int *dummy, nn;
    if (obj->isType("UNSGRD"))
    {
        ((coDoUnstructuredGrid *)obj)->getAddresses(&dummy, &dummy, &x, &y, &z);
        ((coDoUnstructuredGrid *)obj)->getGridSize(&nn, &nn, &num_points);
    }
    else if (obj->isType("LINES"))
    {
        ((coDoLines *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
        num_points = ((coDoLines *)obj)->getNumPoints();
    }
    else if (obj->isType("POLYGN"))
    {
        ((coDoPolygons *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
        num_points = ((coDoPolygons *)obj)->getNumPoints();
    }
    else if (obj->isType("TRIANG"))
    {
        ((coDoTriangleStrips *)obj)->getAddresses(&x, &y, &z, &dummy, &dummy);
        num_points = ((coDoTriangleStrips *)obj)->getNumPoints();
    }
    else if (obj->isType("USTVDT"))
    {
        ((coDoVec3 *)obj)->getAddresses(&x, &y, &z);
        num_points = ((coDoVec3 *)obj)->getNumPoints();
    }
    else if (obj->isType("SETELE"))
    {
        int no_elems;
        const coDistributedObject *const *setList = ((coDoSet *)(obj))->getAllElements(&no_elems);
        int elem;
        for (elem = 0; elem < no_elems; ++elem)
        {
            addPoints(setList[elem]);
        }
    }

    if (x == NULL || y == NULL || z == NULL)
    {
        // Covise::sendWarning("Received unsupported object type at port pointsIn. Please check.");
        return;
    }
    float *tx = xValues, *ty = yValues, *tz = zValues;
    xValues = new float[numExtractedStartpoints + num_points];
    yValues = new float[numExtractedStartpoints + num_points];
    zValues = new float[numExtractedStartpoints + num_points];
    memcpy(xValues, tx, numExtractedStartpoints * sizeof(float));
    memcpy(yValues, ty, numExtractedStartpoints * sizeof(float));
    memcpy(zValues, tz, numExtractedStartpoints * sizeof(float));
    memcpy(xValues + numExtractedStartpoints, x, num_points * sizeof(float));
    memcpy(yValues + numExtractedStartpoints, y, num_points * sizeof(float));
    memcpy(zValues + numExtractedStartpoints, z, num_points * sizeof(float));
    numExtractedStartpoints += num_points;
    delete[] tx;
    delete[] ty;
    delete[] tz;
}
