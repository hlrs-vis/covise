/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POLY_TO_CYL_H
#define _POLY_TO_CYL_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coSimpleModule.h>
using namespace covise;
#include <api/coHideParam.h>
#include <vector>

////// our class
class PolyToCyl : public coSimpleModule
{
private:
    /////////ports
    coInputPort *p_poly_in, *p_box_in;
    coOutputPort *p_poly_out, *p_surf_out, *p_inlet;

    coBooleanParam *p_torus; // whether a torus shall be made
    coFloatParam *p_barrelDiameter; // width of the torus

    virtual void postInst();
    coHideParam *h_barrelDiameter;
    vector<coHideParam *> hparams_;
    void useTransformAttribute(const coDistributedObject *);
    virtual void preHandleObjects(coInputPort **);

    coIntScalarParam *p_factor; // multiple every line point by this factor
    coBooleanParam *p_usefactor; // whether factor shall be used

    virtual int compute(const char *port);

    /// calculate bounding box
    void getBoundingBox(float *xmin, float *xmax,
                        float *ymin, float *ymax,
                        float *zmin, float *zmax,
                        const float *x, const float *y, const float *z,
                        int numVert);
    //void discreteLine( coDoLines *in, float **x, float **y, float **z, int **vl, int **ll);
    coDoLines *discreteLine(const char *new_name, coDoLines *in);

    coDoPolygons *movePolygons(coDoPolygons *in, float mx, float my, float mz, const char *outname);

public:
    PolyToCyl();
};

class Point
{
public:
    Point(float x0, float y0, float z0)
    {
        x_ = x0;
        y_ = y0;
        z_ = z0;
    };
    float x()
    {
        return x_;
    };
    float y()
    {
        return y_;
    };
    float z()
    {
        return z_;
    };

private:
    float x_, y_, z_;
};

class Torus
{
public:
    Torus();
    ~Torus();
    void addVertex(float xo, float yo, float z0, float xi, float yi, float zi);
    coDoPolygons *getObject(const char *obj_name);
    void reOrder();

private:
    std::vector<Point *> outer_line_;
    std::vector<Point *> inner_line_;

    int num_; //number of points
};
#endif
