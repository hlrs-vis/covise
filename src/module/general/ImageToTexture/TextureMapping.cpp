/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


//#include <api/coSimpleModule.h>
#include "TextureMapping.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

TextureMapping::Orientation
TextureMapping::getOrientation() const
{
    return orientation_;
}

TextureMapping::Fit
TextureMapping::getFit() const
{
    return fit_;
}

TextureMapping::SizeControl
TextureMapping::getSizeControl() const
{
    return mode_;
}

TextureMapping::TextureMapping(const TextureMapping &tm,
                               const int *indices,
                               int no_points)
    : no_points_(no_points)
{
    const float *map[2];
    tm.getMapping(&map[0], &map[1]);

    orientation_ = tm.getOrientation();
    fit_ = tm.getFit();
    mode_ = tm.getSizeControl();

    int point;
    txCoord_[0] = new float[no_points];
    txCoord_[1] = new float[no_points];
    for (point = 0; point < no_points; ++point)
    {
        txCoord_[0][point] = map[0][indices[point]];
        txCoord_[1][point] = map[1][indices[point]];
    }
}

TextureMapping::TextureMapping(const float *x_c,
                               const float *y_c,
                               int no_points,
                               Orientation orientation_hint,
                               Fit fit_hint,
                               SizeControl mode_hint,
                               float Width,
                               float Height,
                               const float *bbox,
                               int mirror)
    : orientation_(orientation_hint)
    , fit_(fit_hint)
    , mode_(mode_hint)
    , Width_(Width)
    , Height_(Height)
    , no_points_(no_points)
{
    if (bbox == NULL)
    {
        minX_ = FLT_MAX, maxX_ = -FLT_MAX, minY_ = FLT_MAX, maxY_ = -FLT_MAX;
        int point;
        for (point = 0; point < no_points; ++point)
        {
            if (x_c[point] > maxX_)
                maxX_ = x_c[point];
            if (y_c[point] > maxY_)
                maxY_ = y_c[point];
            if (x_c[point] < minX_)
                minX_ = x_c[point];
            if (y_c[point] < minY_)
                minY_ = y_c[point];
        }
    }
    else
    {
        minX_ = bbox[0];
        maxX_ = bbox[1];
        minY_ = bbox[2];
        maxY_ = bbox[3];
    }
    LimitMapping();

    txCoord_[0] = new float[no_points];
    txCoord_[1] = new float[no_points];

    float factx = limitX_ / (maxX_ - minX_);
    float facty = limitY_ / (maxY_ - minY_);
    int point;
    for (point = 0; point < no_points; point++)
    {
        float tx = (x_c[point] - minX_) * factx;
        float ty = (y_c[point] - minY_) * facty;
        int stepx = int(tx);
        int stepy = int(ty);
        if (mirror)
        {
            tx = (stepx % 2 == 0) ? tx - stepx : 1.0 - tx + stepx;
            ty = (stepy % 2 == 0) ? ty - stepy : 1.0 - ty + stepy;
        }
        else
        {
            tx = tx - stepx;
            ty = ty - stepy;
        }

        switch (orientation_)
        {
        case PORTRAIT:
            txCoord_[0][point] = tx;
            txCoord_[1][point] = ty;
            break;
        case LANDSCAPE:
            txCoord_[0][point] = ty;
            txCoord_[1][point] = tx;
            break;
        default:
            //         Covise::sendError("Bug in TextureMapping: No orientation could be calculated");
            delete[] txCoord_[0];
            delete[] txCoord_[1];
            txCoord_[0] = NULL;
            txCoord_[1] = NULL;
            break;
        }
        //      txIndex[point] = point;
    }
}

void
TextureMapping::getMapping(const float **mx,
                           const float **my) const
{
    *mx = txCoord_[0];
    *my = txCoord_[1];
}

void
TextureMapping::getMapping(const float **mx,
                           const float **my,
                           float minx, float maxx, float miny, float maxy) const
{
    *mx = txCoord_[0];
    *my = txCoord_[1];
    int point;
    for (point = 0; point < no_points_; ++point)
    {
        txCoord_[0][point] = minx + (maxx - minx) * txCoord_[0][point];
        txCoord_[1][point] = miny + (maxy - miny) * txCoord_[1][point];
    }
}

void
TextureMapping::LimitMapping()
{
    // can we fit the object in the image?
    float xx = (maxX_ - minX_) / Width_;
    float yy = (maxY_ - minY_) / Height_;
    float xy = (maxX_ - minX_) / Height_;
    float yx = (maxY_ - minY_) / Width_;
    if (fit_ == FIT // the user explicitly wants fit to geom
        || (fit_ == USE_IMAGE // or wants real size, which is not available
            && mode_ == MANUAL)) // fit image in geometry
    {
        limitX_ = 1.0;
        limitY_ = 1.0;

        if (orientation_ == FREE)
        {
            if (fabs(log(xx / yy)) > fabs(log(xy / yx)))
            {
                orientation_ = LANDSCAPE;
            }
            else
            {
                orientation_ = PORTRAIT;
            }
        }
        return;
    }

    if (orientation_ != FREE) // Fix an orientation;
    {
        if (orientation_ == PORTRAIT)
        {
            limitX_ = xx;
            limitY_ = yy;
        }
        else
        {
            limitX_ = xy;
            limitY_ = yx;
        }
        return;
    }

    if (xx <= 1.0 // orientation_ is FREE
        && yy <= 1.0)
    {
        orientation_ = PORTRAIT;
        limitX_ = xx;
        limitY_ = yy;
    }
    else if (xy <= 1.0
             && yx <= 1.0)
    {
        orientation_ = LANDSCAPE;
        limitX_ = xy;
        limitY_ = yx;
    }
    else // provisional
    {
        if (fabs(log(xx / yy)) > fabs(log(xy / yx)))
        {
            limitX_ = xy;
            limitY_ = yx;
            orientation_ = LANDSCAPE;
        }
        else
        {
            limitX_ = xx;
            limitY_ = yy;
            orientation_ = PORTRAIT;
        }
    }
}

TextureMapping::~TextureMapping()
{
    delete[] txCoord_[0];
    delete[] txCoord_[1];
}
