/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Geometry.h"
#include <util/coviseCompat.h>

// see header for comments

Geometry::Geometry()
    : type_(NPI)
{
    x_c_ = y_c_ = z_c_ = NULL;
    x_o_c_ = y_o_c_ = z_o_c_ = NULL;
    no_ = nox_ = noy_ = noz_ = 0;
    minx_ = miny_ = minz_ = maxx_ = maxy_ = maxz_ = 0.0;
    minx_o_ = miny_o_ = minz_o_ = maxx_o_ = maxy_o_ = maxz_o_ = 0.0;
}

Geometry::~Geometry()
{
}

void
Geometry::setInfo(float *xc, float *yc, float *zc, int no,
                  float *xoc, float *yoc, float *zoc)
{
    type_ = UNS;
    x_c_ = xc;
    y_c_ = yc;
    z_c_ = zc;
    x_o_c_ = xoc;
    y_o_c_ = yoc;
    z_o_c_ = zoc;
    no_ = no;
}

void
Geometry::setInfo(float *xc, float *yc, float *zc, int nx, int ny, int nz,
                  float *xoc, float *yoc, float *zoc)
{
    type_ = STR;
    x_c_ = xc;
    y_c_ = yc;
    z_c_ = zc;
    x_o_c_ = xoc;
    y_o_c_ = yoc;
    z_o_c_ = zoc;
    nox_ = nx;
    noy_ = ny;
    noz_ = nz;
    no_ = nx * ny * nz;
}

void
Geometry::setInfo(float *xc, int nx, float *yc, int ny, float *zc, int nz,
                  float *xoc, float *yoc, float *zoc)
{
    type_ = RCT;
    x_c_ = xc;
    y_c_ = yc;
    z_c_ = zc;
    x_o_c_ = xoc;
    y_o_c_ = yoc;
    z_o_c_ = zoc;
    nox_ = nx;
    noy_ = ny;
    noz_ = nz;
}

void
Geometry::setInfo(float minx, float maxx, float miny, float maxy,
                  float minz, float maxz, int nx, int ny, int nz,
                  float minxo, float maxxo, float minyo, float maxyo,
                  float minzo, float maxzo)
{
    type_ = UNI;
    minx_ = minx;
    miny_ = miny;
    minz_ = minz;
    maxx_ = maxx;
    maxy_ = maxy;
    maxz_ = maxz;

    minx_o_ = minxo;
    miny_o_ = minyo;
    minz_o_ = minzo;
    maxx_o_ = maxxo;
    maxy_o_ = maxyo;
    maxz_o_ = maxzo;

    nox_ = nx;
    noy_ = ny;
    noz_ = nz;
}

Geometry::GeomType
Geometry::getType() const
{
    return type_;
}

int
Geometry::getSize() const
{
    if (type_ == UNS)
    {
        return no_;
    }
    return nox_ * noy_ * noz_;
}

void
Geometry::getSize(int *nx, int *ny, int *nz) const
{
    *nx = nox_;
    *ny = noy_;
    *nz = noz_;
}

void
Geometry::dumpGeometry(float *xc, float *yc, float *zc, bool output) const
{
    int no_points = getSize();

    if (type_ == UNS || type_ == STR)
    {
        float *locX = output ? x_o_c_ : x_c_;
        float *locY = output ? y_o_c_ : y_c_;
        float *locZ = output ? z_o_c_ : z_c_;

        memcpy(xc, locX, no_points * sizeof(float));
        memcpy(yc, locY, no_points * sizeof(float));
        memcpy(zc, locZ, no_points * sizeof(float));
    }
    else if (type_ == RCT)
    {
        float *locX = output ? x_o_c_ : x_c_;
        float *locY = output ? y_o_c_ : y_c_;
        float *locZ = output ? z_o_c_ : z_c_;

        int i, j, k, count = 0;
        for (i = 0; i < nox_; ++i)
        {
            for (j = 0; j < noy_; ++j)
            {
                for (k = 0; k < noz_; ++k)
                {
                    xc[count] = locX[i];
                    yc[count] = locY[j];
                    zc[count] = locZ[k];
                    ++count;
                }
            }
        }
    }
    else if (type_ == UNI)
    {
        int i, j, k, count = 0;
        float minX, minY, minZ;
        float deltaX, deltaY, deltaZ;
        if (output)
        {
            minX = minx_o_;
            minY = miny_o_;
            minZ = minz_o_;
            deltaX = (maxx_o_ - minx_o_) / nox_;
            deltaY = (maxy_o_ - miny_o_) / noy_;
            deltaZ = (maxz_o_ - minz_o_) / noz_;
        }
        else
        {
            minX = minx_;
            minY = miny_;
            minZ = minz_;
            deltaX = (maxx_ - minx_) / nox_;
            deltaY = (maxy_ - miny_) / noy_;
            deltaZ = (maxz_ - minz_) / noz_;
        }
        for (i = 0; i < nox_; ++i)
        {
            float baseX = minX + deltaX * i;
            for (j = 0; j < noy_; ++j)
            {
                float baseY = minY + deltaY * j;
                for (k = 0; k < noz_; ++k)
                {
                    xc[count] = baseX;
                    yc[count] = baseY;
                    zc[count] = minZ + deltaZ * k;
                    ++count;
                }
            }
        }
    }
}
