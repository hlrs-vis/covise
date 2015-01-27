/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Simple Example how to create a set of polygons              **
 **                                                                          **
 ** Name:        IsoMagma                                                  **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: D. Rainer		                                                **
 **                                                                          **
 ** History:  								                                **
 ** April-00     					       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "IsoMagma.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

void main(int argc, char *argv[])
{
    IsoMagma *application = new IsoMagma;
    application->start(argc, argv);
}

IsoMagma::IsoMagma() // this info appears in the module setup window
    : coModule("Polygon Set example Module")
{
    // output port
    // parameters:
    //   port name
    //   string to indicate connections, convention: name of the data type
    //   description
    inRect = addInputPort("rect", "coDoRectilinearGrid", "g");
    inMat = addInputPort("matgrp", "coDoFloat", "g");
    inVOF = addInputPort("vof", "coDoFloat", "d");
    inTemp = addInputPort("temp", "coDoFloat", "d");
    inSurf = addInputPort("surface", "coDoPolygons", "f");

    p_GridOut = addOutputPort("isosurface", "coDoPolygons", "isosurface");
    p_DataOut = addOutputPort("isodataout", "coDoFloat", "isodataout");
    p_NormalsOut = addOutputPort("isonormals", "coDoVec3", "isonormals");
    outVOF = addOutputPort("rVOF", "coDoFloat", "e");
    outTemp = addOutputPort("rTemp", "coDoFloat", "e");
    outSurf = addOutputPort("surface_out", "coDoPolygons", "set of polygons, each polygon is a cube");
    outSurfVOF = addOutputPort("surface_data", "coDoFloat", "set of polygons, each polygon is a cube");
    num_nodes = num_polygons = num_components = 0;
}

#include <alg/coIsoSurface.h>

int IsoMagma::compute()
{
    // get the set object name from the controller
    char objname[1024];
    int num_set = 0;

    int i, j, k, r, s, t;
    int xd, yd, zd;

    // temporary polygon
    int tmp_num;

    // get input data
    rgrid = (coDoRectilinearGrid *)(inRect->getCurrentObject());
    rgrid->getGridSize(&xsize, &ysize, &zsize);
    rgrid->getAddresses(&rx, &ry, &rz);

    // surface of object
    coDoPolygons *in_surf = (coDoPolygons *)(inSurf->getCurrentObject());
    in_surf->getAddresses(&surf_x, &surf_y, &surf_z, &surf_cl, &surf_pl);

    num_in_polygons = in_surf->getNumPolygons();
    num_in_components = in_surf->getNumVertices();
    num_in_nodes = in_surf->getNumPoints();

    // percentage of filling

    // output ( not used now)
    coDoFloat *grid_vof = new coDoFloat(outVOF->getObjName(), xsize, ysize, zsize);
    grid_vof->getAddress(&vof_out);
    outVOF->setCurrentObject(grid_vof);

    // for internal usage
    vof_data = new float[xsize * ysize * zsize];

    //
    // vof_in
    //
    vof_in = (coDoFloat *)(inVOF->getCurrentObject());
    vof_in->getGridSize(&xd, &yd, &zd); // xd should equal xsize, etc, etc !?

    // temp_in
    coDoFloat *temp_in = (coDoFloat *)(inTemp->getCurrentObject());
    float *temp_v;
    temp_in->getAddress(&temp_v);
    // temp_out
    coDoFloat *temp_out = new coDoFloat(outTemp->getObjName(), xd, yd, zd);
    temp_out->getAddress(&temp_data);
    outTemp->setCurrentObject(temp_out);

    // not used
    coDoFloat *mat = (coDoFloat *)(inMat->getCurrentObject());

    // min. temp. of final result in step 4 for testing
    float min_temp = 613.;

    for (i = 0; i < xd * yd * zd; i++)
    {
        vof_data[i] = -1e+30;
    }

    //
    // change values of vof according to the neighbor values
    //
    // quit start: has to be adapted to a new algorithm
    //

    /////////////////////////////////////7
    float sum, sum_temp;
    float exp, num, num_temp, tval, val;

    for (i = 1; i < xd - 1; i++)
        for (j = 1; j < yd - 1; j++)
            for (k = 1; k < zd - 1; k++)
            {
                // vof
                vof_in->getPointValue(i, j, k, &val);
                temp_in->getPointValue(i, j, k, &tval);
                if (val < 0.)
                {
                    exp = expolate(vof_in, i, j, k, 0.);
                    if (exp == 0.)
                    {
                        // for checking <0
                        vof_data[k + j * zd + i * zd * yd] = -1.0e-8;
                    }
                    else
                    {
                        vof_data[k + j * zd + i * zd * yd] = exp;
                    }
                }
                else
                {
                    vof_data[k + j * zd + i * zd * yd] = val;
                }

                // temp
                temp_in->getPointValue(i, j, k, &tval);
                float mat_grp;
                mat->getPointValue(i, j, k, &mat_grp);
                if (tval < min_temp)
                {
                    exp = expolate(temp_in, i, j, k, min_temp);
                    if (exp != min_temp)
                    {
                        temp_data[k + j * zd + i * zd * yd] = exp;
                    }
                    else
                    {
                        temp_data[k + j * zd + i * zd * yd] = 700.;
                    }
                }
                else
                {
                    temp_data[k + j * zd + i * zd * yd] = tval;
                }
            }

    for (i = 0; i < xd * yd * zd; i++)
    {
        if (vof_data[i] < 0.)
            vof_out[i] = -1.0e+30;
        else
            vof_out[i] = vof_data[i];
    }

    // now that we have in vof_out the "extrapolated" data,
    // we may proceed to extracting the isosurface using
    // the new feature of RECT_IsoPlane (see RECT_IsoPlane::createIsoPlane).
    // we need an array of valid cells. A cell is valid
    // if all its nodes are >= 0.0
    {
        int numelem = (xd - 1) * (yd - 1) * (zd - 1);
        bool *cellValid = new bool[numelem];
        int i, j, k, cell = 0;
        for (i = 0; i < xd - 1; ++i)
        {
            for (j = 0; j < yd - 1; ++j)
            {
                for (k = 0; k < zd - 1; ++k, ++cell)
                {
                    int base0 = i * yd * zd + j * zd + k; // @@@ efficiency!!!
                    int base1 = i * yd * zd + j * zd + k + 1; // @@@ efficiency!!!
                    int base2 = i * yd * zd + (j + 1) * zd + k; // @@@ efficiency!!!
                    int base3 = i * yd * zd + (j + 1) * zd + k + 1; // @@@ efficiency!!!
                    int base4 = base0 + yd * zd;
                    int base5 = base1 + yd * zd;
                    int base6 = base2 + yd * zd;
                    int base7 = base3 + yd * zd;
                    if (vof_out[base0] >= 0.0 && vof_out[base1] >= 0.0 && vof_out[base2] >= 0.0 && vof_out[base3] >= 0.0 && vof_out[base4] >= 0.0 && vof_out[base5] >= 0.0 && vof_out[base6] >= 0.0 && vof_out[base7] >= 0.0)
                    {
                        cellValid[cell] = true;
                    }
                    else
                    {
                        cellValid[cell] = false;
                    }
                }
            }
        }

        RECT_IsoPlane *rplane = new RECT_IsoPlane(numelem, xd * yd * zd, 1 /* scalar */,
                                                  xd, yd, zd, rx, ry, rz,
                                                  vof_out, temp_data, NULL, NULL, NULL, 0.85, // @@@
                                                  1 /* interpolate temperature data */);
        rplane->createIsoPlane(cellValid);
        rplane->createcoDistributedObjects(p_GridOut, p_NormalsOut, p_DataOut, 1, 0, "white");
        delete rplane;
        delete[] cellValid;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    float tmp_x[3], tmp_y[3], tmp_z[3];

    //
    // check every triangle if it is part of the "isosurface" on the surface
    //
    out = new Triangles();

    for (r = 0; r < num_in_polygons; r++)
    {
        if (1)
            cerr << ". ";

        tmp_num = (r != num_in_polygons - 1) ? surf_pl[r + 1] - surf_pl[r] : num_in_components - surf_pl[r];
        if (tmp_num == 3)
        {
            for (s = 0; s < tmp_num; s++)
            {
                tmp_x[s] = surf_x[surf_cl[surf_pl[r] + s]];
                tmp_y[s] = surf_y[surf_cl[surf_pl[r] + s]];
                tmp_z[s] = surf_z[surf_cl[surf_pl[r] + s]];
            }
            level = 0;
            check_add_tri(tmp_x, tmp_y, tmp_z);
        }
    }

    // generate grid for isosurface of the rest
    coDoPolygons *sidePoly = out->getDOPolygons(outSurf->getObjName());

    outSurf->setCurrentObject(sidePoly);
    outSurfVOF->setCurrentObject(out->getDOData(outSurfVOF->getObjName()));
    delete out;
    delete[] vof_data;
    return CONTINUE_PIPELINE;
}

//
// adapt values to the average of the "valid" neighbors
//

float IsoMagma::expolate(coDoFloat *data, int i, int j, int k, float not_valid)
{
    int r, s, t;
    float val;
    float num = 0., sum = 0.;
    int rad = 1;

    for (r = -rad; r < rad + 1; r++)
        for (s = -rad; s < rad + 1; s++)
            for (t = -rad; t < rad + 1; t++)
            {
                if (i + r >= 0 && j + s >= 0 && k + t >= 0 && i + r < xsize && j + s < ysize && k + t < zsize)
                {
                    data->getPointValue(i + r, j + s, k + t, &val);
                    if (val > not_valid)
                    {
                        sum += val;
                        num += 1.;
                    }
                }
            }

    if (num > 0.)
    {
        return (sum / num);
    }
    else
    {
        return not_valid;
    }
}

///
void IsoMagma::check_add_tri(float *x, float *y, float *z)
{
    int i;
    float vals[3];
    float temp[3];

    float val = -1.;
    float p[3];
    //float m[3];

    float dat[3];
    int cell[3][3];

    float *arr[3];
    arr[0] = vof_data;

    float *temp_arr[3], temp_vals[3];
    temp_arr[0] = temp_data;

    for (i = 0; i < 3; i++)
    {
        dat[0] = -2;

        p[0] = x[i];
        p[1] = y[i];
        p[2] = z[i];

        rgrid->interpolateField(dat, p, cell[i], 1, 1, arr);
        vals[i] = dat[0];

        rgrid->interpolateField(dat, p, cell[i], 1, 1, temp_arr);

        temp_vals[i] = dat[0];
    }

    float nu = 0.85; //-1.0e-9;

    // change condition in level 14
    if (level > 14)
    {
        if (vals[0] >= 0.5 * nu && vals[1] >= 0.5 * nu && vals[2] >= 0.5 * nu)
        {
            out->addTriangle(x, y, z, temp_vals[0], temp_vals[1], temp_vals[2]);
            cancelCells(cell);
        }
        return;
    }

    // own condition in step 1 to reach a continous result
    if (level < 0)
    {
        float f = 0.3;
        if ((vals[0] >= f && vals[1] >= f)
            || (vals[0] >= f && vals[2] >= f)
            || (vals[1] >= f && vals[2] >= f))
        {
            out->addTriangle(x, y, z, temp_vals[0], temp_vals[1], temp_vals[2]);
            cancelCells(cell);
            return;
        }
    }
    else // general condition
        if (vals[0] >= nu && vals[1] >= nu && vals[2] >= nu)
    {
        out->addTriangle(x, y, z, temp_vals[0], temp_vals[1], temp_vals[2]);
        cancelCells(cell);
        return;
    }

    float nu2 = -1.0e-2;
    if (vals[0] < nu && vals[1] < nu && vals[2] < nu)
        return;

    //
    // if no result was taken, split triangle into two halfs and repeat
    //
    int ch = 0;
    float max = -1.;

    for (i = 0; i < 3; i++)
    {
        if (vals[i] != vals[(i + 1) % 3])
        {
            if (fabs(vals[i] - vals[(i + 1) % 3]) > max)
            {
                ch = i;
                max = fabs(vals[i] - vals[(i + 1) % 3]);
            }
        }
    }

    float *newx = new float[3];
    float *newy = new float[3];
    float *newz = new float[3];
    float *newx2 = new float[3];
    float *newy2 = new float[3];
    float *newz2 = new float[3];

    // check first half

    newx[0] = x[ch];
    newx[1] = 0.5 * (x[ch] + x[(ch + 1) % 3]);
    newx[2] = x[(ch + 2) % 3];

    newy[0] = y[ch];
    newy[1] = 0.5 * (y[ch] + y[(ch + 1) % 3]);
    newy[2] = y[(ch + 2) % 3];

    newz[0] = z[ch];
    newz[1] = 0.5 * (z[ch] + z[(ch + 1) % 3]);
    newz[2] = z[(ch + 2) % 3];

    level++;

    check_add_tri(newx, newy, newz);
    level--;

    // check second half
    newx2[0] = x[(ch + 2) % 3];
    newx2[1] = 0.5 * (x[ch] + x[(ch + 1) % 3]);
    newx2[2] = x[(ch + 1) % 3];

    newy2[0] = y[(ch + 2) % 3];
    newy2[1] = 0.5 * (y[ch] + y[(ch + 1) % 3]);
    newy2[2] = y[(ch + 1) % 3];

    newz2[0] = z[(ch + 2) % 3];
    newz2[1] = 0.5 * (z[ch] + z[(ch + 1) % 3]);
    newz2[2] = z[(ch + 1) % 3];

    level++;
    check_add_tri(newx2, newy2, newz2);
    level--;

    delete[] newx;
    delete[] newy;
    delete[] newz;
    delete[] newx2;
    delete[] newy2;
    delete[] newz2;
}

//
// first try to remember if a cell is useless
//
void IsoMagma::cancelCells(int cells[3][3])
{
    /*
      int i,j,k,p, r, s, t;

      // for every point of the triangle
      for( p=0; p<3; p++ )
      {
         i = cells[p][0];
         j = cells[p][1];
         k = cells[p][2];

         for( r=0; r<2; r++ )
   for( s=0; s<2; s++ )
   for( t=0; t<2; t++ )

   vof_out[ (k+t) + (j+s)*zsize + (i+r)*zsize*ysize ] =  -1.0e-30;
   }
   */
}

/// calculate bounding box of numPoints points (x,y,z)
void IsoMagma::boundingBox(float *min,
                           float *max,
                           int numPoints,
                           float *x,
                           float *y,
                           float *z)
{
    /*   min[0]=min[1]=min[2]=FLTMAX;
      max[0]=max[1]=max[2]=-FLTMAX;*/

    int i;
    for (i = 0; i < numPoints; i++)
    {
        if (x[i] > max[0])
            max[0] = x[i];
        if (y[i] > max[1])
            max[1] = y[i];
        if (z[i] > max[2])
            max[2] = z[i];

        if (x[i] < min[0])
            min[0] = x[i];
        if (y[i] < min[1])
            min[1] = y[i];
        if (z[i] < min[2])
            min[2] = z[i];
    }
}

void IsoMagma::genNormals(float &nx,
                          float &ny,
                          float &nz,
                          int p0,
                          int p1,
                          int p2,
                          float *x,
                          float *y,
                          float *z)
{
    float n[3], x1, y1, z1, x2, y2, z2;

    x1 = x[p1] - x[p0];
    y1 = y[p1] - y[p0];
    z1 = z[p1] - z[p0];
    x2 = x[p2] - x[p0];
    y2 = y[p2] - y[p0];
    z2 = z[p2] - z[p0];

    n[0] = y1 * z2 - y2 * z1;
    n[1] = x2 * z1 - x1 * z2;
    n[2] = x1 * y2 - x2 * y1;

    float nnorm = norm(n);
    if (nnorm > 1.0e-20)
    {
        nx = n[0] / nnorm;
        ny = n[1] / nnorm;
        nz = n[2] / nnorm;
    }
    else
    {
        nx = ny = nz = 0.;
    }
}

inline float IsoMagma::norm(float val[3])
{
    return sqrt(val[0] * val[0]
                + val[1] * val[1]
                + val[2] * val[2]);
}

inline float IsoMagma::getDistance(float nx,
                                   float ny,
                                   float nz,
                                   float x,
                                   float y,
                                   float z)
{
    return (nx * x + ny * y + nz * z);
}

//
// add hexagon to the polygon list
//

void IsoMagma::addBox(float ox,
                      float oy,
                      float oz,
                      float length,
                      float width,
                      float height)
{
    int vertexList[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };
    int i;

    // update polygon list
    //for( i=0; i<6; i++ ) {
    pl[num_polygons++] = num_components; //polygonList[i] + num_components;
    tl[num_polygons - 1] = TYPE_HEXAEDER;
    //}

    // update corner list
    for (i = 0; i < 8; i++)
    {
        cl[num_components++] = vertexList[i] + num_nodes;
    }

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   . height(z)
    //  .       .   .
    //  .   1   .   2
    //  .       . . width(y)
    //  0.......3
    //     length(x)

    xCoords[num_nodes + 4] = ox;
    yCoords[num_nodes + 4] = oy;
    zCoords[num_nodes + 4] = oz;

    xCoords[num_nodes + 5] = ox;
    yCoords[num_nodes + 5] = oy + width;
    zCoords[num_nodes + 5] = oz;

    xCoords[num_nodes + 6] = ox + length;
    yCoords[num_nodes + 6] = oy + width;
    zCoords[num_nodes + 6] = oz;

    xCoords[num_nodes + 7] = ox + length;
    yCoords[num_nodes + 7] = oy;
    zCoords[num_nodes + 7] = oz;

    xCoords[num_nodes + 0] = ox;
    yCoords[num_nodes + 0] = oy;
    zCoords[num_nodes + 0] = oz + height;

    xCoords[num_nodes + 1] = ox;
    yCoords[num_nodes + 1] = oy + width;
    zCoords[num_nodes + 1] = oz + height;

    xCoords[num_nodes + 2] = ox + length;
    yCoords[num_nodes + 2] = oy + width;
    zCoords[num_nodes + 2] = oz + height;

    xCoords[num_nodes + 3] = ox + length;
    yCoords[num_nodes + 3] = oy;
    zCoords[num_nodes + 3] = oz + height;

    num_nodes += 8;
}

//
// simple class to collect triangles and their data
//

Triangles_::Triangles_()
{
    pl_ = new int[NUMT];
    cl_ = new int[NUMT * 3];

    x_ = new float[NUMT * 3];
    y_ = new float[NUMT * 3];
    z_ = new float[NUMT * 3];

    data_ = new float[NUMT * 3];

    num_pl_ = 0;
    num_cl_ = 0;
    num_points_ = 0;
}

Triangles_::~Triangles_()
{
    delete[] pl_;
    delete[] cl_;
    delete[] x_;
    delete[] y_;
    delete[] z_;
    delete[] data_;
}

void Triangles_::addTriangle(float *x, float *y, float *z, float val1, float val2, float val3)
{
    if (num_pl_ > NUMT - 5)
    {
        return;
    }

    data_[num_points_] = val1;
    data_[num_points_ + 1] = val2;
    data_[num_points_ + 2] = val3;

    pl_[num_pl_++] = num_cl_;

    cl_[num_cl_++] = num_points_;
    cl_[num_cl_++] = num_points_ + 1;
    cl_[num_cl_++] = num_points_ + 2;

    int i;
    for (i = 0; i < 3; i++)
    {
        x_[num_points_] = x[i];
        y_[num_points_] = y[i];
        z_[num_points_] = z[i];

        num_points_++;
    }
}

coDoPolygons *Triangles_::getDOPolygons(const char *objname)
{
    coDoPolygons *po;
    po = new coDoPolygons(objname, num_points_, x_, y_, z_,
                          num_cl_, cl_, num_pl_, pl_);
    po->addAttribute("vertexOrder", "2");
    return po;
}

coDoFloat *Triangles_::getDOData(const char *objname)
{
    return new coDoFloat(objname, num_points_, data_);
}
