/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POLYGON_SET_H
#define _POLYGON_SET_H
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

#include <api/coModule.h>
using namespace covise;
//#include <alg/coCuttingSurface.h>

typedef class Triangles_
{
private:
    float *x_, *y_, *z_;
    int *pl_, *cl_;

    int num_pl_, num_cl_, num_points_;

    float *data_;

    enum
    {
        NUMT = 1600000
    };

public:
    Triangles_();
    ~Triangles_();

    void addTriangle(float *x, float *y, float *z, float val1 = 0., float val2 = 0., float val3 = 0.);
    void addPolygon(coDoPolygons *poly);

    coDoPolygons *getDOPolygons(const char *objname);
    coDoFloat *getDOData(const char *objname);
} Triangles;

class IsoMagma : public coModule
{

private:
    // compute callback
    virtual int compute();

    float *xCoords, *yCoords, *zCoords;
    float *outdat;
    int *pl, *cl, *tl;
    int num_polygons, num_nodes, num_components;

    /// rgrid
    int xsize, ysize, zsize, xmin, ymin, zmin;
    float *rx, *ry, *rz;
    coDoRectilinearGrid *rgrid;

    // in polygon
    // components -> vertices
    int num_in_polygons, num_in_nodes, num_in_components;
    float *surf_x, *surf_y, *surf_z;
    int *surf_pl, *surf_cl;

    //
    coDoFloat *vof_in;
    float *vof_data; // internal usage
    float *vof_out; // rVOF output
    float *temp_data; // rTEMP output

    //
    Triangles *out;
    int level;

    /// calculate bounding box of numPoints points (x,y,z)
    void boundingBox(float *min, float *max, int numPoints, float *x, float *y, float *z);
    ///
    float norm(float val[3]);

    ///
    float getDistance(float nx, float ny, float nz, float x, float y, float z);

    ///
    void genNormals(float &nx, float &ny, float &nz, int p0, int p1, int p2, float *x, float *y, float *z);

    /******
      ///
           void rgridPart( float min[3], float max[3], int xdim, int ydim, int zdim, float *x, float *y, float *z,
             int &xd, int &yd, int &zd, int &tmp_x0, int &tmp_y0, int &tmp_z0 );

      ///
      void rgridCell( float p[3], int xdim, int ydim, int zdim, float *x, float *y, float *z,
                 int &xd, int &yd,  int &zd );

           ///
      bool rgridData( float p[3],coDoFloat *data, int xdim, int ydim, int zdim,
      float *x, float *y, float *z, float &val );

      bool rgridData( float p[3], float &val );

      */
    void check_add_tri(float *x, float *y, float *z);
    float expolate(coDoFloat *data, int i, int j, int k, float not_valid);
    void cancelCells(int cells[3][3]);
    /*
      void cutMinMax( int &i, int border);
      int rGridDim( int dim, float min, float max, float *f, int &smin, int &sdim);

      ///
      int BSearch( int begin, int end, float val, float *f, bool &on_border);

           enum{ TOO_LOW=-2, TOO_HIGH=-1 };
      */
    void addBox(float ox, float oy, float oz, float length, float width, float height);

    ///
    bool used[200][400][400];

    // ports
    coOutputPort *p_GridOut, *outTemp, *outVOF, *outSurf, *outSurfVOF;
    coOutputPort *p_NormalsOut;
    coOutputPort *p_DataOut;

    coInputPort *inRect, *inMat, *inTemp, *inVOF, *inSurf;

public:
    IsoMagma();
};
#endif
