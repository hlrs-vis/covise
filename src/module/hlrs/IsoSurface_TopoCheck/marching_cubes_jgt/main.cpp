/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**
 * @file    main.cpp
 * @author  Thomas Lewiner <tomlew@mat.puc-rio.br>
 * @author  Math Dept, PUC-Rio
 * @version 2008.1
 * @date    07/03/2008
 *
 * @brief   MarchingCubes Command Line interface
 */
//________________________________________________

#include <stdio.h>
#include <string.h>
#include "MarchingCubes.h"

// void compute_data( MarchingCubes &mc ) ;

//_____________________________________________________________________________
// main function
int MarchingCubes::lewiner(float isovalue,
                           int nx, // grid sizes
                           int ny,
                           int nz,
                           float xmin,
                           float ymin,
                           float zmin,
                           float dx,
                           float dy,
                           float dz,
                           float *data, // input data
                           int &_nverts, // number of vertices
                           int &_ntrigs, // number of triangles
                           float *&xcoord,
                           float *&ycoord,
                           float *&zcoord,
                           float *&normx, // normal vector
                           float *&normy,
                           float *&normz,
                           int *&vl, // vertex list
                           int *&pl) // polygon list
//-----------------------------------------------------------------------------
{

    MarchingCubes mc;
    mc.set_resolution(nx, ny, nz);

    mc.init_all();

    int sx = mc.size_x();
    int sy = mc.size_y();
    int sz = mc.size_z();

    // copy the data
    // memcpy(mc._data, data, nx*ny*nz*sizeof(float));
    // sorting is the other way round, so memcpy cannot be used ...
    // we have to resort
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                mc._data[i + j * sx + k * sx * sy] = data[k + j * sz + i * sz * sy];
            }
        }
    }

    // compute the isosurface
    mc.run(isovalue);

    fprintf(stderr, "isosurface consists of %d vertices and %d triangles\n", mc.nverts(), mc.ntrigs());

    _nverts = mc.nverts();
    _ntrigs = mc.ntrigs();

    // output
    xcoord = new float[mc.nverts()];
    ycoord = new float[mc.nverts()];
    zcoord = new float[mc.nverts()];
    normx = new float[mc.nverts()];
    normy = new float[mc.nverts()];
    normz = new float[mc.nverts()];

    vl = new int[mc.ntrigs() * 3];
    pl = new int[mc.ntrigs()];

    for (int i = 0; i < mc.nverts(); i++)
    {
        xcoord[i] = mc._vertices[i].x * dx + xmin;
        ycoord[i] = mc._vertices[i].y * dy + ymin;
        zcoord[i] = mc._vertices[i].z * dz + zmin;
        normx[i] = mc._vertices[i].nx;
        normy[i] = mc._vertices[i].ny;
        normz[i] = mc._vertices[i].nz;
    }
    for (int i = 0; i < mc.ntrigs(); i++)
    {
        vl[3 * i + 0] = mc._triangles[i].v1;
        vl[3 * i + 1] = mc._triangles[i].v2;
        vl[3 * i + 2] = mc._triangles[i].v3;
        if ((mc._triangles[i].v1 == -1) || (mc._triangles[i].v1 == -1) || (mc._triangles[i].v1 == -1))
        {
            fprintf(stderr, "-1 in vl at i=%d!\n", i);
        }
        pl[i] = 3 * i;
    }

    //mc.writeIV("test.iv") ;

    mc.clean_all();

    return 0;
}

//_____________________________________________________________________________

/*

//_____________________________________________________________________________
// Compute data
void compute_data( MarchingCubes &mc )
//-----------------------------------------------------------------------------
{
  float x,y,z      ;
  float sx,sy,sz   ;
  float tx,ty,tz   ;

  float r,R ;
  r = 1.85f ;
  R = 4 ;

  sx     = (float) mc.size_x() / 16 ;
  sy     = (float) mc.size_y() / 16 ;
  sz     = (float) mc.size_z() / 16 ;
  tx     = (float) mc.size_x() / (2*sx) ;
  ty     = (float) mc.size_y() / (2*sy) + 1.5f ;
  tz     = (float) mc.size_z() / (2*sz) ;

  for( int k = 0 ; k < mc.size_z() ; k++ )
  {
    z = ( (float) k ) / sz  - tz ;

    for( int j = 0 ; j < mc.size_y() ; j++ )
    {
      y = ( (float) j ) / sy  - ty ;

      for( int i = 0 ; i < mc.size_x() ; i++ )
      {
        x = ( (float) i ) / sx - tx ;
        mc.set_data( x+y+z -3, i,j,k ) ) ;
      }
    }
  }
}
//_____________________________________________________________________________


*/
