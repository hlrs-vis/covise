/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "BezierGrid.h"
#include <do/coDoLines.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>

using namespace covise;

BezierGrid::BezierGrid(int argc, char *argv[])
    : coModule(argc, argv, "BezierGrid module")
{
    pGridIn = addInputPort("GridIn0", "StructuredGrid", "grid input");
    pGridOut = addOutputPort("GridOut0", "StructuredGrid", "grid output");
}

int BezierGrid::compute(const char *port)
{
    (void)port;

    const coDistributedObject *in_obj = pGridIn->getCurrentObject();
    
    const coDoStructuredGrid *grid;
    grid = (const coDoStructuredGrid*)in_obj;
    float *x, *y, *z;
    int xdim,ydim,zdim;
    grid->getGridSize(&xdim,&ydim,&zdim);
    grid->getAddresses(&x,&y,&z);

    float u, v;
    int gridSize=xdim*ydim*zdim;
    vector <float> controlPoints;
    vector <float> x_start, y_start, z_start; 

    std::vector<std::vector<float> > preCasteljauPoints;
    std::vector<std::vector<float> > CasteljauU;    

    vector<float> pnt;
    int step_u = 0;
    int step_v = 0;
     for (v = 0; v <= 1.00f; v += 0.1f)
     {
          step_v += 1;
          step_u = 0;
          for (u = 0; u <= 1.00f; u += 0.1f)
          {
              int counter = 0;
              for (size_t i = 0; i < gridSize; i++)
              {
              controlPoints.push_back(x[i]);
              controlPoints.push_back(y[i]);
              controlPoints.push_back(z[i]);
              preCasteljauPoints.push_back(controlPoints);
              controlPoints.clear();
              counter++;

                   if (counter == xdim)
                   {
                   counter = 0;
                   CasteljauU.push_back(casteljauAproximation(preCasteljauPoints, v));                
                   preCasteljauPoints.clear();
                   }         
              }

              pnt = casteljauAproximation(CasteljauU, u);   
              x_start.push_back(pnt[0]);
              y_start.push_back(pnt[1]);
              z_start.push_back(pnt[2]);
              pnt.clear();
              step_u += 1;         
              CasteljauU.clear();
          }
      }

      float *x_coord, *y_coord, *z_coord;
      x_coord = &x_start[0];
      y_coord = &y_start[0];
      z_coord = &z_start[0];

      int xn_dim = step_u;
      int yn_dim = step_v;
      fprintf(stderr, "xn %d yn %d\n",xn_dim,yn_dim);

      coDoStructuredGrid *Grid1 = new coDoStructuredGrid(pGridOut->getObjName(), xn_dim, yn_dim, 1, x_coord, y_coord, z_coord);
      pGridOut->setCurrentObject(Grid1);

return SUCCESS;
}


vector<float> BezierGrid::casteljauAproximation(std::vector<vector<float> > points, float t)
{
    size_t steps = points.size() - 1;
    vector<float>  output;
    output.reserve(3);

      for (size_t j = 0; size_t(j) < steps; j++)
      {
          for (size_t k = 0; k < steps-j; k++)
          {
          vector <float>tmp;
          tmp.reserve(3);
              for (int l = 0; l < 3; l++)
              { 
              points[k][l] *= (1 - t);
              tmp.push_back(points[k + 1][l]);
              tmp[l] *= t;
              points[k][l] += tmp[l];  
              }
          }
      }
      output.clear();
      for (int o = 0; o < 3; o++)
      {
      output.push_back(points[0][o]);
      }
    
return output;
}

// instantiate an object of class BezierGrid and register with COVISE
MODULE_MAIN(Filter, BezierGrid)
