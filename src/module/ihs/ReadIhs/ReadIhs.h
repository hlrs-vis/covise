#ifndef _READIHS_H
#define _READIHS_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif

class ReadIhs : public coSimpleModule
{

   private:

      //  member functions
      virtual int compute(const char *port);

      //  Parameter names
      const char* grid_Path;
      const char* data_Path;
	  
#ifdef YAC
      coObjInfo Mesh;
      coObjInfo Veloc;
      coObjInfo Press;
      coObjInfo K_name;
      coObjInfo EPS_name;
      coObjInfo RHO_name;
      coObjInfo STR_name;
#else
      const char* Mesh;
      const char* Veloc;
      const char* Press;
      const char* K_name;
      const char* EPS_name;
      const char* RHO_name;
	  const char* STR_name;
#endif 

      //  Local data
      int n_coord,n_elem;
      int *el,*vl,*tl;
      float *x_coord;
      float *y_coord;
      float *z_coord;
      float *str;
      float *eps;
      float *rho;
      //  float *vles;
      float *k;
      float *u, *v, *w;
      float *p;

      //  Shared memory data
      coDoUnstructuredGrid*       mesh;
      coDoPolygons*      polygons;
      coDistributedObject *grid;
      coDoVec3* veloc;
      coDoFloat* press;
      coDoFloat* K;
      coDoFloat* EPS;
      coDoFloat* RHO;
      //   coDoFloat* VLES;
      coDoFloat* STR;

      coFileBrowserParam *grid_path;
      coFileBrowserParam *data_path;
      coIntScalarParam *numt;
	  
      // ports
      coOutputPort *port_mesh;
      coOutputPort *port_velocity;
      coOutputPort *port_pressure;
      coOutputPort *port_K;
      coOutputPort *port_EPS;
      coOutputPort *port_RHO;
      coOutputPort *port_VLES;
      coOutputPort *port_NUt;


   public:
      ReadIhs(int argc, char **argv);
};
#endif                                            // _READIHS_H
