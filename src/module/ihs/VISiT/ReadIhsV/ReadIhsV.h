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

//#include "ApplInterface.h"
#include "coModule.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

class Application : public coModule
{

   private:

      //  member functions
      int compute(void *callbackData);
      void quit(void *callbackData);
      void sockData(int soc);
      void postInst(void);

      // rei: tmp
      FILE *fifoin;
      char fifofilein[256];

      //  Static callback stubs
      //static void computeCallback(void *userData, void *callbackData);
      //static void quitCallback(void *userData, void *callbackData);

      //  Parameter names
      char* grid_Path;
      char* data_Path;
      char* Mesh;
      char* Veloc;
      char* Press;
      char* K_name;
      char* EPS_name;
      char* B_U_name;
      char* STR_name;

      //  Local data
      int n_coord,n_elem;
      int *el,*vl,*tl;
      float *x_coord;
      float *y_coord;
      float *z_coord;
      float *str;
      float *eps;
      float *b_u;
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
      coDoFloat* B_U;
      coDoFloat* STR;

   public:

      Application(int argc, char *argv[])

      {

         Mesh      = 0L;
         Veloc     = 0L;
         Press     = 0L;
         K_name    = 0L;
         EPS_name  = 0L;
         B_U_name  = 0L;
         STR_name  = 0L;
         Covise::set_module_description("Read data from Ihs FENFLOSS");
         Covise::add_port(OUTPUT_PORT,"mesh","coDoUnstructuredGrid|coDoPolygons","Grid");
         Covise::add_port(OUTPUT_PORT,"velocity","Vec3","velocity");
         Covise::add_port(OUTPUT_PORT,"pressure","coDoFloat","pressure");
         Covise::add_port(OUTPUT_PORT,"K","coDoFloat","K");
         Covise::add_port(OUTPUT_PORT,"EPS","coDoFloat","EPS");
         Covise::add_port(OUTPUT_PORT,"B_U","coDoFloat","B_U");
         Covise::add_port(OUTPUT_PORT,"NUt","coDoFloat","Nut");
         Covise::add_port(PARIN,"grid_path","Browser","Grid file path");
         Covise::add_port(PARIN,"data_path","Browser","Data file path");
         Covise::add_port(PARIN,"numt","Scalar","Nuber of Timesteps");
         Covise::set_port_default("grid_path","data/saug3.geo *geo*;*GEO*");
         Covise::set_port_default("data_path","data/saug3.sim.1 *sim*;*erg*;*ERG*");
         Covise::set_port_default("numt","1");
         Covise::init(argc,argv);
         //Covise::set_quit_callback(Application::quitCallback,this);
         //Covise::set_start_callback(Application::computeCallback,this);

      }

      void run() { Covise::main_loop(); }

      ~Application() {}

};
#endif                                            // _READIHS_H
