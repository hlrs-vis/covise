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

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include <do/coDoUnstructuredGrid.h>

class Application
{

   private:

      //  member functions
      void compute(void *callbackData);
      void quit(void *callbackData);

      //  Static callback stubs
      static void computeCallback(void *userData, void *callbackData);
      static void quitCallback(void *userData, void *callbackData);

      //  Parameter names
      char* grid_Path;
      char* Mesh;

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

   public:

      Application(int argc, char *argv[])
      {

         Mesh = 0L;
         Covise::set_module_description( "Read data from Ihs FENFLOSS" );
         Covise::add_port( OUTPUT_PORT, "mesh", "UnstructuredGrid", "Grid" );
         Covise::add_port( PARIN, "grid_path", "Browser", "Grid file path" );
         Covise::set_port_default( "grid_path",
            "/mnt/demoplatte/demo/covise/src/application/CUG/ihs/* *.geo*" );

         Covise::init( argc, argv );
         Covise::set_quit_callback( Application::quitCallback, this );
         Covise::set_start_callback( Application::computeCallback, this );

      }

      void run() { Covise::main_loop(); }

      ~Application() {}

};
#endif                                            // _READIHS_H
