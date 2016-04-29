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
#include "ReadIhsGrid.h"
int main(int argc, char *argv[])
{

   Application *application = new Application(argc,argv);

   application->run();

   return 0;

}


//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
   Application *thisApp = (Application *)userData;
   thisApp->quit(callbackData);
}


void Application::computeCallback(void *userData, void *callbackData)
{
   Application *thisApp = (Application *)userData;
   thisApp->compute(callbackData);
}


//
//
//..........................................................................
//
void Application::quit(void *)
{
   //
   // ...... delete your data here .....
   //

}


void Application::compute(void *)
{
   //
   // ...... do work here ........
   //

   // read input parameters and data object name
   FILE *grid_fp;
   int i,max_index;

   char buf[300];
   int *point_id, *id_ptr;
   int *inverse_index;

   Covise::get_browser_param("grid_path", &grid_Path);

   Mesh = Covise::get_object_name("mesh");

   if ( (grid_fp = Covise::fopen(grid_Path, "r")) == 0 )
   {
      Covise::sendError("ERROR: Can't open file >> %s",grid_Path);
      return;
   }

   //----------------------------------------------------------------------
   // ....... read a FENFLOSS geometry file ........
   //
   // This file format consists of four main parts:
   //     - header (10 lines)
   //     - geometrie information (number of points, elements, connections)
   //     - point coordinates
   //     - element information (all hexahedra)
   //----------------------------------------------------------------------

   //------------------------------------
   // Skip header consisting of ten lines
   //------------------------------------
   for ( i=0; i<10; i++)
   {
      if (fgets( buf, 300, grid_fp )!=NULL)
      {
         fprintf(stderr,"fgets_1 failed in ReadIHS2.cpp");
      }
   }
   //--------------------------------------
   // now read general geometry information
   //--------------------------------------
   if (fscanf( grid_fp, "%d%d%d%d%d\n", &n_coord/*number of points*/, &n_elem/* number of elements*/, &max_index, &max_index, &max_index)!=5)
   {
      fprintf(stderr,"fscanf_1 failed in ReadLat.cpp");
   }

   cerr << "n_coord = " << n_coord << endl;
   cerr << ", n_elem = " << n_elem << endl;
   cerr << ", max_index = " << max_index << endl;

   int result = 0;                                // used to keep track of fscanf return values

   if( Mesh != 0 )
   {

      point_id = new int[n_coord];
      id_ptr = point_id;                          // we use id_ptr for fast indexing

      //----------------------------------------
      // create unstructured grid data structure
      //----------------------------------------
      mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_elem*8, n_coord, 1);
      if ( !mesh->objectOk() )
      {
         Covise::sendError("ERROR: creation of data object 'mesh' failed");
         return;
      }

      //----------------------------------------
      // el : element list
      // vl : connection list
      // tl : type list;
      // x_coord : list containing x coordinates
      // y_coord : list containing y coordinates
      // z_coord : list containing z coordinates
      //----------------------------------------
      mesh->getAddresses( &el, &vl, &x_coord, &y_coord, &z_coord );
      mesh->getTypeList( &tl );

      //-------------------------------------------------
      // Get point coordinates. The line format is:
      //    number(int)    x(float)   y(float)   z(float)
      //-------------------------------------------------
      for( i=0; i<n_coord; i++ )
      {
         result = fscanf( grid_fp, "%d%f%f%f\n",
               id_ptr, x_coord, y_coord, z_coord );
         if( result == EOF )
         {
            Covise::sendError("ERROR: unexpected end of file");
            return;
         }
         x_coord++;
         y_coord++;
         z_coord++;
         id_ptr++;
      }

      max_index = 0;
      id_ptr = point_id;                          // reset id_ptr

      //---------------------------------------------------------
      // search for the highest index value of the last 50 points
      //---------------------------------------------------------
      for( i=n_coord-50; i<n_coord; i++ )
      {
         if( point_id[i] > max_index )
            max_index = point_id[i];
      }

      //------------------------------------------------
      // inverse_index seems to be an inverse index list
      // If so it could be very memory consuming.
      //------------------------------------------------
      inverse_index = new int[max_index+1];
      for( i=0; i<n_coord; i++)
      {
         inverse_index[*id_ptr] = i;
         id_ptr++;
      }

      //---------------------------------------------------------
      // Now read all elements. They are all hexagons in Fenfloss
      //---------------------------------------------------------
      for( i=0; i<n_elem; i++ )
      {
         //--------------
         // read one line
         //--------------
         if (fgets( buf, 300, grid_fp )!=NULL)
         {
            fprintf(stderr,"fgets_2 failed in ReadIHS2.cpp");
         }
         if( feof( grid_fp ) )
         {
            Covise::sendError("ERROR: unexpected end of file");
            return;
         }

         //----------------------
         // analyse line contents
         //----------------------
         sscanf( buf, "%d%d%d%d%d%d%d%d\n", vl,   vl+1, vl+2, vl+3,
               vl+4, vl+5, vl+6, vl+7 );

         //----------------------------------------
         // translate point_ids of the grid file to
         // local ids ranging from 0 to n_coords
         //----------------------------------------

         //   *vl is elem[i][0]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][1]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][2]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][3]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][4]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][5]

         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][6]
         *vl = inverse_index[*vl];
         vl++;

         //   *vl is elem[i][7]
         *vl = inverse_index[*vl];
         vl++;

         //---------------------------------------------------------
         // each element consists of eight points thus the beginning
         // of the (i+1)th element is to be found at i*8
         //---------------------------------------------------------
         *el++ = i*8;
         *tl++ = TYPE_HEXAGON;
      }

      delete[] inverse_index;
      delete[] point_id;
   }
   else
   {
      Covise::sendError("ERROR: object name not correct for 'mesh'");
      return;
   }

   fclose(grid_fp);

   delete mesh;
}
