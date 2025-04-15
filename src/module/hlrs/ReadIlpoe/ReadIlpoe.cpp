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
#include "ReadIlpoe.h"
#include <do/coDoPolygons.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadIlpoe::ReadIlpoe(int argc, char** argv) : coModule(argc, argv)
{

   // Parameters 
   filenameParam = addFileBrowserParam("file_path","Data file path");
   
// Ports
   p_polyOut = addOutputPort("polygons","Polygons","polygons of txt mesh");

}


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadIlpoe::compute(const char* port) {

 sendInfo("%s", port);

  FILE * file;
  const char * filename = filenameParam->getValue();

  coDoPolygons *polygonObj;

  file = fopen(filename, "r");

  if(file == NULL) {
      sendError("ERROR: Can't open file %s", filename);
      return STOP_PIPELINE;
  }  
  
  int number_of_vertices = 0, number_of_triangles = 0;

  fscanf(file, "%i %i", &number_of_vertices, &number_of_triangles);

  const char * polygonObjName = p_polyOut->getObjName();

  if(polygonObjName) {

    polygonObj = new coDoPolygons(polygonObjName, number_of_vertices, 3 * number_of_triangles, number_of_triangles);
    p_polyOut->setCurrentObject(polygonObj);

    float * vertices_x;
    float * vertices_y;
    float * vertices_z;
    int * corners;
    int * triangles;

    polygonObj->getAddresses(&vertices_x, &vertices_y, &vertices_z, &corners, &triangles);

    int dummy = 0;
    for(int i = 0; i < number_of_vertices; i++) 
      fscanf(file, "%i %f %f %f", &dummy, &vertices_x[i], &vertices_y[i], &vertices_z[i]);

    for(int i = 0; i < number_of_triangles; i++)
      fscanf(file, "%i %i %i %i", &dummy, &corners[3*i], &corners[3*i+1], &corners[3*i+2]);

    for(int i = 0; i < number_of_triangles; i++)
      triangles[i] = 3 * i;
    
    fclose(file);

  } else {
    fclose(file);
    fprintf(stderr,"Covise::get_object_name failed\n");
    return FAIL;
  }


  sendInfo("%s with %i vertices and %i triangles loaded successfully.", filename, number_of_vertices, number_of_triangles);

  return SUCCESS;
}


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MODULE_MAIN(IO, ReadIlpoe);
