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

//#define E1
//#define E2
//#define E3
//#define E4
#define E5

// this includes our own class's headers
#include "Hello.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Hello::Hello(int argc, char *argv[])
: coModule(argc, argv, "Hello, world! program")
{
#ifdef E1
   stringParam = addStringParam("Output Text", "Text that should be sent to COVISE");
#endif
#ifdef E2
   boolParam = addBooleanParam("Show String", "Show the text parameter in the Control Panel");
#endif

#if defined(E3) || defined(E5)
   geoOutPort = addOutputPort("geometries", "Lines|Polygons", "Output geometries");
#endif
#ifdef E4
   dataOutPort = addOutputPort("data", "Unstructured_S3D_Data", "Output data");
#endif

}


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Hello::compute(const char *)
{
#ifdef E1
   sendInfo(stringParam->getValue());
#endif

#ifdef E3
   float * x;
   float * y;
   float * z;
   int * cornerList;
   int * shapeList;
#endif

#ifdef E4
   float * values;
#endif

#ifdef E5
   float x[8];
   float y[8];
   float z[8];
   int vertexList[24] = {0,1,2,3,  1,5,6,2,  5,4,7,6,  4,0,3,7,  3,2,6,7,  1,0,4,5};
   int polygonList[6] = {0, 4, 8, 12, 16, 20};
#endif

#ifdef E3
   coDoLines * lines;

   lines = new coDoLines(geoOutPort->getObjName(), 4, 5, 1);

   lines->getAddresses(&x, &y, &z, &cornerList, &shapeList);
#endif

#if defined(E3) || defined(E5)
   x[0] =  0.0f;
   y[0] =  0.0f;
   z[0] = -1.0f;

   x[1] =  1.0f;
   y[1] =  0.0f;
   z[1] = -1.0f;

   x[2] =  1.0f;
   y[2] =  1.0f;
   z[2] = -1.0f;

   x[3] =  0.0f;
   y[3] =  1.0f;
   z[3] = -1.0f;
#endif

#ifdef E5

   x[4] =  0.0f;
   y[4] =  0.0f;
   z[4] = -2.0f;

   x[5] =  1.0f;
   y[5] =  0.0f;
   z[5] = -2.0f;

   x[6] =  1.0f;
   y[6] =  1.0f;
   z[6] = -2.0f;

   x[7] =  0.0f;
   y[7] =  1.0f;
   z[7] = -2.0f;
#endif

#ifdef E3
   for (int ctr = 0; ctr < 4; ++ctr) cornerList[ctr] = ctr;
   cornerList[4] = 0;

   shapeList[0] = 0;

   geoOutPort->setCurrentObject(lines);
#endif
#ifdef E5
   coDoPolygons * polys = new coDoPolygons(geoOutPort->getObjName(),
         8, x, y, z,
         24, vertexList,
         6, polygonList);

   geoOutPort->setCurrentObject(polys);
#endif

#ifdef E4
   coDoFloat * data;

   data = new coDoFloat(dataOutPort->getObjName(), 4);
   data->getAddress(&values);

   for (int ctr = 0; ctr < 4; ++ctr) values[ctr] = ctr;

   dataOutPort->setCurrentObject(data);
#endif

   return SUCCESS;
}


#ifdef E2
void Hello::param(const char * paramName, bool inMapLoading)
{
   if(!inMapLoading)
   {
      if (!strcmp(paramName, boolParam->getName()))
      {
         if (boolParam->getValue())
            stringParam->show();
         else
            stringParam->hide();
      }
   }
}
#endif

MODULE_MAIN(Hello)
