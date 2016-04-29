/**************************************************************************\ 
 **                                                              2002      **
 **                                                                        **
 ** Description:  COVISE relabs     New application module               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  M. Becker                                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  01.07.02  V1.0                                                  **
\**************************************************************************/
#ifdef _WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "relabs.h"

relabs::relabs(int argc, char *argv[])
: coSimpleModule(argc, argv, "relabs")
{
   //ports
   p_grid = addInputPort ("grid","UnstructuredGrid|Polygons","computation grid");
   p_velo_in  = addInputPort ("velocity", "Vec3","input vector data");
   p_velo_out = addOutputPort ("v_out", "Vec3","v transformed");

   p_rotaxis = addChoiceParam("Rotation_axis", "ConnectionMethod");
   s_rotaxis[0] = strdup("x");
   s_rotaxis[1] = strdup("y");
   s_rotaxis[2] = strdup("z");
   p_rotaxis->setValue(3, s_rotaxis, RotZ);

   p_revolutions = addFloatParam("rpm", "revolutions per minute");
   p_revolutions->setValue(250.0);

   p_direction = addChoiceParam("rel2abs_or_abs2rel", "Rel2Abs or Abs2Rel");
   s_direction[0] = strdup("abs2rel");
   s_direction[1] = strdup("rel2abs");
   p_direction->setValue(2, s_direction, Rel2Abs);
}


int relabs::compute(const char *)
{

   int i;

   int n;
   int num_values_grid;
   int num_values_velo;

   int rotaxis;
   float rotspeed;

   float *x, *y, *z;
   float *u, *v, *w;

   coDoVec3 *v_out_vector = NULL;

   rotaxis=p_rotaxis->getValue();
//   cerr << "rotaxis: " << s_rotaxis[rotaxis] << endl;

   rotspeed=p_revolutions->getValue();

   // get grid
   const coDistributedObject *obj = p_grid->getCurrentObject();

   // ------- Unstructured grid
   if (obj->isType("UNSGRD"))
   {
      // unstructured vector data

      coDoUnstructuredGrid *unsgrd = (coDoUnstructuredGrid *) obj;
      int numConn,numElem;
      int *elemList,*connList;

      unsgrd->getAddresses(&elemList, &connList, &x, &y, &z);
      unsgrd->getGridSize(&numElem, &numConn, &num_values_grid);
   }
   else if (obj->isType("POLYGN"))
   {
      coDoPolygons *polygons = (coDoPolygons *) obj;
      int numConn,numElem;
      int *elemList,*connList;
      polygons->getAddresses(&x, &y, &z, &connList, &elemList);
      num_values_grid = polygons->getNumPoints();
      numConn = polygons->getNumVertices();
      numElem = polygons->getNumPolygons();
   }
   else
   {
      sendError("Received illegal type at port '%s'",p_grid->getName());
      return FAIL;
   }
   if (!obj)
   {
      sendError("Did not receive object at port '%s'",p_grid->getName());
      return FAIL;
   }

   // get velocity data
   obj = p_velo_in->getCurrentObject();
   if (!obj)
   {
      //error message nerves during running simulation (update simulation data every n iterations)
      //sendError("Did not receive object at port '%s'",p_velo_in->getName());

      return FAIL;
   }

   if (obj->isType("USTVDT"))
   {
      // unstructured vector data

      coDoVec3 *vector_data = (coDoVec3 *) obj;
      num_values_velo = vector_data->getNumPoints();
      vector_data -> getAddresses(&u, &v, &w);
   }
   else
   {
      sendError("Received illegal type at port '%s'", p_velo_in->getName());
      return FAIL;
   }
   if (!obj)
   {
      sendError("Did not receive object at port '%s'", p_velo_in->getName());
      return FAIL;
   }

   // control num_values_grid = num_values_velo
   if (num_values_grid != num_values_velo)
   {
      sendError("grid coord size and velocity array size are not the same!");
      return FAIL;
   }

   n = num_values_grid;

   // calculate output
   float r = 0.0;                                   // radius of knot
   //float v_xy;                                    // velocity of knot in xy-plane
   //float alpha;                                   // angle of coorinates
   //float beta;                                    // angle between vx and vy

   float *vout_x;
   float *vout_y;
   float *vout_z;

   // create output objects
   //scalar
   v_out_vector = new coDoVec3(p_velo_out->getObjName(), n);

   // get pointers
   //vector
   v_out_vector -> getAddresses(&vout_x, &vout_y, &vout_z);
   /* 
       if (rotaxis==0)	// x-axis
       {
          vu_vector_data -> getAddresses(&vu_z, &vu_x, &vu_y);
          vr_vector_data -> getAddresses(&vr_z, &vr_x, &vr_y);
          vm_vector_data -> getAddresses(&vm_z, &vm_x, &vm_y);
      }
       if (rotaxis==1) // y-axis
       {
          vu_vector_data -> getAddresses(&vu_y, &vu_z, &vu_x);
          vr_vector_data -> getAddresses(&vr_y, &vr_z, &vr_x);
   vm_vector_data -> getAddresses(&vm_y, &vm_z, &vm_x);
   }
   if (rotaxis==2) // z-axis
   {
   vu_vector_data -> getAddresses(&vu_x, &vu_y, &vu_z);
   vr_vector_data -> getAddresses(&vr_x, &vr_y, &vr_z);
   vm_vector_data -> getAddresses(&vm_x, &vm_y, &vm_z);
   }
   */
   // calculation

   //float factor;
   float omega = M_PI*rotspeed/30.;
   int abs2rel=p_direction->getValue();
   float rad, vu, vr;

   if (abs2rel==0)
      r=1;
   if (abs2rel==1)
      r=-1;

   if (rotaxis==0)                                // x-axis
   {
      for (i=0; i<n; i++)
      {
         // abs2rel: r== 1
         // rel2abs: r==-1
         vout_x[i]=u[i];                          //OK

         vout_y[i]=v[i]+r*omega*z[i];
         vout_z[i]=w[i]-r*omega*y[i];
      }
   }

   if (rotaxis==1)                                //y-axis
   {
      for (i=0; i<n; i++)
      {
         // abs2rel: r== 1
         // rel2abs: r==-1
         vout_x[i]=u[i]+r*omega*z[i];
         vout_y[i]=v[i];                          //OK
         vout_z[i]=w[i]-r*omega*x[i];
      }
   }

   if (rotaxis==2)                                //z-axis
   {
      for (i=0; i<n; i++)
      {
         // abs2rel: r== 1
         // rel2abs: r==-1
	 rad = sqrt(x[i]*x[i] + y[i]*y[i]);
	 vu = (x[i]*v[i]-y[i]*u[i])/rad;
	 vr = (x[i]*u[i]+y[i]*v[i])/rad;
	 
	 vu += -r*rad*omega;
	 
	 vout_x[i] = (x[i]*vr-y[i]*vu)/rad;
	 vout_y[i] = (y[i]*vr+x[i]*vu)/rad;
	 
         //vout_x[i]=u[i]+r*omega*y[i];
         //vout_y[i]=v[i]-r*omega*x[i];

         vout_z[i]=w[i];                          //OK
      }
   }

   /*
      // change axis back!
       if (rotaxis == 0)
       {
         // x ist zu z geworden
           // y ist zu x geworden
           // z ist zu y geworden
           float *vtemp;
           vtemp = vout_x;
           vout_x = vout_y;
           vout_y = vout_z;
   vout_z = vtemp;
   }
   if (rotaxis == 1)
   {
   // x ist zu y geworden
   // y ist zu z geworden
   // z ist zu x geworden
   float *vtemp;
   vtemp = vout_y;
   vout_x = vout_z;
   vout_y = vout_x;
   vout_z = vtemp;
   }
   */

   // set ports
   // vector
   p_velo_out->setCurrentObject(v_out_vector);

   return SUCCESS;
}

MODULE_MAIN(VISiT, relabs)
