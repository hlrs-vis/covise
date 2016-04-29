/**************************************************************************\ 
 **                                                              2002      **
 **                                                                        **
 ** Description:  COVISE VeloIHS     New application module               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  M. Becker                                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  01.07.02  V1.0                                                  **
\**************************************************************************/

#include <math.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "VeloIHS.h"

VeloIHS::VeloIHS(int argc, char *argv[])
: coSimpleModule(argc, argv, "VeloIHS")
{
   //ports
   p_grid = addInputPort ("grid","UnstructuredGrid|Polygons","computation grid");
   p_velo_in  = addInputPort ("velocity", "Vec3","input vector data");

   p_vu_vector_out = addOutputPort ("vu_vector", "Vec3","Vu (vector)");
   p_vr_vector_out = addOutputPort ("vr_vector", "Vec3","Vr (vector)");
   p_vm_vector_out = addOutputPort ("vm_vector", "Vec3","Vm (vector)");

   p_v_scalar_out = addOutputPort ("v", "Float","V (scalar)");
   p_vu_scalar_out = addOutputPort ("vu", "Float","Vu (scalar)");
   p_vr_scalar_out = addOutputPort ("vr", "Float","Vr (scalar)");
   p_vm_scalar_out = addOutputPort ("vm", "Float","Vm (scalar)");
   p_rvu_scalar_out = addOutputPort ("r_times_vu", "Float","r*Vu (scalar)");

   p_rotaxis = addChoiceParam("Rotation_axis", "ConnectionMethod");
   s_rotaxis[0] = strdup("x");
   s_rotaxis[1] = strdup("y");
   s_rotaxis[2] = strdup("z");
   p_rotaxis->setValue(3, s_rotaxis, RotZ);
}


int VeloIHS::compute(const char *)
{

   int i;

   int n;
   int num_values_grid;
   int num_values_velo;

   int rotaxis;

   float *x, *y, *z;

   coDoFloat *v_scalar_data = NULL;
   coDoFloat *vu_scalar_data = NULL;
   coDoFloat *vr_scalar_data = NULL;
   coDoFloat *vm_scalar_data = NULL;
   coDoFloat *rvu_scalar_data = NULL;

   coDoVec3 *vu_vector_data = NULL;
   coDoVec3 *vr_vector_data = NULL;
   coDoVec3 *vm_vector_data = NULL;

   rotaxis=p_rotaxis->getValue();
//   cerr << "rotaxis: " << s_rotaxis[rotaxis] << endl;

   // get grid
   const coDistributedObject *obj = p_grid->getCurrentObject();

   // ------- Unstructured grid
   if (obj->isType("UNSGRD"))
   {
      // unstructured vector data

      const coDoUnstructuredGrid *unsgrd = (const coDoUnstructuredGrid *) obj;
      int numConn,numElem;
      int *elemList,*connList;

      unsgrd->getAddresses(&elemList, &connList, &x, &y, &z);
      unsgrd->getGridSize(&numElem, &numConn, &num_values_grid);
   }
   else if (obj->isType("POLYGN"))
   {
      // polygons
      const coDoPolygons *polys = (const coDoPolygons *) obj;
      //int numConn,numElem;
      int *elemList,*connList;

      polys->getAddresses(&x, &y, &z, &connList, &elemList);
      num_values_grid = polys->getNumPoints();
      //numElem = polys->getNumPolygons();
      //numConn = polys->getNumVertices();
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

   if (!obj->isType("USTVDT"))
   {
      sendError("Received illegal type at port '%s'", p_velo_in->getName());
      return FAIL;
   }

   float *u, *v, *w;
   // unstructured vector data
   coDoVec3 *vector_data = (coDoVec3 *) obj;
   num_values_velo = vector_data->getNumPoints();
   if (rotaxis==0)                             // x-axis
      vector_data -> getAddresses(&w, &u, &v);
   else if (rotaxis==1)                        // y-axis
      vector_data -> getAddresses(&v, &w, &u);
   else                                        // z-axis
      vector_data -> getAddresses(&u, &v, &w);

   // control num_values_grid = num_values_velo
   if (num_values_grid != num_values_velo)
   {
      sendError("grid coord size and velocity array size are not the same!");
      return FAIL;
   }

   n = num_values_grid;

   // calculate output
   float r;                                       // radius of knot
   float v_xy;                                    // velocity of knot in xy-plane
   float alpha;                                   // angle of coorinates
   float beta;                                    // angle between vx and vy

   float *vabs;
   float *vu;
   float *vr;
   float *vm;
   float *rvu;

   float *vu_x, *vu_y, *vu_z;
   float *vr_x, *vr_y, *vr_z;
   float *vm_x, *vm_y, *vm_z;

   // create output objects
   //scalar
   v_scalar_data = new coDoFloat(p_v_scalar_out->getObjName(), n);
   vu_scalar_data = new coDoFloat(p_vu_scalar_out->getObjName(), n);
   vr_scalar_data = new coDoFloat(p_vr_scalar_out->getObjName(), n);
   vm_scalar_data = new coDoFloat(p_vm_scalar_out->getObjName(), n);
   rvu_scalar_data = new coDoFloat(p_rvu_scalar_out->getObjName(), n);
   //vector
   vu_vector_data = new coDoVec3(p_vu_vector_out->getObjName(), n);
   vr_vector_data = new coDoVec3(p_vr_vector_out->getObjName(), n);
   vm_vector_data = new coDoVec3(p_vm_vector_out->getObjName(), n);

   // get pointers
   //scalar
   v_scalar_data -> getAddress(&vabs);
   vu_scalar_data -> getAddress(&vu);
   vr_scalar_data -> getAddress(&vr);
   vm_scalar_data -> getAddress(&vm);
   rvu_scalar_data -> getAddress(&rvu);
   //vector
   if (rotaxis==0)                               // x-axis
   {
      vu_vector_data -> getAddresses(&vu_z, &vu_x, &vu_y);
      vr_vector_data -> getAddresses(&vr_z, &vr_x, &vr_y);
      vm_vector_data -> getAddresses(&vm_z, &vm_x, &vm_y);
   }
   else if (rotaxis==1)                          // y-axis
   {
      vu_vector_data -> getAddresses(&vu_y, &vu_z, &vu_x);
      vr_vector_data -> getAddresses(&vr_y, &vr_z, &vr_x);
      vm_vector_data -> getAddresses(&vm_y, &vm_z, &vm_x);
   }
   else                                          // z-axis
   {
      vu_vector_data -> getAddresses(&vu_x, &vu_y, &vu_z);
      vr_vector_data -> getAddresses(&vr_x, &vr_y, &vr_z);
      vm_vector_data -> getAddresses(&vm_x, &vm_y, &vm_z);
   }

   // calculation

   float factor;

   for (i=0; i<n; i++)
   {
      r       = sqrt (  (x[i]*x[i]) + (y[i]*y[i])  );
      v_xy    = sqrt (  (u[i]*u[i]) + (v[i]*v[i])  );
      alpha   = atan2 ( y[i] , x[i] );
      beta    = atan2 ( v[i] , u[i] );

      vabs[i] = sqrt ( u[i]*u[i] + v[i]*v[i] + w[i]*w[i] );
      vr[i] = -cos ( alpha-beta ) * v_xy;
      vu[i] = -sin ( alpha-beta ) * v_xy;
      vm[i] = sqrt (  (w[i]*w[i]) + (vr[i]*vr[i])   );
      rvu[i] = r * vu[i];

      // vu vector
      factor = vu[i] / r;
      vu_x[i] = factor * (-y[i]);
      vu_y[i] = factor * (x[i]);
      // vu_z[i] = 0!

      // vr vector
      factor = vr[i] / r;
      vr_x[i] = factor * x[i];
      vr_y[i] = factor * y[i];
      // vr_z = 0!

      // vm vector
      vm_x[i] = vr_x[i];
      vm_y[i] = vr_y[i];
      vm_z[i] = w[i];
   }

   // z-components of vu and vr are = 0
   memset (vu_z, 0, n * sizeof(int));
   memset (vr_z, 0, n * sizeof(int));

   // set ports
   // scalar
   
   vu_scalar_data->addAttribute("SPECIES", "vu");
   p_vu_scalar_out->setCurrentObject(vu_scalar_data);
   vr_scalar_data->addAttribute("SPECIES", "vr");
   p_vr_scalar_out->setCurrentObject(vr_scalar_data);
   vm_scalar_data->addAttribute("SPECIES", "vm");
   p_vm_scalar_out->setCurrentObject(vm_scalar_data);
   rvu_scalar_data->addAttribute("SPECIES", "r*vu");
   p_rvu_scalar_out->setCurrentObject(rvu_scalar_data);
   // vector
   vu_vector_data->addAttribute("SPECIES", "vu");
   p_vu_vector_out->setCurrentObject(vu_vector_data);
   vr_vector_data->addAttribute("SPECIES", "vr");
   p_vr_vector_out->setCurrentObject(vr_vector_data);
   vm_vector_data->addAttribute("SPECIES", "vm");
   p_vm_vector_out->setCurrentObject(vm_vector_data);

   return SUCCESS;
}

MODULE_MAIN(VISiT, VeloIHS)
