#include "ShowDistBoco.h"
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>

ShowDistBoco::ShowDistBoco(int argc, char *argv[])
                                                  // description in the module setup window
:coModule(argc, argv, "IHS - Darstellung von verteilten Randbedingungen")
{
   // input ports ******************************************************************************************************

   //input ports
   gridInPort = addInputPort("gridInPort","UnstructuredGrid","Grid");
   cellInPort = addInputPort("cellInPort","Polygons","Cells");
                                                  // From DomainDecomposition Runner!
   distbocoInputPort = addInputPort("distbocoInputPort","USR_DistFenflossBoco","Distributed Boundary Conditions");

   // output ports ******************************************************************************************************
   velocityOutPort = addOutputPort("velocityOutPort","Vec3","Velocity");

}


int ShowDistBoco::compute(const char *)
{
   bool exec_flag(true);

   const coDistributedObject *gridObj = gridInPort->getCurrentObject();
   const coDistributedObject *cellObj = cellInPort->getCurrentObject();
   const coDistributedObject *distbocoObj = distbocoInputPort->getCurrentObject();

   if(!gridObj)                                   //if(!(gridObj || cellObj || distbocoObj))
      exec_flag = false;
   if(!cellObj)
      exec_flag = false;
   if(!distbocoObj)
      exec_flag = false;

   if(exec_flag)
   {
      if(!gridObj->isType("UNSGRD"))
         exec_flag = false;
      if(!cellObj->isType("POLYGN"))
         exec_flag = false;
      if(!distbocoObj->isType("SETELE"))
         exec_flag = false;
   }

   if(exec_flag)
   {

      sendInfo("Building coDoVec3 Object ...");

      int igeb(0);
      int numConn(0);
      int numElem(0);
      int n_gridpoints(0);
      //int lengthc_list(0);
      int *elemList = NULL;
      int *connList = NULL;
      int *p_list = NULL;
      int *c_list = NULL;

      float *x_g;
      float *y_g;
      float *z_g;

      float *x_p;
      float *y_p;
      float *z_p;

      float *u;
      float *v;
      float *w;

      //set<int> velo_nodes;
      //pair<set<int>::iterator,bool> pr;

      //coDistributedObject **pboco;

      const coDoUnstructuredGrid *grid = (const coDoUnstructuredGrid *)gridObj;

      const coDoSet *distbocoDSSetObj = (const coDoSet *)distbocoObj;

      const coDistributedObject *const *bocoArr = distbocoDSSetObj->getAllElements(&igeb);

      grid->getAddresses(&elemList,&connList,&x_g,&y_g,&z_g);
      grid->getGridSize(&numElem,&numConn,&n_gridpoints);

      coDoPolygons *inlet_polygons = (coDoPolygons *)cellObj;

      //lengthc_list = inlet_polygons->getNumVertices();
      inlet_polygons->getAddresses(&x_p,&y_p,&z_p,&c_list,&p_list);

      /*for(int i(0); i<lengthc_list; i++) {
         velo_nodes.insert(c_list[i]);
      }*/

      coDoVec3 *bc_velocity =
         new coDoVec3(velocityOutPort->getObjName(),n_gridpoints);

      bc_velocity->getAddresses(&u,&v,&w);

      ofstream ofs_v("veloinfo.out");
      ofstream ofs_p("pressinfo.out");

      for(int i(0); i<igeb; i++)
      {

         int num(0);
         int size(0);
         int points(0);
         int number(0);

         double x(0.0);
         double y(0.0);
         double r(0.0);

         double _u(0.0);
         double _v(0.0);
         double _w(0.0);

         int *adr = NULL;

         float *dirichletadr = NULL;

         const coDoSet *bocoobjSet = (const coDoSet *)bocoArr[i];
         const coDistributedObject *const *bocoobjArr = bocoobjSet->getAllElements(&num);

         coDoIntArr *intarr = (coDoIntArr *)bocoobjArr[1];
         coDoFloat *dirichletval = (coDoFloat *)bocoobjArr[13];

         size = intarr->getSize();
         adr = intarr->getAddress();

         points = dirichletval->getNumPoints();
         dirichletval->getAddress(&dirichletadr);

         if(size != points)
            sendInfo("This actually shouldn't happen, so better check this out ...");
         //cout << "size = " << size << ", points = " << points << endl;

         for(int j(0); j<size-1; j+=6)            //Warum ist size = points+1?????
         {

            number = adr[j];

            /*u[adr[j]] = dirichletadr[j];		//adr[j] ist Knoten-Nummer aus UNSGRD
            v[adr[j]] = dirichletadr[j+1];
            w[adr[j]] = dirichletadr[j+2];*/

            _u = dirichletadr[j];
            _v = dirichletadr[j+1];
            _w = dirichletadr[j+2];

            /*u[number] = dirichletadr[j];		//adr[j] ist Knoten-Nummer aus UNSGRD
            v[number] = dirichletadr[j+1];
            w[number] = dirichletadr[j+2];*/

            u[number] = _u;                       //adr[j] ist Knoten-Nummer aus UNSGRD
            v[number] = _v;
            w[number] = _w;

            x = x_g[adr[j]];
            y = y_g[adr[j]];

            r = sqrt(pow(x,2.0) + pow(y,2.0));
            //r = sqrt(pow(x_g[adr[j]],2.0) + pow(y_g[adr[j]],2.0));

            ofs_v << number << "  " << x << "  " << y << "  " << r << "  " << _u << "  " << _v << "  " << _w << endl;
         }

      }

      ofs_v.close();
      ofs_p.close();

      velocityOutPort->setCurrentObject(bc_velocity);
      //setCurrentObject(...), butt-monkey!!!
   }

   return CONTINUE_PIPELINE;
}

MODULE_MAIN(VISiT, ShowDistBoco)
