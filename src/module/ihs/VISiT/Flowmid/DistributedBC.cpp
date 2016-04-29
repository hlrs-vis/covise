#include "DistributedBC.h"
#include "api/coModule.h"

DistributedBC::DistributedBC(coDoSet *distbocoSet)
{

   bocoArr = distbocoSet->getAllElements(&igeb);
   pboco = new coDistributedObject *[igeb+1];
   pboco[igeb] = NULL;
};

int DistributedBC::setBoundaryCondition(coInterpolator *interpolator, coDoPolygons *target_polygons)
{

   int ret(0);

   for(int i(0); i<igeb; i++)
   {

      coDistributedObject *boc[RNUM];
      boc[RNUM-1] = NULL;

      char *setname;
      int num;

      coDoSet *bocoobjSet = (coDoSet *)bocoArr[i];
      coDistributedObject *const *bocoobjArr = bocoobjSet->getAllElements(&num);
      setname = bocoobjSet->getName();

      for(int j(0); j<num; j++)
      {
         if(bocoobjArr[j]->isType("INTARR"))
            boc[j] = FlowmidClasses::FlowmidUtil::dupIntArr((coDoIntArr *)bocoobjArr[j]);
         if(bocoobjArr[j]->isType("USTSDT"))
            boc[j] = FlowmidClasses::FlowmidUtil::dupFloatArr((coDoFloat *)bocoobjArr[j]);
      }

      if(interpolator->getType() == "Field")
      {

         int size(0);
         int numdim(0);
         int points(0);

         int *c_list = NULL;
         int *p_list = NULL;
         int *adr = NULL;

         //double r;
         double x,y,z;
         double vxyz[3] = {0.0,0.0,0.0};

         float *dirichletadr = NULL;
         float *xcoords,*ycoords,*zcoords;

         coDoIntArr *intarr = (coDoIntArr *)boc[1];
         size = intarr->getSize();
         numdim = intarr->getNumDimensions();
         adr = intarr->getAddress();

         interpolator->setTargetArea(target_polygons,intarr);

         coDoFloat *dirichletval = (coDoFloat *)boc[13];
         dirichletval->getAddress(&dirichletadr);

         target_polygons->getAddresses(&xcoords,&ycoords,&zcoords,&c_list,&p_list);
         points = target_polygons->getNumPoints();

         if(numdim != 1)
            cout << "Warning: dirichlet-node array has wrong dimension." << endl;

         if(size > 1)
         {

            ofstream ofs("targetinfo.out");
            double vx_min_tar(0.0);
            double vy_min_tar(0.0);
            double vz_min_tar(0.0);
            double vx_max_tar(0.0);
            double vy_max_tar(0.0);
            double vz_max_tar(0.0);
            double w(0.0);
            double vu_min(0.0);
            double vu_max(0.0);
            double vr_min(0.0);
            double vr_max(0.0);
            double vz_min(0.0);
            double vz_max(0.0);

            vx_min_tar = vy_min_tar = vz_min_tar = FLT_MAX;
            vx_max_tar = vy_max_tar = vz_max_tar = -FLT_MAX;

            vu_min = vr_min = vz_min = FLT_MAX;
            vu_max = vr_max = vz_max = -FLT_MAX;

            for(int k(0); k<size-1; k+=6)
            {

               double r,vx,vy,vu,vr,vz;

               x = xcoords[adr[k]];
               y = ycoords[adr[k]];
               z = zcoords[adr[k]];
               r = sqrt(pow(x,2.0) + pow(y,2.0));

               if(!interpolator->getFieldValue(x,y,z,vxyz))
                  cout << "Warning: error in interpolator." << endl;

               vx = vxyz[0];
               vy = vxyz[1];
               vz = vxyz[2];
               w = vxyz[2];
               vu = (vy*x - vx*y)/r;
               vr = (vx*x + vy*y)/r;
               ofs << adr[k] << "  " << x << "  " << y << "  " << r << "  " << vu << "  " << vr << "  " << vxyz[2]
                  << "  " << vx << "  " << vy << "  " << w;
               ofs << endl;

               if(vx < vx_min_tar)
                  vx_min_tar = vx;
               if(vy < vy_min_tar)
                  vy_min_tar = vy;
               if(w < vz_min_tar)
                  vz_min_tar = w;
               if(vx > vx_max_tar)
                  vx_max_tar = vx;
               if(vy > vy_max_tar)
                  vy_max_tar = vy;
               if(w > vz_max_tar)
                  vz_max_tar = w;

               if(vu < vu_min)
                  vu_min = vu;
               if(vr < vr_min)
                  vr_min = vr;
               if(vz < vz_min)
                  vz_min = vz;
               if(vu > vu_max)
                  vu_max = vu;
               if(vr > vr_max)
                  vr_max = vr;
               if(vz > vz_max)
                  vz_max = vz;

               for(int l(0); l<3; l++)
                  dirichletadr[k+l] = (float)vxyz[l];
            }
            ofs << "vu_min = " << vu_min << ", vu_max = " << vu_max << endl;
            ofs << "vr_min = " << vr_min << ", vr_max = " << vr_max << endl;
            ofs << "vx_min_tar = " << vx_min_tar << ", vx_max_tar = " << vx_max_tar << endl;
            ofs << "vy_min_tar = " << vy_min_tar << ", vy_max_tar = " << vy_max_tar << endl;
            ofs << "vz_min_tar = " << vz_min_tar << ", vz_max_tar = " << vz_max_tar << endl;
            ofs.close();
         }
      }

      if(interpolator->getType() == "Scalar")
      {

         int size(0);
         int numdim(0);
         int points(0);

         int *c_list = NULL;
         int *p_list = NULL;
         int *adr = NULL;

         double x_m,y_m,z_m;
         double scalar;

         float *pressureadr = NULL;
         float *xcoords,*ycoords,*zcoords;

         coDoIntArr *intarr = (coDoIntArr *)boc[8];
         size = intarr->getSize();
         numdim = intarr->getNumDimensions();
         adr = intarr->getAddress();

         interpolator->setTargetArea(target_polygons,intarr);

         coDoFloat *pressureval = (coDoFloat *)boc[14];
         pressureval->getAddress(&pressureadr);

         target_polygons->getAddresses(&xcoords,&ycoords,&zcoords,&c_list,&p_list);
         points = target_polygons->getNumPoints();

         if(numdim != 1)
            cout << "Warning: pressure-node array has wrong dimension." << endl;

         if(size > 1)
         {

            for(int k(0); k<size-1; k+=4)
            {

               x_m = y_m = z_m = 0.0;

               for(int l(0); l<4; l++)
               {
                  x_m += xcoords[adr[l*size/4+k/4]-1]/4;
                  y_m += ycoords[adr[l*size/4+k/4]-1]/4;
               }

               if(!interpolator->getScalarValue(x_m,y_m,z_m,&scalar))
                  cout << "Warning: error in interpolator." << endl;

               pressureadr[k/4] = (float)scalar;
            }
         }
      }

      pboco[i] = new coDoSet(setname,boc);
   }

   return ret;
};

coDistributedObject **DistributedBC::getPortObj()
{

   return pboco;
};
