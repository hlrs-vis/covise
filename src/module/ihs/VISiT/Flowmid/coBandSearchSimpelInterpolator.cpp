#include "coBandSearchSimpelInterpolator.h"
#include "FlowmidUtil.h"
#include <iostream>
#include <math.h>
#include <limits.h>

coBandSearchSimpelInterpolator::coBandSearchSimpelInterpolator(coDistributedObject *cellobj, coDistributedObject *dataobj)
{
   //Constructor must be secured against invalid cell- & data-objects e.g. rmin == rmax
   polygons = (coDoPolygons *)cellobj;

   int *p_list;
   int *c_list;
   int lengthc_list;
   double min,max;
   double r;
   pair<set<int>::iterator,bool> pr;

   lengthc_list = polygons->getNumVertices();
   polygons->getAddresses(&xcoords, &ycoords, &zcoords, &c_list, &p_list);

   ratio = -6.0;
                                                  //Schleppzeiger zur Bestimmung von rmax/rmin initialisieren
   min = max = sqrt(pow(xcoords[c_list[0]],2.0)+pow(ycoords[c_list[0]],2.0));

   for(int i(0); i<lengthc_list; i++)
   {
      pr = nodeset.insert(c_list[i]);
      if(pr.second)
      {
         r=sqrt(pow(xcoords[c_list[i]],2.0)+pow(ycoords[c_list[i]],2.0));
         if(r<min)
            min = r;
         if(r>max)
            max = r;
      }
   }

   source_rmin = min;
   source_rmax = max;

   //weiter im Text mit Gruppeneinteilung
   iband = ((int)sqrt(nodeset.size()))-1;         //Bandeinteilung: muss mit if(rmin != rmax) gekapselt werden!
   zerlegeBandIntervalle();

   for(int i(0); i<iband; i++)
   {
      git_grp *band = new git_grp();
      band->rmin = bandIntervalle[i];
      band->rmax = bandIntervalle[i+1];
      baender[i] = band;
   }
   baender[0]->rmin -= ((source_rmax - source_rmin)/50);
   baender[iband-1]->rmax += ((source_rmax - source_rmin)/50);

   source_rmin = baender[0]->rmin;
   source_rmax = baender[iband-1]->rmax;

   //Knoten in die Baender verteilen
   for(set<int>::const_iterator sci = nodeset.begin(); sci != nodeset.end(); sci++)
   {
      r = sqrt(pow(xcoords[*sci],2.0)+pow(ycoords[*sci],2.0));
      for(int i(0); i<iband; i++)
      {
         if(r >= baender[i]->rmin && r < baender[i]->rmax)
         {
            baender[i]->nodelist.push_back(*sci);
         }
      }
   }

   //weiter im Text mit dataobj
   if(dataobj->isType("USTSDT"))
   {
      type = "Scalar";
      scalardata = (coDoFloat *)dataobj;
   }
   if(dataobj->isType("USTVDT"))
   {
      type = "Field";
      fielddata = (coDoVec3 *)dataobj;
   }

   //weiter im Text mit mitteln der Gruppen
   gruppenMitteln();
   gruppenCheck();

   //Create verbose Output for Controll Purposes when coupling Downstream
   if(type == "Field")
   {

      float *vx_data;
      float *vy_data;
      float *w_data;

      double x(0.0);
      double y(0.0);
      double z(0.0);
      double vx(0.0);
      double vy(0.0);
      double vu(0.0);
      double vr(0.0);
      double w(0.0);
      double vu_max(0.0);
      double vu_min(0.0);
      double vr_max(0.0);
      double vr_min(0.0);
      double w_max(0.0);
      double w_min(0.0);
      double vx_max_src(0.0);
      double vy_max_src(0.0);
      double vz_max_src(0.0);
      double vx_min_src(0.0);
      double vy_min_src(0.0);
      double vz_min_src(0.0);

      /*<tmp>*/
      double v_xy(0.0);
      double alpha(0.0);
      double beta(0.0);
      double vu_2(0.0);
      double vr_2(0.0);
      /*</tmp>*/

      vu_max = vr_max = w_max = -FLT_MAX;
      vu_min = vr_min = w_min = FLT_MAX;

      fielddata->getAddresses(&vx_data,&vy_data,&w_data);

      ofstream ofs("sourceinfo.out");

      ofs << "*************************** Source Plane Nodes ***************************" << endl;

      vx_max_src = vy_max_src = vz_max_src = -FLT_MAX;
      vx_min_src = vy_min_src = vz_min_src = FLT_MAX;

      for(set<int>::const_iterator sci = nodeset.begin(); sci != nodeset.end(); sci++)
      {

         x = xcoords[*sci];
         y = ycoords[*sci];
         z = zcoords[*sci];

         vx = vx_data[*sci];
         vy = vy_data[*sci];
         w = w_data[*sci];

         if(vx < vx_min_src)
            vx_min_src = vx;
         if(vy < vy_min_src)
            vy_min_src = vy;
         if(w < vz_min_src)
            vz_min_src = w;

         if(vx > vx_max_src)
            vx_max_src = vx;
         if(vy > vy_max_src)
            vy_max_src = vy;
         if(w > vz_max_src)
            vz_max_src = w;

         r = sqrt(pow(x,2.0)+pow(y,2.0));
         vu = (vy*x - vx*y)/r;
         vr = (vx*x + vy*y)/r;
         /*<tmp>*/
         v_xy = sqrt(pow(vx,2.0) + pow(vy,2.0));
         alpha = atan2(y,x);
         beta = atan2(vy,vx);
         vr_2 = -cos(alpha-beta)*v_xy;
         vu_2 = -sin(alpha-beta)*v_xy;
         /*</tmp>*/

         if(vu > vu_max)
            vu_max = vu;
         if(vu < vu_min)
            vu_min = vu;

         if(vr > vr_max)
            vr_max = vr;
         if(vr < vr_min)
            vr_min = vr;

         if(w > w_max)
            w_max = w;
         if(w < w_min)
            w_min = w;

         //ofs << *sci << "\t" << r << "\t" << vu << "\t" << vr << "\t" << w
         //				<< "\t" << vu_2 << "\t" << vr_2 << ", vx = " << vx << ", vy = " << vy << ", w = " << w << endl;
         ofs << *sci << "  " << x << "  " << y << "  " << r << "  " << vu << "  " << vr << "  " << w
            << "  " << vx << "  " << vy << "  " << w << endl;
         /*<tmp>
         for(int i(0); i<6; i++) {
            ofs << *sci << "\t";
            if(i == 0)
               ofs << vx_data[*sci] << endl;
            if(i == 1)
               ofs << vy_data[*sci] << endl;
            if(i == 2)
               ofs << w_data[*sci] << endl;
            if(i == 3)
               ofs << 0.0 << endl;
         if(i == 4)
         ofs << 0.0 << endl;
         if(i == 5)
         ofs << 0.0 << endl;
         }
         </tmp>*/
         //ofs << *sci << "\t" << r << "\t" << vu << "\t" << vr << "\t" << w << endl;

      }

      ofs <<  "vu_min = " << vu_min << ", vu_max = " << vu_max << endl;
      ofs << "vr_min = " << vr_min << ", vr_max = " << vr_max << endl;
      ofs << "w_min = " << w_min << ", w_max = " << w_max << endl;
      ofs << "vx_min_src = " << vx_min_src << ", vx_max_src = " << vx_max_src << endl;
      ofs << "vy_min_src = " << vy_min_src << ", vy_max_src = " << vy_max_src << endl;
      ofs << "vz_min_src = " << vz_min_src << ", vz_max_src = " << vz_max_src << endl;

      ofs.close();
   }
}


int coBandSearchSimpelInterpolator::getScalarValue(double x, double y, double z, double *scalar)
{

   double r_org, r;
   int i;

   z = 0.0;                                       //z nicht benutzt, Initialisierung um warning zu vermeiden
   r_org = sqrt(pow(x,2.0)+pow(y,2.0));
   r = FlowmidClasses::FlowmidUtil::mapIntervall(source_rmin,source_rmax,target_rmin,target_rmax,r_org);

   if(r < source_rmin || source_rmax > r)         //To be replaced by correct exception-handling!
   {
      cout << "Error: Interval mapping failed, out of Bounds." << endl;
      return ERROR;
   }

   if(type != "Scalar")
   {
      cout << "Wrong interpolator type." << endl; //To be replaced by correct exception-handling!
      return ERROR;
   }

   for(i=0; i<iband-1; i++)
   {
      if(baender[i]->r <= r && baender[i+1]->r > r)
      {
         *scalar = linearInterpolation(baender[i]->r,baender[i]->scalar,baender[i+1]->r,baender[i+1]->scalar,r);
         break;
      }
   }

   if(baender[iband-1]->r == r)
      *scalar = baender[iband-1]->scalar;

   if(r < baender[0]->r)
      *scalar = linearInterpolation(baender[0]->r,baender[0]->scalar,baender[1]->r,baender[1]->scalar,r);

   if(r > baender[iband-1]->r)
      *scalar = linearInterpolation(baender[iband-2]->r,baender[iband-2]->scalar,baender[iband-1]->r,baender[iband-1]->scalar,r);

   return SUCCESS;
}


int coBandSearchSimpelInterpolator::getFieldValue(double x, double y, double z, double *field)
{

   double r_org, r;
   double v_u=0.0, v_r=0.0;
   int i;

   z = 0.0;                                       //z nicht benutzt, Initialisierung um warning zu vermeiden
   r_org = sqrt(pow(x,2.0)+pow(y,2.0));
   r = FlowmidClasses::FlowmidUtil::mapIntervall(source_rmin,source_rmax,target_rmin,target_rmax,r_org);

   if(r < source_rmin || r > source_rmax)         //To be replaced by correct exception-handling!
   {
      cout << "Error: Interval mapping failed, out of Bounds." << endl;
      return ERROR;
   }

   if(type != "Field")
   {
      cout << "Wrong interpolator type." << endl; //To be replaced by correct exception-handling!
      return ERROR;
   }

   for(i=0; i<iband-1; i++)
      if(baender[i]->r <= r && baender[i+1]->r > r)
   {
      v_u = linearInterpolation(baender[i]->r,baender[i]->field[0],baender[i+1]->r,baender[i+1]->field[0],r);
      v_r = linearInterpolation(baender[i]->r,baender[i]->field[1],baender[i+1]->r,baender[i+1]->field[1],r);
      field[2] = linearInterpolation(baender[i]->r,baender[i]->field[2],baender[i+1]->r,baender[i+1]->field[2],r);
      break;
   }

   if(baender[iband-1]->r == r)
   {
      v_u = baender[iband-1]->field[0];
      v_r = baender[iband-1]->field[1];
      field[2] = baender[iband-1]->field[2];
   }

   if(r < baender[0]->r)
   {
      v_u = linearInterpolation(baender[0]->r,baender[0]->field[0],baender[1]->r,baender[1]->field[0],r);
      v_r = linearInterpolation(baender[0]->r,baender[0]->field[1],baender[1]->r,baender[1]->field[1],r);
      field[2] = linearInterpolation(baender[0]->r,baender[0]->field[2],baender[1]->r,baender[1]->field[2],r);
   }

   if(r > baender[iband-1]->r)
   {
      v_u = linearInterpolation(baender[iband-2]->r,baender[iband-2]->field[0],baender[iband-1]->r,baender[iband-1]->field[0],r);
      v_r = linearInterpolation(baender[iband-2]->r,baender[iband-2]->field[1],baender[iband-1]->r,baender[iband-1]->field[1],r);
      field[2] = linearInterpolation(baender[iband-2]->r,baender[iband-2]->field[2],baender[iband-1]->r,baender[iband-1]->field[2],r);
   }

   field[0] = (v_r*x - v_u*y)/r;                  //r statt r_org
   field[1] = (v_u*x + v_r*y)/r;

   return SUCCESS;
}


void coBandSearchSimpelInterpolator::setTargetArea(coDoPolygons *target_polygons, coDoIntArr *target_nodes)
{

   int lengthc_list(0);                           //TODO: Flag einfuehren zur Abfrage ob target_* gesetzt.

   int *p_list = NULL;
   int *c_list = NULL;

   double r;
   float *xcoords, *ycoords, *zcoords;

   targetPolygons = target_polygons;
   targetNodes = target_nodes;

   lengthc_list = target_polygons->getNumVertices();
   target_polygons->getAddresses(&xcoords,&ycoords,&zcoords,&c_list,&p_list);

   r = sqrt(pow(xcoords[c_list[0]],2.0) + pow(ycoords[c_list[0]],2.0));
   target_rmin = target_rmax = r;

   for(int k(0); k<lengthc_list; k++)
   {
      r = sqrt(pow(xcoords[c_list[k]],2.0) + pow(ycoords[c_list[k]],2.0));
      if(r<target_rmin)
         target_rmin = r;
      if(r>target_rmax)
         target_rmax = r;
   }
}


string coBandSearchSimpelInterpolator::getType()
{
   return type;
}


void coBandSearchSimpelInterpolator::zerlegeBandIntervalle()
{

   int odd;
   double mstep,s1,s2,ds,step;
   double parval(0);

   odd = iband%2;
   mstep = (source_rmax-source_rmin)/(iband-1);
   if(ratio < 0)
      ratio = -1/ratio;

   s1 = s2 = 0.0;
   if(odd)
   {
      s1 = 2*mstep*ratio/(ratio+1);
      s2 = 2*mstep/(1+ratio);
   }
   else
   {
      s1 = 2*(source_rmax-source_rmin)*ratio/(ratio*iband+iband-2);
      s2 = 2*(source_rmax-source_rmin)/(ratio*iband+iband-2);
   }

   ds = s2 - s1;
   parval = source_rmin;

   bandIntervalle.push_back(parval);              //evtl. fuer Bandintervalle vector durch map ersetzen

   for(int i(0); i < iband; i++)
   {
      if(odd)
      {
         if(i <= (iband-1)/2)
            step = s1+ds*2*(i-1)/(iband-3);
         else
            step = s2-ds*2*(i-1-0.5*(iband-1))/(iband-3);
      }
      else
      {
         if(i <= (iband/2))
            step = s1+ds*2*(i-1)/(iband-2);
         else
            step = s2-ds*2*(i-0.5*iband)/(iband-2);
      }
      parval += step;
      bandIntervalle.push_back(parval);
   }

   bandIntervalle.push_back(source_rmax);
}


void coBandSearchSimpelInterpolator::gruppenMitteln()
{

   int ikn;
   double r, sumr;
   float *scalarval;
   float x,y;                                     //TODO: Code Revision, bei Berechnung von r mit x & y statt xkoord[ikn]
   float vx,vy;
   float vu,vr;
   float *fieldval1, *fieldval2, *fieldval3;
   double sumscalar;
   double sumfield1, sumfield2, sumfield3;

   if(type == "Scalar")
      scalardata->getAddress(&scalarval);
   if(type == "Field")
      fielddata->getAddresses(&fieldval1,&fieldval2,&fieldval3);

   for(int i(0); i<iband; i++)
   {
      sumr = 0.0;
      sumfield1 = sumfield2 = sumfield3 = 0.0;
      sumscalar = 0.0;
      for(int j(0); j<baender[i]->nodelist.size(); j++)
      {
         ikn = baender[i]->nodelist[j];
         r = sqrt(pow(xcoords[ikn],2.0)+pow(ycoords[ikn],2.0));
         sumr += r/(baender[i]->nodelist.size());
         if(type == "Scalar")
         {
            sumscalar += scalarval[ikn]/(baender[i]->nodelist.size());
         }
         if(type == "Field")
         {
            x = xcoords[ikn];
            y = ycoords[ikn];
            vx = fieldval1[ikn];
            vy = fieldval2[ikn];
            vu = (vy*x - vx*y)/r;
            vr = (vx*x + vy*y)/r;
            sumfield1 += vu/(baender[i]->nodelist.size());
            sumfield2 += vr/(baender[i]->nodelist.size());
            sumfield3 += fieldval3[ikn]/(baender[i]->nodelist.size());
         }
      }
      baender[i]->r = sumr;
      if(type == "Scalar")
      {
         baender[i]->scalar = sumscalar;
      }
      if(type == "Field")
      {
         baender[i]->field[0] = sumfield1;
         baender[i]->field[1] = sumfield2;
         baender[i]->field[2] = sumfield3;
      }
   }
}


void coBandSearchSimpelInterpolator::gruppenCheck()
{

   int first_empty(0);
   bool empty_flag(false);

   if(baender[0]->nodelist.size() == 0)
   {
      baender[0]->r = (baender[0]->rmax + baender[0]->rmin)/2;
      baender[0]->scalar = 0.0;
      baender[0]->field[0] = 0.0;
      baender[0]->field[1] = 0.0;
      baender[0]->field[2] = 0.0;
   }
   if(baender[iband-1]->nodelist.size() == 0)
   {
      baender[iband-1]->r = (baender[iband-1]->rmax + baender[iband-1]->rmin)/2;
      baender[iband-1]->scalar = 0.0;
      baender[iband-1]->field[0] = 0.0;
      baender[iband-1]->field[1] = 0.0;
      baender[iband-1]->field[2] = 0.0;
   }

   for(int i(1); i<iband-1; i++)
   {
      if(baender[i]->nodelist.size() == 0)
      {
         if(!empty_flag)
            first_empty = i;
         empty_flag = true;
      }

      if(empty_flag && baender[i]->nodelist.size() != 0)
      {

         empty_flag = false;

         for(int k(0); k<i-first_empty; k++)
         {

            baender[first_empty + k]->r = (baender[first_empty + k]->rmax
               + baender[first_empty + k]->rmin)/2;

            baender[first_empty + k]->scalar =
               linearInterpolation(baender[first_empty-1]->r,baender[first_empty-1]->scalar,
               baender[i]->r,baender[i]->scalar,baender[first_empty + k]->r);
            baender[first_empty + k]->field[0] =
               linearInterpolation(baender[first_empty-1]->r,baender[first_empty-1]->field[0],
               baender[i]->r,baender[i]->field[0],baender[first_empty + k]->r);
            baender[first_empty + k]->field[1] =
               linearInterpolation(baender[first_empty-1]->r,baender[first_empty-1]->field[1],
               baender[i]->r,baender[i]->field[1],baender[first_empty + k]->r);
            baender[first_empty + k]->field[2] =
               linearInterpolation(baender[first_empty-1]->r,baender[first_empty-1]->field[2],
               baender[i]->r,baender[i]->field[2],baender[first_empty + k]->r);
         }
      }

   }
}


inline double coBandSearchSimpelInterpolator::linearInterpolation(double x1, double y1, double x2, double y2, double x)
{

   double y(0.0);

   y = y1 + (y2 - y1)/(x2 - x1) * (x - x1);

   return y;
}


void coBandSearchSimpelInterpolator::writeInfo(char *fname)
{

   int i(0);

   ofstream ofs(fname);

   //ofs << "************* Knoten Informationen *************" << endl;
   //ofs << "Anzahl Knoten: " << nodeset.size() << endl;		//TODO: Mehr info, aus fielddata & scalardata fuer Debugging
   /*for(set<int>::const_iterator sci = nodeset.begin(); sci != nodeset.end(); sci++) {
      ofs << "i: " << i++ << ", Knoten-Nummer: " << *sci << endl;
   }*/

   //ofs << "source_rmin = " << source_rmin << ", source_rmax = " << source_rmax << endl;
   //ofs << "target_rmin = " << target_rmin << ", target_rmax = " << target_rmax << endl;

   //ofs << "********** Mittelwerte der einzelnen Baender **********" << endl;
   //ofs << "iband = " << iband << endl;

   for(i = 0; i < iband; i++)
   {
      //ofs << "baender[" << i << "]->r = " << baender[i]->r;
      ofs << baender[i]->r;
      //ofs << ", baender[" << i << "]->rmin = " << baender[i]->rmin << ", baender[" << i << "]->rmax" << baender[i]->rmax << ", baender[" << i << "]->scalar =  " << baender[i]->scalar;
      for(int j(0); j<3; j++)
         ofs << "\t" << baender[i]->field[j];
      //ofs << ", field[" << j << "] " << baender[i]->field[j];
      ofs << endl;
   }

   ofs.close();
}
