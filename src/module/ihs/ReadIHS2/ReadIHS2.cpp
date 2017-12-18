/**************************************************************************\ 
 **                                                              2002      **
 **                                                                        **
 ** Description:  ReadIHS2           New application module               **
 **                                                                        **
 ** Covise input module for 3D IHS files                                   **
 **                                                                        **
 **                                                                        **
 ** Author:  M. Becker                                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  19.12.02  V1.0                                                  **
 **        03.03.03  V1.1                                                  **
 \**************************************************************************/

#include "ReadIHS2.h"
#include <math.h>
#include <do/coDoIntArr.h>

#define sqr(x) (x*x)

ReadIHS2::ReadIHS2(int argc, char *argv[])
: coModule(argc, argv, "Read data from Ihs FENFLOSS")
{

   char buf[300];

   //ports & parameters

   port_grid = addOutputPort ("grid","UnstructuredGrid","computation grid");
   port_velocity  = addOutputPort ("velocity", "Vec3","output velocity");
   port_pressure = addOutputPort ("pressure", "Float","output pressure");
   port_k = addOutputPort ("k", "Float","output k");
   port_eps = addOutputPort ("eps", "Float","output eps");
   port_boco = addOutputPort ("boco", "USR_FenflossBoco", "Boundary Conditions");
   port_pressrb = addOutputPort ("press_rb", "Polygons","pressure boundary conditions");
   port_wall = addOutputPort ("wall", "Polygons","wall elements");
   port_bila = addOutputPort ("bila_elems", "Polygons","marked elements");
   port_bcin = addOutputPort ("bcin", "Polygons","inlet elements");
   

#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE");
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(defaultDir)
      sprintf(buf,"%s/",defaultDir);
   else
      sprintf(buf,"/data/");
   p_geoFile = addFileBrowserParam("geoFile","Geometry File");
   p_geoFile->setValue(buf,"*.geo;*.GEO");

   p_rbFile = addFileBrowserParam("rbFile","Geometry File");
   p_rbFile->setValue(buf,"*.rb;*.bc;*.RB;*.BC");

   p_simFile = addFileBrowserParam("simFile","Geometry File");
   p_simFile->setValue(buf,"*.sim;*.erg;*.sim*");

   p_readsim = addBooleanParam("ReadSimfile", "yes or no");
   p_readsim->setValue(1);

   p_dimension = addInt32Param("Dimension", "dimension");
   p_dimension->setValue(0);

   p_numbered = addBooleanParam("NumberedConnList", "yes or no");
   p_numbered->setValue(1);

   p_scalingfactor = addFloatParam("ScalingFactor", "scaling factor for geometry");
   p_scalingfactor->setValue(1.0);

   p_showallbilas = addBooleanParam("ShowAllBilas", "yes or no");
   p_showallbilas->setValue(0);

   p_bilanr1 = addInt32Param("bilanr1", "nr of marked elements");
   p_bilanr1->setValue(0);

   p_bilanr2 = addInt32Param("bilanr2", "nr of marked elements");
   p_bilanr2->setValue(0);

   p_bilanr3 = addInt32Param("bilanr3", "nr of marked elements");
   p_bilanr3->setValue(0);

   p_bilanr4 = addInt32Param("bilanr4", "nr of marked elements");
   p_bilanr4->setValue(0);

   p_bilanr5 = addInt32Param("bilanr5", "nr of marked elements");
   p_bilanr5->setValue(0);

   p_showallwalls = addBooleanParam("ShowAllWalls", "yes or no");
   p_showallwalls->setValue(0);

   p_wallnr1 = addInt32Param("wallnr1", "nr of wall elements");
   p_wallnr1->setValue(0);

   p_wallnr2 = addInt32Param("wallnr2", "nr of wall elements");
   p_wallnr2->setValue(0);

   p_wallnr3 = addInt32Param("wallnr3", "nr of wall elements");
   p_wallnr3->setValue(0);

   p_wallnr4 = addInt32Param("wallnr4", "nr of wall elements");
   p_wallnr4->setValue(0);

   p_wallnr5 = addInt32Param("wallnr5", "nr of wall elements");
   p_wallnr5->setValue(0);

   p_pressnr1 = addInt32Param("pressnr1", "nr of press elements");
   p_pressnr1->setValue(0);

   p_pressnr2 = addInt32Param("pressnr2", "nr of press elements");
   p_pressnr2->setValue(0);

   p_pressnr3 = addInt32Param("pressnr3", "nr of press elements");
   p_pressnr3->setValue(0);

   p_pressnr4 = addInt32Param("pressnr4", "nr of press elements");
   p_pressnr4->setValue(0);

   p_pressnr5 = addInt32Param("pressnr5", "nr of press elements");
   p_pressnr5->setValue(0);

   p_create_boco_obj = addBooleanParam("CreateBocoObject", "yes or no");
   p_create_boco_obj->setValue(0);

   p_generate_inlet_boco = addBooleanParam("GenerateInletBoco", "yes or no");
   p_generate_inlet_boco->setValue(0);

   p_abs2rel = addBooleanParam("TransformAbs2Rel", "yes or no");
   p_abs2rel->setValue(0);

   p_n =  addFloatParam("RevolutionsPerSecond", "revolutions per second");
   p_n->setValue(250.0);

   p_RotAxis = addChoiceParam("RotationAxis", "RotationAxis");
   s_RotAxis[0] = strdup("x");
   s_RotAxis[1] = strdup("y");
   s_RotAxis[2] = strdup("z");
   p_RotAxis->setValue(3, s_RotAxis, RotX);

   p_bila_in = addInt32Param("BilanrInlet", "bilanr of inlet area");
   p_bila_in->setValue(100);

   p_bila_out = addInt32Param("BilanrOutlet", "bilanr of outlet area");
   p_bila_out->setValue(200);

   p_periodic_1 = addInt32Param("Bilanr1Periodic", "bilanr of 1st periodoc area");
   p_periodic_1->setValue(110);

   p_periodic_2 = addInt32Param("Bilanr2Periodic", "bilanr of 2nd periodic area");
   p_periodic_2->setValue(120);

}


void ReadIHS2::postInst()
{
   p_geoFile->show();                             // visible in control panel
   p_rbFile->show();
   p_simFile->show();

   p_readsim->show();

   // p_numbered->show();

   p_scalingfactor->show();

   p_showallbilas->show();
   p_showallwalls->show();

   p_bilanr1->show();
   p_bilanr2->show();
   p_bilanr3->show();
   p_bilanr4->show();
   p_bilanr5->show();

   p_wallnr1->show();
   p_wallnr2->show();
   p_wallnr3->show();
   p_wallnr4->show();
   p_wallnr5->show();

   p_pressnr1->show();
   p_pressnr2->show();
   p_pressnr3->show();
   p_pressnr4->show();
   p_pressnr5->show();

   p_create_boco_obj->show();
   p_generate_inlet_boco->show();
   p_bila_in->show();
   p_bila_out->show();
   p_periodic_1->show();
   p_periodic_2->show();

}


void ReadIHS2::quit()
{
   if (geo->x)
      delete [] geo->x; 
   if (geo->x)
      delete [] geo->y;

   if (geo->is3d)
   {
      if (geo->elem_3d)
	     delete [] geo->elem_3d;
      if (geo->z)
	     delete [] geo->z;
   }
   else
   {
      if (geo->elem_2d)
         delete [] geo->elem_2d;
   }

   // rb
   if (geo->elem_mark)
      delete [] geo->elem_mark;
   if (geo->special_mark)
      delete [] geo->special_mark;
   if (geo->wall)
      delete [] geo->wall;
   if (geo->special_wall)
      delete [] geo->special_wall;

   // cfd
   if (geo->u)
      delete [] geo->u;
   if (geo->v)
      delete [] geo->v;
   if (geo->w)
      delete [] geo->w;
   if (geo->p)
      delete [] geo->p;
   if (geo->k)
      delete [] geo->k;
   if (geo->eps)
      delete [] geo->eps;
   if (geo->p_elem)
      delete [] geo->p_elem;

   if (geo->create_boco_object)
   {
      if (geo->bcin)
         delete [] geo->bcin;
      if (geo->bcout)
         delete [] geo->bcout;
      if (geo->bcperiodic1)
         delete [] geo->bcperiodic1;
      if (geo->bcperiodic2)
	     delete [] geo->bcperiodic2;
      if (geo->dirichlet_nodes)
         delete [] geo->dirichlet_nodes;
      if (geo->dirichlet_values)
         delete [] geo->dirichlet_values;
   }

   if (geo->bilanrlist)
      free (geo->bilanrlist);
   if (geo->bilanames)
   {
      for (int i=0;i<geo->n_bilanrs;i++)
	  {
	     free (geo->bilanames[i]);
	  }  delete [] geo->bilanames;
   }

   if (geo->wallnrlist)
      free (geo->wallnrlist);
   if (geo->wallnames)
   {
      for (int i=0;i<geo->n_wallnrs;i++)
	  {
	     free (geo->wallnames[i]);
	  }  delete [] geo->wallnames;
   }

   if (geo->presnrlist)
      free (geo->presnrlist);
   if (geo->presnames)
   {
      for (int i=0;i<geo->n_presnrs;i++)
	  {
	     free (geo->presnames[i]);
	  }  delete [] geo->presnames;
   }

   // :-)
}


int ReadIHS2::compute(const char *)
{

   geo = new struct geometry;

   printf("\n\n**********************************\n");
   printf("         ReadIHS2 Start          \n");
   printf("**********************************\n");

   geo->isgeo=0;
   geo->isrb=0;
   geo->issim=0;
   geo->is3d=0;

   geo->create_boco_object=0;

   geo->numbered = p_numbered->getValue();

   geo->scalingfactor = p_scalingfactor->getValue();

   // Covise data arrays
   coDoVec3 *velocity = NULL;
   coDoFloat *press = NULL;
   coDoFloat *k = NULL;
   coDoFloat *eps = NULL;

   // get geofile from parameter
   Covise::getname(geo->geofile, p_geoFile->getValue());
   printf("\ngeofile: %s\n", geo->geofile);

   // get rbfile from parameter
   Covise::getname(geo->rbfile, p_rbFile->getValue());
   printf("rbfile: %s\n", geo->rbfile);

   // get simfile from parameter
   Covise::getname(geo->simfile, p_simFile->getValue());

   // is there a geo / sim / rbfile?
   Check_geo_sim_rb(geo);

   if (p_readsim->getValue() == 0)
   {
      strcpy(geo->simfile,"0");
   }

   printf("simfile: %s\n", geo->simfile);

   if (geo->isgeo==0)
   {
      sendError("Kann geofile nicht lesen!\n");
      return FAIL;
   }
   if (geo->isrb==0)
   {
      sendError("Kann rbfile nicht lesen!\n");
      return FAIL;
   }

   if ( (geo->issim==0) && ( strcmp(geo->simfile,"0") )  )
   {
      sendError("Kann simfile nicht lesen! Wenn kein simfile vorhanden, simfile = '0' setzen!\n");
      return FAIL;
   }

   //    if (geo->issim==0)
   {
      /* 
         if (p_create_boco_obj->getValue()==1)
         {
         sendError("Kann kein boco-Objekt erstellen, da kein simfile vorhanden!\n");
         p_create_boco_obj->setValue(0);
         }
       */
   }
   //    else
   {
      if (p_create_boco_obj->getValue()==1)
         geo->create_boco_object = 1;
   }

   // get bila-nr from control panel
   geo->bilanr1 = p_bilanr1->getValue();
   geo->bilanr2 = p_bilanr2->getValue();
   geo->bilanr3 = p_bilanr3->getValue();
   geo->bilanr4 = p_bilanr4->getValue();
   geo->bilanr5 = p_bilanr5->getValue();

   // get wall-nr from control panel
   geo->wallnr1 = p_wallnr1->getValue();
   geo->wallnr2 = p_wallnr2->getValue();
   geo->wallnr3 = p_wallnr3->getValue();
   geo->wallnr4 = p_wallnr4->getValue();
   geo->wallnr5 = p_wallnr5->getValue();

   // get wall-nr from control panel
   geo->pressnr1 = p_pressnr1->getValue();
   geo->pressnr2 = p_pressnr2->getValue();
   geo->pressnr3 = p_pressnr3->getValue();
   geo->pressnr4 = p_pressnr4->getValue();
   geo->pressnr5 = p_pressnr5->getValue();

   // get bila-nrs for boco object from control panel
   geo->bcinnr = p_bila_in->getValue();
   geo->bcoutnr = p_bila_out->getValue();
   geo->bcperiodicnr1 = p_periodic_1->getValue();
   geo->bcperiodicnr2 =  p_periodic_2->getValue();

   // ++++++++++++++++++++++++++++++
   // Read Geo-, Sim- and RB-File
   // ++++++++++++++++++++++++++++++
   // Read data from geofile, simfile and bc file

   if (ReadGeoSimRB(geo)==FAIL)
   {
      return FAIL;
   }

   // ++++++++++++++++++++++++++++++
   // fill covise data structures
   // ++++++++++++++++++++++++++++++
   // Transfer data from struct geometry to covise data arrays and set ports

   if (Data2Covise(geo, velocity, press, k, eps)==1)
   {
      return FAIL;
   }

   // ++++++++++++++++++++++++++++++
   // create boundary conditions object
   // ++++++++++++++++++++++++++++++
   if ( (geo->is3d) && (geo->create_boco_object) )
   {

      coDistributedObject *partObj[9];
      const char *basename = port_boco->getObjName();

      if (CreateBocoObject(partObj, geo, basename)==FAIL)
      {
         return(STOP_PIPELINE);
      }

      assert(partObj[8] == NULL);
      coDoSet *set = new coDoSet((char*)basename,(coDistributedObject **)partObj);
      for (int i = 0; i < 9; i++)
         delete partObj[i];

      port_boco->setCurrentObject(set);
   }

   return SUCCESS;
}


void ReadIHS2::Check_geo_sim_rb(struct geometry *geo)
{
   ifstream fin;

   // check geo
   fin.open(geo->geofile);
   if(fin.good() == false)
   {
      sendError("Kann geofile nicht lesen!\n");
   }
   else
   {
      geo->isgeo=1;
      fin.close();
   }

   //check rb
   fin.open(geo->rbfile);
   if(fin.good() == false)
   {
      sendError("Kann rbfile nicht lesen!\n");
   }
   else
   {
      geo->isrb=1;
      fin.close();
   }

   //check sim
   if (/*(geo->simfile != 0) && */ p_readsim->getValue()==1 )
   {
      fin.open(geo->simfile);
      if(fin.good() == false) {}
      else
      {
         geo->issim=1;
         fin.close();
      }
   }

}

int ReadIHS2::ReadGeoSimRB(struct geometry *geo)
{
   FILE *stream;
   char datei[300];

   char buf[200], errbuf[200];

   int i, n, z, dummy, n_elem, len;
   int dimension = 0;

   bool comment;

   float value;
   char *name = new char[10];

   geo->new_rbfile = 0;                           // new (=1) or old (=0) rb-file format

   geo->n_special_mark = 0;
   geo->n_special_wall = 0;

   geo->n_bcin = 0;
   geo->n_bcout = 0;
   geo->n_bcperiodic1 = 0;
   geo->n_bcperiodic2 = 0;

   // +++++++++++++++++++++++++++++++
   // read geofile
   // +++++++++++++++++++++++++++++++

   strcpy(datei, geo->geofile);
   if( (stream = fopen( &datei[0], "r" )) == NULL )
   {
	   sprintf(errbuf,"Could not open %s\n",datei);
	   sendError("%s",errbuf);
	   return FAIL ;
   }

   // read comments and check for dimension
   comment=true;
   i=0;
   while (comment)
   {
      if (fgets(buf,200,stream)!=buf)
      {
		  sprintf(errbuf,"fgets failed for %s\n",datei);
		  sendError("%s",errbuf);
		  return FAIL ;
      }
	  len=(int)strlen(buf);
	  for(int j=0; j < len; j++)
             buf[j] = tolower(buf[j]);
	  if (strstr(buf,"#")) {
		  if (strstr(buf,"dimension")) {
			  char *buf2=strrchr(buf,':');
			  buf2++;
			  dimension=atoi(buf2);
		  }
	  }
	  else {
		  comment=false;
		  break;
	  }
	  i++;
	  if (i > 100) {
		  sprintf(errbuf,"corrupted file %s! To many comment lines!\n",datei);
		  return FAIL;
	  }
   }
   
   // check dimension
   if (dimension!=2 && dimension!=3) {
	   dimension=p_dimension->getValue();
   }
//   fprintf(stderr,"dimension=%d\n",dimension);
   if (dimension!=2 && dimension!=3) {
	   sprintf(errbuf,"Failed to determine dimension from %s\n and ctrl panel\n",datei);
	   sendError("%s",errbuf);
	   return FAIL;
   }
   geo->is3d=dimension-2;  // 0 for 2D, 1 for 3D

   // read header
   sscanf(buf, "%d%d%d%d%d%d%d%d", &geo->n_nodes, &n_elem, &dummy, &dummy, 
		  &dummy, &dummy, &geo->knmaxnr, &geo->elmaxnr );

   fprintf(stderr,"\nNodes              : %6d\n", geo->n_nodes);
   fprintf(stderr,"Elements           : %6d\n", n_elem);

   if (dimension==3)
   {
	   geo->n_elem_3d = n_elem;
   }
   else if (dimension==2)
   {
	   geo->n_elem_2d = n_elem;
   }
   else {
	   sprintf(errbuf,"invalid dimension=%d !\n",dimension);
	   return FAIL;
   }

   // allocate dynamic arrays
   geo->x = new float[geo->knmaxnr];
   geo->y = new float[geo->knmaxnr];

   if (geo->is3d)
   {
      geo->z = new float[geo->knmaxnr];
      geo->elem_3d = new int[8*geo->n_elem_3d];
   }
   else
   {
      geo->elem_2d = new int[4*geo->n_elem_2d];
   }

   //Koordinaten einlesen aus Datei
   if (geo->is3d==1)
   {
      for (i=0; i<geo->n_nodes; i++)
      {
         if (fgets(buf,200,stream)!=buf)
         {
            fprintf(stderr,"fgets_4 failed in ReadIHS2.cpp");
			return FAIL;
         }
		 sscanf(buf, "%d", &n);
         sscanf(buf, "%d%f%f%f", &dummy, &geo->x[n-1],
				&geo->y[n-1], &geo->z[n-1]);
      }
   }
   else
   {
      for (i=0; i<geo->n_nodes; i++)
      {
         if (fgets(buf,200,stream)!=buf)
         {
            fprintf(stderr,"fgets_5 failed in ReadIHS2.cpp");
			return FAIL;
         }
      }
	  sscanf(buf, "%d", &n);
	  sscanf(buf, "%d%f%f", &dummy, &geo->x[n-1], &geo->y[n-1]);
   }

   // scaling
   if (geo->scalingfactor != 1.)
   {
	   if (geo->is3d) {
		   for (i=0; i<geo->knmaxnr; i++)
		   {
			   geo->x[i] *= geo->scalingfactor;
			   geo->y[i] *= geo->scalingfactor;
			   geo->z[i] *= geo->scalingfactor;
		   }
	   }
	   else {
		   for (i=0; i<geo->knmaxnr; i++)
		   {
			   geo->x[i] *= geo->scalingfactor;
			   geo->y[i] *= geo->scalingfactor;
		   }
	   }
   }

   //DEBUG Ausgabe Knoten 3D stdout
   /*	for (i=0; i<geo->n_nodes; i++)
      {
      if ( (i==1) || (i==geo->n_nodes) )
      {
      printf("%6d. %10.6lf %10.6lf %10.6lf \n", i, geo->x[i], geo->y[i], geo->z[i]);
      }
      }
    */

   //Elemente einlesen aus Datei
   if (geo->is3d==1)
   {
      for (i=0; i<geo->n_elem_3d; i++)
      {
         if (geo->numbered==1)
         {
            if (fgets(buf,200,stream)!=buf)
            {
				if (i < geo->n_elem_3d-1) {
					sprintf(errbuf,"Unexpected eof in %s\n",datei);
					sendError("%s",errbuf);
					return FAIL;
				}
            }
            sscanf(buf, "%d%d%d%d%d%d%d%d%d%d", &dummy, 
				   &geo->elem_3d[8*i], &geo->elem_3d[8*i+1],
				   &geo->elem_3d[8*i+2], &geo->elem_3d[8*i+3],
				   &geo->elem_3d[8*i+4], &geo->elem_3d[8*i+5],
				   &geo->elem_3d[8*i+6], &geo->elem_3d[8*i+7], &dummy);
            if ( geo->elem_3d[8*i+7] < 1 )
            {
               sendError("error reading connectivity list! try to uncheck param numbered conn.-list!\n");
               return FAIL;
            }
         }
         else
         {
			 if (fgets(buf,200,stream)!=buf)
            {
				sendError("fgets_7 failed in ReadIHS2.cpp");
				return FAIL;
            }
			 sscanf(buf, "%d%d%d%d%d%d%d%d", 
					&geo->elem_3d[8*i], &geo->elem_3d[8*i+1],
                  &geo->elem_3d[8*i+2], &geo->elem_3d[8*i+3], 
					&geo->elem_3d[8*i+4], &geo->elem_3d[8*i+5],
					&geo->elem_3d[8*i+6], &geo->elem_3d[8*i+7]);
         }
      }
   }
   else
   {
	   for (i=0; i<geo->n_elem_2d; i++)
	   {
		  if (geo->numbered==1)
		  {
			  sscanf(buf, "%d%d%d%d%d%d", &dummy,
					 &geo->elem_2d[4*i], &geo->elem_2d[4*i+1],
					 &geo->elem_2d[4*i+2], &geo->elem_2d[4*i+3], &dummy);
			  if (fgets(buf,200,stream)!=buf)
			  {
				  sendError("fgets_8 failed in ReadIHS2.cpp");
				  return FAIL;
			  }
         }
         else
         {
			 sscanf(buf, "%d%d%d%d", &geo->elem_2d[4*i], &geo->elem_2d[4*i+1],
					&geo->elem_2d[4*i+2], &geo->elem_2d[4*i+3]);
			 if (fgets(buf,200,stream)!=buf)
			 {
				sendError("fgets_9 failed in ReadIHS2.cpp");
				return FAIL;
			 }
         }
	   }
   }

   // switch from fortran- to c-numbering: -1
   if (geo->is3d==1)
   {
	   for (i=0; i<geo->n_elem_3d; i++)
	   {
		   geo->elem_3d[8*i+0]--;
		   geo->elem_3d[8*i+1]--;
		   geo->elem_3d[8*i+2]--;
		   geo->elem_3d[8*i+3]--;
		   geo->elem_3d[8*i+4]--;
		   geo->elem_3d[8*i+5]--;
		   geo->elem_3d[8*i+6]--;
		   geo->elem_3d[8*i+7]--;
	   }
   }
   else
   {
	   for (i=0; i<geo->n_elem_2d; i++)
	   {
		   geo->elem_2d[4*i+0]--;
		   geo->elem_2d[4*i+1]--;
		   geo->elem_2d[4*i+2]--;
		   geo->elem_2d[4*i+3]--;
	   }
   }

#ifdef DEBUG
   //ausgabe geo->elem_3d stdout (c-notation, same as in file!)
   fprintf(stderr,"\nAusgabe erstes und letztes Element (DEBUG)\n");
   if (geo->is3d==1)
   {
	   i=0;
	   fprintf(stderr,"%6d. %6d %6d %6d %6d %6d %6d %6d %6d \n", i+1,
			   geo->elem_3d[8*i]+1, geo->elem_3d[8*i+1]+1,
			   geo->elem_3d[8*i+2]+1, geo->elem_3d[8*i+3]+1,
			   geo->elem_3d[8*i+4]+1, geo->elem_3d[8*i+5]+1,
			   geo->elem_3d[8*i+6]+1, geo->elem_3d[8*i+7]+1);
	   i=geo->n_elem_3d-1;
	   fprintf(stderr,"%6d. %6d %6d %6d %6d %6d %6d %6d %6d \n", i+1,
			   geo->elem_3d[8*i]+1, geo->elem_3d[8*i+1]+1,
			   geo->elem_3d[8*i+2]+1, geo->elem_3d[8*i+3]+1,
			   geo->elem_3d[8*i+4]+1, geo->elem_3d[8*i+5]+1,
			   geo->elem_3d[8*i+6]+1, geo->elem_3d[8*i+7]+1);
   }
   else
   {
	   i=0;
	   fprintf(stderr,"%6d. %6d %6d %6d %6d\n", i+1,
			   geo->elem_2d[4*i]+1, geo->elem_2d[4*i+1]+1,
			   geo->elem_2d[4*i+2]+1, geo->elem_2d[4*i+3]+1);
	   i=geo->n_elem_2d-1;
	   fprintf(stderr,"%6d. %6d %6d %6d %6d\n", i+1,
			   geo->elem_2d[4*i]+1, geo->elem_2d[4*i+1]+1,
			   geo->elem_2d[4*i+2]+1, geo->elem_2d[4*i+3]+1);
   }
#endif

   fclose(stream);

   printf("\ngeofile gelesen!\n\n");
   // end read geofile

   // +++++++++++++++++++++++++++++++
   // read rb-File
   // +++++++++++++++++++++++++++++++

   strcpy(datei, geo->rbfile);
   if( (stream = fopen( &datei[0], "r" )) == NULL )
   {
      fprintf(stderr, "Kann File %s nicht lesen!!!\n", geo->rbfile);
      return FAIL;
   }

   else
   {

      int n_values;
      int j;

      for (i=0; i<10; i++)                        //Kommentarzeilen
      {
         if (fgets(buf,200,stream)!=buf)
         {
            sendError("fgets_10 failed in ReadIHS2.cpp");
         }
      }

      // Zeigerzeile:
      // 1. Eintritts_RB
      // 2. Waende
      // 3. Druck-RB
      // 4. Symmetrierandbedingungen
      // 5. ?
      // 6. ?
      // 7. Elementmarkierungen
      // 8. Knotenmarkierungen

      if (fgets(buf,200,stream)!=buf)
      {
         sendError("fgets_11 failed in ReadIHS2.cpp");
      }
      n_values = sscanf(buf, "%d%d%d%d%d%d%d%d",&geo->n_in_rb,
            &geo->n_wall,
            &geo->n_press_rb,
            &geo->n_syme,
            &z,
            &z,
            &geo->n_elem_mark,
            &geo->n_kmark);

      printf("Elementmarkierungen : %6d\n", geo->n_elem_mark);
      printf("Wandelemente        : %6d\n", geo->n_wall);
      printf("Eintr.-RB           : %6d\n", geo->n_in_rb);
      printf("Druck.-RB           : %6d\n\n", geo->n_press_rb);

      if (geo->is3d==1)
      {
         if (n_values > 7)
            printf("Knotenmarkierungen  : %6d\n\n", geo->n_kmark);
      }

      if (fgets(buf,200,stream)!=buf)
      {
         sendError("fgets_12 failed in ReadIHS2.cpp");
      }

      //Ein_rb: bisher nur fuer boco_object gebraucht!
      if (geo->create_boco_object)
      {
         geo->dirichlet_nodes = new int[geo->n_in_rb];
         geo->dirichlet_values = new float[5*geo->n_in_rb];
      }

      if (geo->n_in_rb > 0)
      {
         for (i=0; i<geo->n_in_rb/5; i++)         // fuer jede ein_rb 5 Zeilen!
         {
            if (geo->create_boco_object)
            {
               sscanf(buf,"%d%d%f", &dummy, &dummy, &geo->dirichlet_values[5*i]);
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_13 failed in ReadIHS2.cpp");
               }
               sscanf(buf,"%d%d%f", &dummy, &dummy, &geo->dirichlet_values[5*i+1]);
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_14 failed in ReadIHS2.cpp");
               }
               sscanf(buf,"%d%d%f", &dummy, &dummy, &geo->dirichlet_values[5*i+2]);
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_15 failed in ReadIHS2.cpp");
               }
               sscanf(buf,"%d%d%f", &dummy, &dummy, &geo->dirichlet_values[5*i+3]);
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_16 failed in ReadIHS2.cpp");
               }
               sscanf(buf,"%d%d%f", &geo->dirichlet_nodes[i], &dummy, &geo->dirichlet_values[5*i+4]);
               // switch to c-notation
               geo->dirichlet_nodes[i]--;
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_17 failed in ReadIHS2.cpp");
               }
            }
            else
            {
               for (j=0; j<5; j++)                // jump over!
               {
                  if (fgets(buf,200,stream)!=buf)
                  {
                     sendError("fgets_18 failed in ReadIHS2.cpp");
                  }
               }
            }

         }
      }

      // check if we have an old or a new rb-file format
      for (i=0; i<strlen(buf); i++)
      {
         if ( isalpha (buf[i]) )
         {
            geo->new_rbfile = 1;
            printf("Neues rb-Fileformat!\n");
            break;
         }
      }
      if (geo->new_rbfile == 0)
      {
         printf("Altes rb-Fileformat!\n");
      }
      else
      {
         // groesste Zahl annehmen, da unklar -> hinterher anpassen!
         // so ein quatsch!
		 //geo->n_press_rb=geo->n_elem_mark;	 
         //geo->n_wall=geo->n_elem_mark;
      }

      if (geo->is3d==1)
      {
         if (geo->new_rbfile == 0)
		 {
		 	geo->press_rb = new int[5*geo->n_press_rb];
         	//geo->press_rb_value = new float[geo->n_press_rb];
		 }
		 else
		 {
		 	geo->press_rb = new int[6*geo->n_press_rb];
         	//geo->press_rb_value = new float[geo->n_press_rb];
		 }		 
         // Knoten 1-4, Vol.-Element, Markierung
         geo->elem_mark = new int[6*geo->n_elem_mark];
         for (i=0;i<6*geo->n_elem_mark;i++)
         {
            geo->elem_mark[i]=-1;
         }

         if (geo->new_rbfile==0)
         {
            geo->wall = new int[5*geo->n_wall];   // Knoten 1-4, Vol.-Element
         }
         else
         {
            geo->wall = new int[6*geo->n_wall];   // Knoten 1-4, Vol.-Element, Nummer
         }
      }
      else
      {
         // Knoten 1-4, Vol.-Element, Markierung
         geo->elem_mark = new int[4*geo->n_elem_mark];
         for (i=0;i<4*geo->n_elem_mark;i++)
         {
            geo->elem_mark[i]=-1;
         }

         geo->wall = new int[3*geo->n_wall];      // Knoten 1+2, Fl.-Element
      }

      // reibungsfreie Waende einlesen
      // TODO

      if (geo->is3d==1)
      {

         // old rbfile-format

         if(geo->new_rbfile==0)
         {
            // read walls
            for (i=0; i<geo->n_wall; i++)
            {
               sscanf(buf, "%d%d%d%d%d%d%d%d",
                     &geo->wall[5*i], &geo->wall[5*i+1], &geo->wall[5*i+2], &geo->wall[5*i+3], &z, &z, &z, &geo->wall[5*i+4]);

               // switch to c-notation
               geo->wall[5*i+0]--;
               geo->wall[5*i+1]--;
               geo->wall[5*i+2]--;
               geo->wall[5*i+3]--;
               geo->wall[5*i+4]--;

               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_19 failed in ReadIHS2.cpp");
               }
            }

            // read press-rb
            for (i=0; i<geo->n_press_rb; i++)
            {
               sscanf(buf, "%d%d%d%d%d%f%d",
                     &geo->press_rb[5*i], &geo->press_rb[5*i+1], &geo->press_rb[5*i+2], &geo->press_rb[5*i+3],
                     &z, &geo->press_rb_value[i], &geo->press_rb[5*i+4]);

               // switch to c-notation
               geo->press_rb[5*i+0]--;
               geo->press_rb[5*i+1]--;
               geo->press_rb[5*i+2]--;
               geo->press_rb[5*i+3]--;
               geo->press_rb[5*i+4]--;

               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_20 failed in ReadIHS2.cpp");
               }
            }

            // jump over symmetric-rb
            for (i=0; i<geo->n_syme; i++)
            {
               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_21 failed in ReadIHS2.cpp");
               }
            }

            // read bila
            int bilanr;
            int last_bilanr = -1;
            geo->bilanrlist = (int *)calloc(100, sizeof(int));
			geo->bilanames = new char *[50];
            geo->n_bilanrs = 0;

            for (i=0; i<geo->n_elem_mark; i++)
            {
               sscanf(buf, "%d%d%d%d%d%d", &geo->elem_mark[6*i], &geo->elem_mark[6*i+1],
                     &geo->elem_mark[6*i+2], &geo->elem_mark[6*i+3], &geo->elem_mark[6*i+4], &geo->elem_mark[6*i+5]);

               // switch to c-notation
               geo->elem_mark[6*i+0]--;
               geo->elem_mark[6*i+1]--;
               geo->elem_mark[6*i+2]--;
               geo->elem_mark[6*i+3]--;
               geo->elem_mark[6*i+4]--;

               bilanr = geo->elem_mark[6*i+5];

               if (bilanr == geo->bilanr1) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr2) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr3) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr4) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr5) {geo->n_special_mark++;}
               if (bilanr == geo->bcinnr) {geo->n_bcin++;}
               if (bilanr == geo->bcoutnr) {geo->n_bcout++;}
               if (bilanr == geo->bcperiodicnr1)
               {
                  geo->n_bcperiodic1++;
               }
               if (bilanr == geo->bcperiodicnr2)
               {
                  geo->n_bcperiodic2++;
               }

               if (bilanr != last_bilanr)
               {
                  if (checkfornewnr (geo->bilanrlist, &(geo->n_bilanrs), geo->bilanames, bilanr, last_bilanr, name)==1)
                  {
                     return FAIL;
                  }
               }

               countnrs (geo->n_bilanrs, geo->bilanrlist, bilanr);

               last_bilanr = bilanr;

               if (fgets(buf,200,stream)!=buf)
               {
                  sendError("fgets_22 failed in ReadIHS2.cpp");
               }
            }
         }

         // new rbfile-format
         int nwalls=0;
         int npress=0;
         if(geo->new_rbfile==1)
         {

            int number[6];
            int iswall=0;
            int ismark=0;
            int ispress=0;

            int bilanr, wallnr, presnr;
            int last_bilanr = -1, last_wallnr = -1, last_presnr = -1;
            geo->bilanrlist = (int *)calloc(100, sizeof(int));
            geo->wallnrlist = (int *)calloc(100, sizeof(int));
            geo->presnrlist = (int *)calloc(100, sizeof(int));
			geo->bilanames = new char *[50];
			geo->wallnames = new char *[50];
			geo->presnames = new char *[50];
            geo->n_bilanrs = 0;
            geo->n_wallnrs = 0;
            geo->n_presnrs = 0;

            // read walls
            for (i=0; i<geo->n_wall; i++)
            {
               n_values = sscanf(buf, "%d%d%d%d%d%d%f%s", &number[0], &number[1],
                     &number[2], &number[3], &number[4], &number[5], &value, name);

               for (j=0;j<5;j++)
               {
                  number[j]--;
               }

               wallnr = number[5];
//if ((i%100)==0)
//{
//   cerr << buf;
//   cerr << i << ". wallnr = " << wallnr << endl;
//}
               if (  (number[5]==geo->wallnr1) ||
                     (number[5]==geo->wallnr2) ||
                     (number[5]==geo->wallnr3) ||
                     (number[5]==geo->wallnr4) ||
                     (number[5]==geo->wallnr5)   )
                  iswall=1;
               if (p_showallwalls->getValue()==1)
                  iswall=1;
               // fuer Online-Sim werden alle Waende gebraucht (-> distboco-Objekt!)
               if (p_create_boco_obj->getValue()==1)
                  iswall=1;

               if (iswall==1)
               {
                  geo->wall[6*nwalls+0]=number[0];
                  geo->wall[6*nwalls+1]=number[1];
                  geo->wall[6*nwalls+2]=number[2];
                  geo->wall[6*nwalls+3]=number[3];
                  geo->wall[6*nwalls+4]=number[4];
                  geo->wall[6*nwalls+5]=number[5];
                  nwalls++;
               }

               if (wallnr != last_wallnr)
               {
//cerr << "wallnr = " << wallnr << ", last_wallnr = " << last_wallnr << endl;
                  if (checkfornewnr (geo->wallnrlist, &geo->n_wallnrs, geo->wallnames, wallnr, last_wallnr, name)==1)
                  {
                     return FAIL;
                  }
               }

               countnrs (geo->n_wallnrs, geo->wallnrlist, wallnr);

               last_wallnr = wallnr;

               if(fgets(buf,200,stream) == NULL)
                  fprintf(stderr,"fgets failed in ReadNIHS2.cpp");
            }

            // read pressure bcs
            for (i=0; i<geo->n_press_rb; i++)
            {
               n_values = sscanf(buf, "%d%d%d%d%d%d%f%s", &number[0], &number[1],
                     &number[2], &number[3], &number[4], &number[5], &value, name);

               for (j=0;j<5;j++)
               {
                  number[j]--;
               }

               presnr = number[5];

               if (  (number[5]==geo->pressnr1) ||
                     (number[5]==geo->pressnr2) ||
                     (number[5]==geo->pressnr3) ||
                     (number[5]==geo->pressnr4) ||
                     (number[5]==geo->pressnr5)   )
                  ispress=1;

               if (ispress==1)
               {
                  geo->press_rb[6*npress+0]=number[0]; // node1
                  geo->press_rb[6*npress+1]=number[1]; // node2
                  geo->press_rb[6*npress+2]=number[2]; // node3
                  geo->press_rb[6*npress+3]=number[3]; // node4
                  geo->press_rb[6*npress+4]=number[4]; // volume element
                  geo->press_rb[6*npress+5]=number[5]; // bilanr
                  npress++;
               }

               if (presnr != last_presnr)
               {
                  if (checkfornewnr (geo->presnrlist, &geo->n_presnrs, geo->presnames, presnr, last_presnr, name)==1)
                  {
                     return FAIL;
                  }
               }

               countnrs (geo->n_presnrs, geo->presnrlist, presnr);

               last_presnr = presnr;
			   
               if(fgets(buf,200,stream) == NULL)
                  fprintf(stderr,"fgets failed in ReadNIHS2.cpp");
            }

            // read bilas
            for (i=0; i<geo->n_elem_mark; i++)
            {
               iswall=0;
               ismark=0;
               n_values = sscanf(buf, "%d%d%d%d%d%d%f%s", &number[0], &number[1],
                     &number[2], &number[3], &number[4], &number[5], &value, name);

               for (j=0;j<5;j++)
               {
                  number[j]--;
               }

               bilanr = number[5];

               if (bilanr == geo->bilanr1) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr2) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr3) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr4) {geo->n_special_mark++;}
               if (bilanr == geo->bilanr5) {geo->n_special_mark++;}
               if (bilanr == geo->bcinnr) {geo->n_bcin++;}
               if (bilanr == geo->bcoutnr) {geo->n_bcout++;}
               if (bilanr == geo->bcperiodicnr1) {geo->n_bcperiodic1++;}
               if (bilanr == geo->bcperiodicnr2) {geo->n_bcperiodic2++;}
 
               if (     (number[5]==geo->bilanr1) ||
                     (number[5]==geo->bilanr2) ||
                     (number[5]==geo->bilanr3) ||
                     (number[5]==geo->bilanr4) ||
                     (number[5]==geo->bilanr5)   )
                  ismark=1;
               if (p_showallbilas->getValue()==1)
                  ismark=1;
               // fuer Online-Sim werden alle bilas gebraucht (-> distboco-Objekt!)
               if (p_create_boco_obj->getValue()==1)
                  ismark=1;

				// Elemente nur nehmen, wenn sie Markierungen sind!
               if (ismark==1)
               {
                  geo->elem_mark[6*i+0]=number[0];
                  geo->elem_mark[6*i+1]=number[1];
                  geo->elem_mark[6*i+2]=number[2];
                  geo->elem_mark[6*i+3]=number[3];
                  geo->elem_mark[6*i+4]=number[4];
                  geo->elem_mark[6*i+5]=number[5];
			   }
			   
               if (bilanr != last_bilanr)
               {
                  if (checkfornewnr (geo->bilanrlist, &geo->n_bilanrs, geo->bilanames, bilanr, last_bilanr, name)==1)
                  {
                     return FAIL;
                  }
               }

	       countnrs (geo->n_bilanrs, geo->bilanrlist, bilanr);

               last_bilanr = bilanr;

               if(fgets(buf,200,stream) == NULL)
                  fprintf(stderr,"fgets failed in ReadNIHS2.cpp");
            }
         }

         geo->n_press_rb=npress;
         geo->n_wall=nwalls;
/*		 
cerr << "geo->n_bcin = " << geo->n_bcin << endl;
cerr << "geo->n_bcout = " << geo->n_bcout << endl;
cerr << "geo->n_bcperiodic1 = " << geo->n_bcperiodic1 << endl;
cerr << "geo->n_bcperiodic2 = " << geo->n_bcperiodic2 << endl;
*/

      }
      else                                        // 2D-geometry
      {
         // read wall
         for (i=0; i<geo->n_wall; i++)
         {
            sscanf(buf, "%d%d%d%d%d%d",
                  &geo->wall[3*i], &geo->wall[3*i+1], &z, &z, &z, &geo->wall[3*i+2]);

            // switch to c-notation
            geo->wall[3*i+0]--;
            geo->wall[3*i+1]--;
            geo->wall[3*i+2]--;

            if (fgets(buf,200,stream)!=buf)
            {
               sendError("fgets_24 failed in ReadIHS2.cpp");
            }
         }

         // read / jump over press-rb
         for (i=0; i<geo->n_press_rb; i++)
         {
            if (fgets(buf,200,stream)!=buf)
            {
               sendError("fgets_25 failed in ReadIHS2.cpp");
            }
         }

         // read bila
         for (i=0; i<geo->n_elem_mark; i++)
         {
            sscanf(buf, "%d%d%d%d", &geo->elem_mark[4*i], &geo->elem_mark[4*i+1],
                  &geo->elem_mark[4*i+2], &geo->elem_mark[4*i+3]);

            // switch to c-notation
            geo->elem_mark[4*i+0]--;
            geo->elem_mark[4*i+1]--;
            geo->elem_mark[4*i+2]--;

            if ( (geo->elem_mark[4*i+3]==geo->bilanr1)
                  || (geo->elem_mark[4*i+3]==geo->bilanr2)
                  || (geo->elem_mark[4*i+3]==geo->bilanr3)
                  || (geo->elem_mark[4*i+3]==geo->bilanr4)
                  || (geo->elem_mark[4*i+3]==geo->bilanr5) )
            {geo->n_special_mark++;}
            if (fgets(buf,200,stream)!=buf)
            {
               sendError("fgets_26 failed in ReadIHS2.cpp");
            }
         }
      }

      fclose(stream);

      // print wallnrs existing and their quantities in file
      fprintf(stderr,"\nVorkommende Waende, Anzahl\n");
      int gesamt = 0;
	  //cerr << "geo->n_wallnrs: " << geo->n_wallnrs << endl;
      for (i=0; i<geo->n_wallnrs; i++)
      {
         fprintf(stderr, "%d: %d\n", geo->wallnrlist[2*i], geo->wallnrlist[2*i+1]/*, geo->wallnames[i]*/);
         gesamt += geo->wallnrlist[2*i+1];
      }
      fprintf(stderr,"Gesamt: %d\n", gesamt);

      // print presnrs existing and their quantities in file
      fprintf(stderr,"\nVorkommende Druckrandbedingungen, Anzahl\n");
      gesamt = 0;
	  //cerr << "geo->n_presnrs: " << geo->n_presnrs << endl;
      for (i=0; i<geo->n_presnrs; i++)
      {
         fprintf(stderr, "%d: %d\n", geo->presnrlist[2*i], geo->presnrlist[2*i+1]/*, geo->presnames[i]*/);
         gesamt += geo->presnrlist[2*i+1];
      }
      fprintf(stderr,"Gesamt: %d\n", gesamt);

      // print bilanrs existing and their quantities in file
      fprintf(stderr,"\nVorkommende Bilas, Anzahl\n");
      gesamt = 0;
	  //cerr << "geo->n_bilanrs: " << geo->n_bilanrs << endl;
      for (i=0; i<geo->n_bilanrs; i++)
      {
         fprintf(stderr, "%d: %d\n", geo->bilanrlist[2*i], geo->bilanrlist[2*i+1]/*, geo->bilanames[i]*/);
         gesamt += geo->bilanrlist[2*i+1];
      }
      fprintf(stderr,"Gesamt: %d\n", gesamt);

      printf("\nrbfile gelesen!\n\n");

   }

   // +++++++++++++++++++++++++++++++
   // read simfile
   // +++++++++++++++++++++++++++++++

   if (geo->issim)
   {
	   int n_rnod;
	   int n_relm;
	   int n_rnod_inst;
	   int n_col[3];


      geo->u   = new float[geo->knmaxnr];
      geo->v   = new float[geo->knmaxnr];
      geo->w   = new float[geo->knmaxnr];
      geo->k   = new float[geo->knmaxnr];
      geo->eps = new float[geo->knmaxnr];
      geo->p   = new float[geo->knmaxnr];

      if (geo->is3d == 1)
      {
         geo->p_elem = new float[geo->n_elem_3d+1];
      }
      else
      {
         geo->p_elem = new float[geo->n_elem_2d+1];
      }

      strcpy(datei, geo->simfile);
      if( (stream = fopen( &datei[0], "r" )) == NULL )
      {
         fprintf(stderr, "Kann File %s nicht lesen!!!\n", geo->simfile);
         return FAIL;
      }

      do
      {
         if (fgets(buf,200,stream)!=buf)
         {
            sendError("fgets_27 failed in ReadIHS2.cpp");
         }
      }
      while (buf[0]=='#');                        // so lange lesen, bis erstes Zeichen kein Kommentarzeichen mehr!
	  sscanf(buf,"%d%d%d%d%d%d",&n_rnod,&n_relm,&n_rnod_inst,
			 &n_col[0],&n_col[1],&n_col[2]);

	  if (n_rnod != geo->n_nodes) {
		  sendError("Numbers in result and geometry file do not match!");
	  }
      if (fgets(buf,200,stream)!=buf)
      {
         sendError("fgets_28 failed in ReadIHS2.cpp");
      }                      // Erste Datenzeile lesen

      //Einlesen der Werte aus Datei
      for (i=0; i<n_rnod; i++)
      {
         sscanf(buf, "%d", &n);
         sscanf(buf, "%d%f%f%f%f%f%f", &z, &geo->u[n-1], &geo->v[n-1], &geo->w[n-1], &geo->k[n-1], &geo->eps[n-1], &geo->p[n-1]);

         if (fgets(buf,200,stream)!=buf)
         {
            sendError("Could not read nodal results");
         }
      }

      if (p_abs2rel->getValue())
      {
         float omega = float(M_PI * p_n->getValue() / 30.);
         int rotaxis = p_RotAxis->getValue();

         if (!strcmp(s_RotAxis[rotaxis], "x"))
         {
            for (i = 0; i < geo->knmaxnr; i++)
            {
               geo->v[i] = geo->v[i] - omega * geo->z[i];
               geo->w[i] = geo->w[i] + omega * geo->y[i];
            }
         }
         if (!strcmp(s_RotAxis[rotaxis], "y"))
         {
            for (i = 0; i < geo->knmaxnr; i++)
            {
               geo->u[i] = geo->u[i] + omega * geo->z[i];
               geo->w[i] = geo->w[i] - omega * geo->x[i];
            }
         }
         if (!strcmp(s_RotAxis[rotaxis], "z"))
         {
            for (i = 0; i < geo->knmaxnr; i++)
            {
               geo->u[i] = geo->u[i] + omega * geo->y[i];
               geo->v[i] = geo->v[i] - omega * geo->x[i];
            }
         }
      }

      for (i=0; i<n_relm; i++)
      {
         sscanf(buf, "%d%f", &z, &geo->p_elem[i]);
         if ((fgets(buf,200,stream)!=buf) && (i < geo->n_elem_3d-1))
         {
	     char buf[200];
	     sprintf(buf," unexpected end of erg-file\n src: %s (%d)\n",
		     __FILE__,__LINE__);
            sendError("%s",buf);
         }
      }

      /*
      //DEBUG
      //Ausgabe der Werte
      printf("Werte im Gitter:\n");
      printf("    Nr. |          u          v          w          k        eps          p\n");
      printf(" ----------------------------------------------------------------------------\n");
      for (i=1; i<geo->n_nodes+1; i++)
      {
      if (i==1)
      {
      printf("%6d. | %10.4lf %10.4lf %10.4lf %10.4lf %10.4lf %10.4lf \n", i, geo->u[i], geo->v[i], geo->w[i],
      geo->k[i], geo->eps[i], geo->p[i]);
      }
      }
       */

      printf("simfile gelesen!\n\n");

      fclose(stream);

   }

   return(0);

}


int ReadIHS2::Data2Covise(struct geometry *geo, coDoVec3 *velocity, coDoFloat *press,
      coDoFloat *k, coDoFloat *eps)
{

   int i;
   char buf[100];
   char buf2[100];

   // ++++++++++++++++++++++++++++++
   // fill grid data
   // ++++++++++++++++++++++++++++++

   coDoUnstructuredGrid *unsGrd =
      // name of USG object
      new coDoUnstructuredGrid(port_grid->getObjName(),
            geo->n_elem_3d,                             // number of elements
            8*geo->n_elem_3d,                           // number of connectivities
            geo->knmaxnr,                               // number of coordinates
            1);                                         // does type list exist?

   int *elem,*conn,*type;
   float *xc,*yc,*zc;

   unsGrd->getAddresses(&elem, &conn, &xc, &yc, &zc);
   unsGrd->getTypeList(&type);

   int n_mark = 0;

   for (i = 0; i < geo->n_elem_3d; i++)
   {
      *elem = 8*i;               elem++;

      *conn = geo->elem_3d[8*i+0];   conn++;
      *conn = geo->elem_3d[8*i+1];   conn++;
      *conn = geo->elem_3d[8*i+2];   conn++;
      *conn = geo->elem_3d[8*i+3];   conn++;
      *conn = geo->elem_3d[8*i+4];   conn++;
      *conn = geo->elem_3d[8*i+5];   conn++;
      *conn = geo->elem_3d[8*i+6];   conn++;
      *conn = geo->elem_3d[8*i+7];   conn++;
      *type = TYPE_HEXAGON;            type++;
   }

   // copy geometry coordinates to unsgrd
   memcpy(xc, geo->x, (geo->knmaxnr)*sizeof(float));
   memcpy(yc, geo->y, (geo->knmaxnr)*sizeof(float));
   memcpy(zc, geo->z, (geo->knmaxnr)*sizeof(float));

   // set out port
   port_grid->setCurrentObject(unsGrd);

   // ++++++++++++++++++++++++++++++
   // fill bila 2D-Elements
   // ++++++++++++++++++++++++++++++

   if ( (p_showallbilas->getValue()==0) && (geo->n_elem_mark > 0) )
   {

      // fill just marked 3D-Elements with specific numbers

      int j=0;
      int *clist, *plist;
      float *xp,*yp,*zp;

      int *pol_list = new int[geo->n_special_mark];

      geo->special_mark = new int[4*geo->n_elem_mark];

      for (i=0; i<geo->n_elem_mark; i++)
      {

         if ( (geo->elem_mark[6*i+5]==geo->bilanr1)
               || (geo->elem_mark[6*i+5]==geo->bilanr2)
               || (geo->elem_mark[6*i+5]==geo->bilanr3)
               || (geo->elem_mark[6*i+5]==geo->bilanr4)
               || (geo->elem_mark[6*i+5]==geo->bilanr5) )
         {
            geo->special_mark[4*j+0]=geo->elem_mark[6*i+0];
            geo->special_mark[4*j+1]=geo->elem_mark[6*i+1];
            geo->special_mark[4*j+2]=geo->elem_mark[6*i+2];
            geo->special_mark[4*j+3]=geo->elem_mark[6*i+3];
            pol_list[j]=4*j;
            j++;
         }
      }

      geo->n_special_mark=j;
      //realloc can be done here to save mem

      // name of Polygon object
      coDoPolygons *bila = new coDoPolygons(port_bila->getObjName(),
            geo->knmaxnr,                            // number of points
            4*geo->n_special_mark,                   // number of corners
            geo->n_special_mark);                    // number of polygons

      bila->getAddresses(&xp, &yp, &zp, &clist, &plist);

      for (i = 0; i < geo->n_special_mark; i++)
      {
         *plist = 4*i; plist++;
         *clist = geo->special_mark[4*i+0]; clist++;
         *clist = geo->special_mark[4*i+1]; clist++;
         *clist = geo->special_mark[4*i+2]; clist++;
         *clist = geo->special_mark[4*i+3]; clist++;
      }

      // copy geometry coordinates to specMark
      memcpy(xp, geo->x, (geo->knmaxnr)*sizeof(float));
      memcpy(yp, geo->y, (geo->knmaxnr)*sizeof(float));
      memcpy(zp, geo->z, (geo->knmaxnr)*sizeof(float));

      sprintf(buf, "Anzahl Flaechenelemente mit Markierung(en)");
      if (geo->bilanr1 != 0)
      {
         n_mark++;
         sprintf(buf2, " %d", geo->bilanr1); strcat(buf, buf2);
      }
      if (geo->bilanr2 != 0)
      {
         n_mark++;
         sprintf(buf2, " %d", geo->bilanr2); strcat(buf, buf2);
      }
      if (geo->bilanr3 != 0)
      {
         n_mark++;
         sprintf(buf2, " %d", geo->bilanr3); strcat(buf, buf2);
      }
      if (geo->bilanr4 != 0)
      {
         n_mark++;
         sprintf(buf2, " %d", geo->bilanr4); strcat(buf, buf2);
      }
      if (geo->bilanr5 != 0)
      {
         n_mark++;
         sprintf(buf2, " %d", geo->bilanr5); strcat(buf, buf2);
      }

      if (n_mark > 0)
      {
         sprintf(buf2, " : %d\n", geo->n_special_mark); strcat(buf, buf2);
         printf("%s", buf);
      }

      bila->addAttribute("vertexOrder","1");
      port_bila->setCurrentObject(bila);
      
	  if (pol_list)
      	delete[] pol_list;
   }

   if ( (p_showallbilas->getValue()==1) && (geo->n_elem_mark>0) )
   {

      // fill all marked 2D-Elements

      int *clist, *plist;
      float *xp,*yp,*zp;

      int *pol_list = new int[geo->n_elem_mark];

      // name of Polygon object
      coDoPolygons *elem_mark = new coDoPolygons(port_bila->getObjName(),
            geo->knmaxnr,                            // number of points
            4*geo->n_elem_mark,                      // number of corners
            geo->n_elem_mark);                       // number of polygons

      elem_mark->getAddresses(&xp, &yp, &zp, &clist, &plist);

      for (i = 0; i < geo->n_elem_mark; i++)
      {
         *plist = 4*i; plist++;
         *clist = geo->elem_mark[6*i+0]; clist++;
         *clist = geo->elem_mark[6*i+1]; clist++;
         *clist = geo->elem_mark[6*i+2]; clist++;
         *clist = geo->elem_mark[6*i+3]; clist++;
      }

      // copy geometry coordinates to elem_mark
      memcpy(xp, geo->x, (geo->knmaxnr)*sizeof(float));
      memcpy(yp, geo->y, (geo->knmaxnr)*sizeof(float));
      memcpy(zp, geo->z, (geo->knmaxnr)*sizeof(float));

      elem_mark->addAttribute("vertexOrder","1");
      port_bila->setCurrentObject(elem_mark);

	  if (pol_list)
         delete[] pol_list;
   }

   // ++++++++++++++++++++++++++++++
   // fill wall
   // ++++++++++++++++++++++++++++++

   // old rb-file or new and all values
   if ( ( (p_showallwalls->getValue()==1) || (geo->new_rbfile == 0) ) && (geo->n_wall>0) )
   {

      // fill all walls

      int *pol_list = new int[geo->n_wall];
      float *xp, *yp, *zp;
      int *clist, *plist;

      int ncolwall;

      // name of Polygon object
      coDoPolygons *wall = new coDoPolygons(port_wall->getObjName(),
            geo->knmaxnr,                            // number of points
            4*geo->n_wall,                           // number of corners
            geo->n_wall);                            // number of polygons

      wall->getAddresses(&xp, &yp, &zp, &clist, &plist);

      if (geo->new_rbfile==0)
      {
         ncolwall=5;
      }
      else if (geo->new_rbfile==1)
      {
         ncolwall=6;
      }
      else
      {
         cerr << "ReadIHS2: using ncolwall uninitialized" << endl;
         ncolwall=0; // XXX: just to keep compiler happy
      }


      for (i=0; i<geo->n_wall; i++)
      {
         *plist = 4*i; plist++;
         *clist = geo->wall[ncolwall*i+0]; clist++;
         *clist = geo->wall[ncolwall*i+1]; clist++;
         *clist = geo->wall[ncolwall*i+2]; clist++;
         *clist = geo->wall[ncolwall*i+3]; clist++;
      }

      // copy geometry coordinates to wall
      memcpy(xp, geo->x, (geo->knmaxnr)*sizeof(float));
      memcpy(yp, geo->y, (geo->knmaxnr)*sizeof(float));
      memcpy(zp, geo->z, (geo->knmaxnr)*sizeof(float));

      wall->addAttribute("vertexOrder","1");
      wall->addAttribute("MATERIAL","metal metal.30");
      port_wall->setCurrentObject(wall);

	  if (pol_list)
         delete[] pol_list;
   }

   else                                           // ( geo->new_rbfile==1 && p_showallwalls->getValue()==0 )
   {
      // just fill marked walls
      if (geo->n_special_wall > 0)
	  {
    	  int *pol_list = new int[geo->n_special_wall];
    	  float *xp, *yp, *zp;
    	  int *clist, *plist;
    	  int j=0;

    	  // name of Polygon object
    	  coDoPolygons *wall = new coDoPolygons(port_wall->getObjName(),
            	geo->knmaxnr,                            // number of points
            	4*geo->n_special_wall,                   // number of corners
            	geo->n_special_wall);                    // number of polygons

    	  wall->getAddresses(&xp, &yp, &zp, &clist, &plist);

    	  geo->special_wall = new int[4*geo->n_special_wall];
    	  for (i=0; i<geo->n_wall; i++)
    	  {
        	 if ( (geo->wall[6*i+5]==geo->wallnr1)
            	   || (geo->wall[6*i+5]==geo->wallnr2)
            	   || (geo->wall[6*i+5]==geo->wallnr3)
            	   || (geo->wall[6*i+5]==geo->wallnr4)
            	   || (geo->wall[6*i+5]==geo->wallnr5) )
        	 {
            	geo->special_wall[4*j+0]=geo->wall[6*i+0];
            	geo->special_wall[4*j+1]=geo->wall[6*i+1];
            	geo->special_wall[4*j+2]=geo->wall[6*i+2];
            	geo->special_wall[4*j+3]=geo->wall[6*i+3];
            	pol_list[j]=4*j;
            	j++;
        	 }
    	  }

    	  for (i = 0; i < geo->n_special_wall; i++)
    	  {
        	 *plist = 4*i; plist++;
        	 *clist = geo->special_wall[4*i+0]; clist++;
        	 *clist = geo->special_wall[4*i+1]; clist++;
        	 *clist = geo->special_wall[4*i+2]; clist++;
        	 *clist = geo->special_wall[4*i+3]; clist++;
    	  }

    	  // copy geometry coordinates to wall
    	  memcpy(xp, geo->x, (geo->knmaxnr)*sizeof(float));
    	  memcpy(yp, geo->y, (geo->knmaxnr)*sizeof(float));
    	  memcpy(zp, geo->z, (geo->knmaxnr)*sizeof(float));

    	  n_mark=0;

    	  sprintf(buf, "Anzahl Wandelemente mit Markierung(en)");
    	  if (geo->wallnr1 != 0)
    	  {
        	 n_mark++;
        	 sprintf(buf2, " %d", geo->wallnr1); strcat(buf, buf2);
    	  }
    	  if (geo->wallnr2 != 0)
    	  {
        	 n_mark++;
        	 sprintf(buf2, " %d", geo->wallnr2); strcat(buf, buf2);
    	  }
    	  if (geo->wallnr3 != 0)
    	  {
        	 n_mark++;
        	 sprintf(buf2, " %d", geo->wallnr3); strcat(buf, buf2);
    	  }
    	  if (geo->wallnr4 != 0)
    	  {
        	 n_mark++;
        	 sprintf(buf2, " %d", geo->wallnr4); strcat(buf, buf2);
    	  }
    	  if (geo->wallnr5 != 0)
    	  {
        	 n_mark++;
        	 sprintf(buf2, " %d", geo->wallnr5); strcat(buf, buf2);
    	  }

    	  if (n_mark > 0)
    	  {
        	 sprintf(buf2, " : %d\n", geo->n_special_wall); strcat(buf, buf2);
        	 printf("%s", buf);
    	  }

    	  wall->addAttribute("vertexOrder","1");
    	  wall->addAttribute("MATERIAL","metal metal.30");
    	  if (wall)
	  		port_wall->setCurrentObject(wall);

		  if (pol_list)
        	 delete[] pol_list;
      }
      if (geo->n_special_wall == 0)
	  {
fprintf(stderr,"geo->n_special_wall = 0!\n");
	  }
   }

   // ++++++++++++++++++++++++++++++
   // fill press_rb
   // ++++++++++++++++++++++++++++++

   if (geo->n_press_rb > 0)
   {

      // fill all press_rb

      int *clist, *plist;
      float *xp,*yp,*zp;

      int *pol_list = new int[geo->n_press_rb];

      // name of Polygon object
      coDoPolygons *press_rb = new coDoPolygons(port_pressrb->getObjName(),
            geo->knmaxnr,                            // number of points
            4*geo->n_press_rb,                       // number of corners
            geo->n_press_rb);                        // number of polygons

      press_rb->getAddresses(&xp, &yp, &zp, &clist, &plist);

      for (i=0; i<geo->n_press_rb; i++)
      {
         *plist = 4*i; plist++;
         *clist = geo->press_rb[6*i+0]; clist++;
         *clist = geo->press_rb[6*i+1]; clist++;
         *clist = geo->press_rb[6*i+2]; clist++;
         *clist = geo->press_rb[6*i+3]; clist++;
      }

      // copy geometry coordinates to elem_mark
      memcpy(xp, geo->x, (geo->knmaxnr)*sizeof(float));
      memcpy(yp, geo->y, (geo->knmaxnr)*sizeof(float));
      memcpy(zp, geo->z, (geo->knmaxnr)*sizeof(float));

      press_rb->addAttribute("vertexOrder","1");
	  if (press_rb)
      	port_pressrb->setCurrentObject(press_rb);

	  if (pol_list)
      	delete[] pol_list;
   }

   if ( strcmp(geo->simfile,"0") )
   {

      // ++++++++++++++++++++++++++++++
      // fill pressure
      // ++++++++++++++++++++++++++++++

      float *p;
	  //p = new float[geo->knmaxnr];

      press = new coDoFloat(port_pressure->getObjName(), geo->knmaxnr);
      press -> getAddress(&p);
      for(i=0; i<geo->knmaxnr; i++)
      {
         p[i] = geo->p[i];
      }
	  if (press)
      	 port_pressure->setCurrentObject(press);

      // ++++++++++++++++++++++++++++++
      // fill velocity
      // ++++++++++++++++++++++++++++++

      float *vx, *vy, *vz;
      /*vx = new float[geo->knmaxnr];
      vy = new float[geo->knmaxnr];
      vz = new float[geo->knmaxnr];*/

      velocity = new coDoVec3(port_velocity->getObjName(), geo->knmaxnr);
      velocity -> getAddresses(&vx, &vy, &vz);
      for(i=0; i<geo->knmaxnr; i++)
      {
         vx[i] = geo->u[i];
         vy[i] = geo->v[i];
         vz[i] = geo->w[i];
      }
	  if (velocity)
      	port_pressure->setCurrentObject(velocity);

      // ++++++++++++++++++++++++++++++
      // fill k
      // ++++++++++++++++++++++++++++++

      float *_k;
      _k = new float[geo->knmaxnr];

      k = new coDoFloat(port_k->getObjName(), geo->knmaxnr);
      k -> getAddress(&_k);
      for(i=0; i<geo->knmaxnr; i++)
      {
         _k[i] = geo->k[i];
      }
	  if (k)
      	port_k->setCurrentObject(k);

      // ++++++++++++++++++++++++++++++
      // fill eps
      // ++++++++++++++++++++++++++++++

      float *e;
      e = new float[geo->n_nodes];

      eps = new coDoFloat(port_eps->getObjName(), geo->knmaxnr);
      eps -> getAddress(&e);
      for(i=0; i<geo->knmaxnr; i++)
      {
         e[i] = geo->eps[i];
      }
	  if (eps)
      	port_pressure->setCurrentObject(eps);

   }

   printf("\n**********************************\n");
   printf("         ReadIHS2 Ende           \n");
   printf("**********************************\n");

   return(0);
}


int ReadIHS2::CreateBocoObject(coDistributedObject **partObj, struct geometry *geo, const char *basename)
{
   int *data;
   float *bPtr;

   char name[256];

   int i;
   int size[2];

   int n_col_wall;

   // alloc memory
   geo->bcin        = new int[6*geo->n_bcin];
   geo->bcout       = new int[6*geo->n_bcout];
   geo->bcperiodic1 = new int[6*geo->n_bcperiodic1];
   geo->bcperiodic2 = new int[6*geo->n_bcperiodic2];

   int bcin_counter        = 0;
   int bcout_counter       = 0;
   int bcperiodic1_counter = 0;
   int bcperiodic2_counter = 0;

   // fill memory from geo->elem_mark
   for (i=0; i<geo->n_elem_mark; i++)
   {
      if (geo->elem_mark[6*i+5] == geo->bcinnr)
      {
         geo->bcin[6*bcin_counter+0] = geo->elem_mark[6*i+0];
         geo->bcin[6*bcin_counter+1] = geo->elem_mark[6*i+1];
         geo->bcin[6*bcin_counter+2] = geo->elem_mark[6*i+2];
         geo->bcin[6*bcin_counter+3] = geo->elem_mark[6*i+3];
         geo->bcin[6*bcin_counter+4] = geo->elem_mark[6*i+4];
         geo->bcin[6*bcin_counter+5] = geo->elem_mark[6*i+5];
         bcin_counter++;
      }
      if (geo->elem_mark[6*i+5] == geo->bcoutnr)
      {
         geo->bcout[6*bcout_counter+0] = geo->elem_mark[6*i+0];
         geo->bcout[6*bcout_counter+1] = geo->elem_mark[6*i+1];
         geo->bcout[6*bcout_counter+2] = geo->elem_mark[6*i+2];
         geo->bcout[6*bcout_counter+3] = geo->elem_mark[6*i+3];
         geo->bcout[6*bcout_counter+4] = geo->elem_mark[6*i+4];
         geo->bcout[6*bcout_counter+5] = geo->elem_mark[6*i+5];
         bcout_counter++;
      }
      if (geo->elem_mark[6*i+5] == geo->bcperiodicnr1)
      {
         geo->bcperiodic1[6*bcperiodic1_counter+0] = geo->elem_mark[6*i+0];
         geo->bcperiodic1[6*bcperiodic1_counter+1] = geo->elem_mark[6*i+1];
         geo->bcperiodic1[6*bcperiodic1_counter+2] = geo->elem_mark[6*i+2];
         geo->bcperiodic1[6*bcperiodic1_counter+3] = geo->elem_mark[6*i+3];
         geo->bcperiodic1[6*bcperiodic1_counter+4] = geo->elem_mark[6*i+4];
         geo->bcperiodic1[6*bcperiodic1_counter+5] = geo->elem_mark[6*i+5];
         bcperiodic1_counter++;
      }
      if (geo->elem_mark[6*i+5] == geo->bcperiodicnr2)
      {
         geo->bcperiodic2[6*bcperiodic2_counter+0] = geo->elem_mark[6*i+0];
         geo->bcperiodic2[6*bcperiodic2_counter+1] = geo->elem_mark[6*i+1];
         geo->bcperiodic2[6*bcperiodic2_counter+2] = geo->elem_mark[6*i+2];
         geo->bcperiodic2[6*bcperiodic2_counter+3] = geo->elem_mark[6*i+3];
         geo->bcperiodic2[6*bcperiodic2_counter+4] = geo->elem_mark[6*i+4];
         geo->bcperiodic2[6*bcperiodic2_counter+5] = geo->elem_mark[6*i+5];
         bcperiodic2_counter++;
      }
   }

   if (    (bcin_counter!= geo->n_bcin) ||
         (bcout_counter!= geo->n_bcout) ||
         (bcperiodic1_counter != geo->n_bcperiodic1) ||
         (bcperiodic2_counter != geo->n_bcperiodic2)    )
   {
      printf("Fehler in CreateBocoObject!\n");
      fprintf(stderr,"\tbcin_counter = %d, geo->n_bcin = %d\n", bcin_counter, geo->n_bcin);
      fprintf(stderr,"\tbcout_counter = %d, geo->n_bcout = %d\n", bcout_counter, geo->n_bcout);
      fprintf(stderr,"\tbcperiodic1_counter = %d, geo->n_bcperiodic1 = %d\n", bcperiodic1_counter, geo->n_bcperiodic1);
      fprintf(stderr,"\tbcperiodic2_counter = %d, geo->n_bcperiodic2 = %d\n", bcperiodic2_counter, geo->n_bcperiodic2);
      fprintf(stderr,"Aborting!\n");	  
      return FAIL;
   }

   //   0. number of columns per info
   sprintf(name,"%s_colinfo",basename);
   size[0] = 6;
   size[1] = 0;
   coDoIntArr *colInfo = new coDoIntArr(name,1,size);
   data = colInfo->getAddress();
   data[0] = COL_NODE;                            // (=2)
   data[1] = COL_ELEM;                            // (=2)
   data[2] = COL_DIRICLET;                        // (=2)
   data[3] = COL_WALL;                            // (=6)
   data[4] = COL_BALANCE;                         // (=6)
   data[5] = COL_PRESS;                           // (=6)
   partObj[0]=colInfo;

   //   1. type of node
   sprintf(name,"%s_nodeinfo",basename);
   size[0] = COL_NODE;
   size[1] = geo->n_nodes;
   coDoIntArr *nodeInfo = new coDoIntArr(name,2,size);
   data = nodeInfo->getAddress();
   for (i = 0; i < geo->n_nodes; i++)
   {
      *data++ = i+1;                              // may be, that we later do it really correct
      *data++ = 0;                                // same comment ;-)
   }
   partObj[1]=nodeInfo;

   //   2. type of element
   sprintf(name,"%s_eleminfo",basename);
   size[0] = 2;
   size[1] = geo->n_elem_3d*COL_ELEM;
   coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
   data = elemInfo->getAddress();
   for (i = 0; i < geo->n_elem_3d; i++)
   {
      *data++ = i+1;                              // may be, that we later do it really corect
      *data++ = 0;                                // same comment ;-)
   }
   partObj[2]=elemInfo;

   //   3. list of nodes with bc (a node may appear more than one time)
   //      and its types
   sprintf(name,"%s_diricletNodes",basename);
   size [0] = COL_DIRICLET;
   size [1] = 5*(geo->n_in_rb/5);
   coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
   data = diricletNodes->getAddress();

   //   4. corresponding value to 3.
   sprintf(name,"%s_diricletValue",basename);
   coDoFloat *diricletValues
      = new coDoFloat(name, 5*(geo->n_in_rb/5) );
   diricletValues->getAddress(&bPtr);

   if (!p_generate_inlet_boco->getValue())
   {
      for (i = 0; i < (geo->n_in_rb/5); i++)
      {
         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
         *data++ = 1;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+0];  // u
		 
         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
         *data++ = 2;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+1];  // v

         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
         *data++ = 3;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+2];  // w

         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
         *data++ = 4;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+3];  // k

         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
         *data++ = 5;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+4];  // epsilon

//         *data++ = geo->dirichlet_nodes[i]+1;     // node-number
//         *data++ = 6;                             // type of node
//         *bPtr++ = 0.0;                           // temperature
      }
      if (p_abs2rel->getValue())
      {
         float omega = float(M_PI * p_n->getValue() / 30.);
         int rotaxis = p_RotAxis->getValue();
         int nodenr;

         diricletValues->getAddress(&bPtr);

         if (!strcmp(s_RotAxis[rotaxis], "x"))
         {
            for (i = 0; i < (geo->n_in_rb/5); i++)
            {
               nodenr = geo->dirichlet_nodes[i];

               bPtr++;
               *bPtr++ = geo->dirichlet_values[5*i+1] - omega * geo->z[nodenr];
               *bPtr++ = geo->dirichlet_values[5*i+2] + omega * geo->y[nodenr];
               bPtr++;                           // epsilon
               bPtr++;                           // k
               bPtr++;
            }
         }
         if (!strcmp(s_RotAxis[rotaxis], "y"))
         {
            for (i = 0; i < (geo->n_in_rb/5); i++)
            {
               nodenr = geo->dirichlet_nodes[i];

               *bPtr++ = geo->dirichlet_values[5*i+0] + omega * geo->z[nodenr];
               bPtr++;
               *bPtr++ = geo->dirichlet_values[5*i+2] - omega * geo->x[nodenr];
               bPtr++;                           // epsilon
               bPtr++;                           // k
               bPtr++;
            }
         }
         if (!strcmp(s_RotAxis[rotaxis], "z"))
         {
            for (i = 0; i < (geo->n_in_rb/5); i++)
            {
               nodenr = geo->dirichlet_nodes[i];

               *bPtr++ = geo->dirichlet_values[5*i+0] - omega * geo->y[nodenr];
               *bPtr++ = geo->dirichlet_values[5*i+1] + omega * geo->x[nodenr];
               bPtr++;
               bPtr++;                           // epsilon
               bPtr++;                           // k
               bPtr++;
            }
         }
      }
   }
   else                                           //generate radial inflow boco
   {
      int nodenr;
      float alpha;
      //float rad;
      float v = 4.0;

      for (i = 0; i < (geo->n_in_rb/5); i++)
      {

         nodenr = geo->dirichlet_nodes[i];
         //rad = sqrt ( sqr(geo->x[nodenr]) + sqr(geo->y[nodenr]) );
         alpha = atan2 (geo->y[nodenr], geo->x[nodenr]);

         *data++ = nodenr+1;                      // node-number
         *data++ = 1;                             // type of node
         *bPtr++ = v * sin(alpha);                // u

         *data++ = nodenr+1;                      // node-number
         *data++ = 2;                             // type of node
         *bPtr++ = v * cos(alpha);                // v

         *data++ = nodenr+1;                      // node-number
         *data++ = 3;                             // type of node
         *bPtr++ = -v;                            // w

         *data++ = nodenr+1;                      // node-number
         *data++ = 4;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+3];  // k

         *data++ = nodenr+1;                      // node-number
         *data++ = 5;                             // type of node
         *bPtr++ = geo->dirichlet_values[5*i+4];  // epsilon

//         *data++ = nodenr+1;                      // node-number
//         *data++ = 6;                             // type of node
//         *bPtr++ = 0.0;                           // temperature
      }
   }

   partObj[3] = diricletNodes;
   partObj[4] = diricletValues;

   //   5. wall
   sprintf(name,"%s_wall",basename);
   size[0] = COL_WALL;
   size[1] = geo->n_wall;
   coDoIntArr *faces = new coDoIntArr(name, 2, size );
   data = faces->getAddress();
   if (geo->new_rbfile)
   {
      n_col_wall = 6;
   }
   else
   {
      n_col_wall = 5;
   }
   if (geo->new_rbfile==0)
   {
	   for (i = 0; i < geo->n_wall; i++)
	   {
    	  *data++ = geo->wall[n_col_wall*i+0]+1;
    	  *data++ = geo->wall[n_col_wall*i+1]+1;
    	  *data++ = geo->wall[n_col_wall*i+2]+1;
    	  *data++ = geo->wall[n_col_wall*i+3]+1;
    	  *data++ = geo->wall[n_col_wall*i+4]+1;
		  *data++ = 55;								  // take number 55 for walls
    	  *data++ = 0;                                // wall: moving | standing. here so far: always standing!
	   }
   }
   else
   {
	   for (i = 0; i < geo->n_wall; i++)
	   {
    	  *data++ = geo->wall[n_col_wall*i+0]+1;
    	  *data++ = geo->wall[n_col_wall*i+1]+1;
    	  *data++ = geo->wall[n_col_wall*i+2]+1;
    	  *data++ = geo->wall[n_col_wall*i+3]+1;
    	  *data++ = geo->wall[n_col_wall*i+4]+1;
    	  *data++ = geo->wall[n_col_wall*i+5];        // take wall number from rb-File
    	  *data++ = 0;                                // wall: moving | standing. here so far: always standing!
	   }
   }
   partObj[5]=faces;

   //   6. balance
   sprintf(name,"%s_balance",basename);
   size[0] = COL_BALANCE;
   size[1] =   geo->n_bcin +
      geo->n_bcout +
      geo->n_bcperiodic1 +
      geo->n_bcperiodic2;

   coDoIntArr *balance = new coDoIntArr(name, 2, size );
   data=balance->getAddress();
   for (i = 0; i < geo->n_bcin; i++)
   {
      *data++ = geo->bcin[6*i+0]+1;
      *data++ = geo->bcin[6*i+1]+1;
      *data++ = geo->bcin[6*i+2]+1;
      *data++ = geo->bcin[6*i+3]+1;
      *data++ = geo->bcin[6*i+4]+1;               // volume element
      *data++ = geo->bcin[6*i+5];                 // bila-nr.
   }
   for (i = 0; i < geo->n_bcout; i++)
   {
      *data++ = geo->bcout[6*i+0]+1;
      *data++ = geo->bcout[6*i+1]+1;
      *data++ = geo->bcout[6*i+2]+1;
      *data++ = geo->bcout[6*i+3]+1;
      *data++ = geo->bcout[6*i+4]+1;              // volume element
      *data++ = geo->bcout[6*i+5];                // bila-nr.
   }

   for (i = 0; i < geo->n_bcperiodic1; i++)
   {
      *data++ = geo->bcperiodic1[6*i+0]+1;
      *data++ = geo->bcperiodic1[6*i+1]+1;
      *data++ = geo->bcperiodic1[6*i+2]+1;
      *data++ = geo->bcperiodic1[6*i+3]+1;
      *data++ = geo->bcperiodic1[6*i+4]+1;        // volume element
      *data++ = geo->bcperiodic1[6*i+5];          // bila-nr.
   }

   for (i = 0; i < geo->n_bcperiodic2; i++)
   {
      *data++ = geo->bcperiodic2[6*i+0]+1;
      *data++ = geo->bcperiodic2[6*i+1]+1;
      *data++ = geo->bcperiodic2[6*i+2]+1;
      *data++ = geo->bcperiodic2[6*i+3]+1;
      *data++ = geo->bcperiodic2[6*i+4]+1;        // volume element
      *data++ = geo->bcperiodic2[6*i+5];          // bila-nr.
   }
   partObj[6] = balance;

   //  7. pressure bc
   sprintf(name,"%s_pressElems",basename);
   size[0] = COL_PRESS;
   if (geo->n_press_rb>0)
   {
      // use pressure bc from bc-file
      size[1] = geo->n_press_rb;
      coDoIntArr *pressElems = new coDoIntArr(name, 2, size );
      data=pressElems->getAddress();

      //  8. pressure bc: value
      sprintf(name,"%s_pressVal",basename);
//      coDoFloat *pressValues  = new coDoFloat(name, geo->n_press_rb);
//      pressValues->getAddress(&bPtr);

      // old rbfile-format

      if(geo->new_rbfile==0)
      {
         for (i = 0; i < geo->n_press_rb; i++)
         {
        	 *data++ = geo->press_rb[5*i+0]+1;
        	 *data++ = geo->press_rb[5*i+1]+1;
        	 *data++ = geo->press_rb[5*i+2]+1;
        	 *data++ = geo->press_rb[5*i+3]+1;
        	 *data++ = geo->press_rb[5*i+4]+1;        // volume lement
        	 *data++ = 77;                            // pressure bila
//        	 *bPtr++ = geo->press_rb_value[i];        // value
         }
      }
	  else
	  {
    	  for (i = 0; i < geo->n_press_rb; i++)
    	  {
        	 *data++ = geo->press_rb[6*i+0]+1;
        	 *data++ = geo->press_rb[6*i+1]+1;
        	 *data++ = geo->press_rb[6*i+2]+1;
        	 *data++ = geo->press_rb[6*i+3]+1;
        	 *data++ = geo->press_rb[6*i+4]+1;        // volume lement
        	 *data++ = 77;                            // pressure bila
//        	 *bPtr++ = geo->press_rb_value[i];        // value
    	  }
      }
      partObj[7] = pressElems;
//      partObj[8] = pressValues;
   }
   else
   {
      // use bcout to generate pressure bc
      size[1] = geo->n_bcout;
      coDoIntArr *pressElems = new coDoIntArr(name, 2, size );
      data=pressElems->getAddress();

      //  8. pressure bc: value for outlet elements
      sprintf(name,"%s_pressVal",basename);
//      coDoFloat *pressValues = new coDoFloat(name, geo->n_bcout);
//      pressValues->getAddress(&bPtr);
      for (i = 0; i < geo->n_bcout; i++)
      {
         *data++ = geo->bcout[6*i+0]+1;
         *data++ = geo->bcout[6*i+1]+1;
         *data++ = geo->bcout[6*i+2]+1;
         *data++ = geo->bcout[6*i+3]+1;
         *data++ = geo->bcout[6*i+4]+1;           // volume element
		 *data++ = 77;                            // pressure bila
//         *bPtr++ = 0.0;                           // initialise pressure bc to 0.0
      }
      partObj[7] = pressElems;
//      partObj[8] = pressValues;
   }

   partObj[8] = NULL;
/*
   // generate bcin Polygons
   bcin_counter=0;
   for (i=0; i<geo->n_bcin;i++)
   {
      geo->bcin[6*bcin_counter+0];
      geo->bcin[6*bcin_counter+1];
      geo->bcin[6*bcin_counter+2];
      geo->bcin[6*bcin_counter+3];
      geo->bcin[6*bcin_counter+4];
      geo->bcin[6*bcin_counter+5];
      bcin_counter++;
   }
*/
   return(0);
}



int ReadIHS2::checkfornewnr (int *nrlist, int *n_nrs, char **names, int nr, int last_nr, char *name)
{
   int nr_already_known = 0;

   int i;

   if (*n_nrs==100)
   {
      sendError("\n\nFehler! es werden maximal 100 verschiedene bila-Nummern unterstuetzt!\n");
      return FAIL;
   }

   if (last_nr != -1)
   {
      for (i=0; i<*n_nrs; i++)
      {
         if (nr == nrlist[2*i])
         {
	     nr_already_known = 1;
	     break;
         }
      }
      if (nr_already_known == 0)
      {
         nrlist[2*(*n_nrs)] = nr;
		 names[*n_nrs]=strdup(name);
         (*n_nrs)++;
      }
   }
   else
   {
      *n_nrs = 0;
      nrlist[2*0+0] = nr;
      names[*n_nrs]=strdup(name);
   }

   return(0);

}


int ReadIHS2::countnrs (int n_nrs, int *nr_list, int nr)
{
   int i;
   for (i=0; i<n_nrs; i++)
   {
      if (nr_list[2*i] == nr)
         nr_list[2*i+1]++;
   }
   return(0);
}

MODULE_MAIN(IO, ReadIHS2)
