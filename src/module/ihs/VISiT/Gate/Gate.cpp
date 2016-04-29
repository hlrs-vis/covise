#include <config/CoviseConfig.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <strings.h>
#else
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>

#include "Gate.h"
#include <Gate/include/ggrid.h>
#include <General/include/log.h>
#include <General/include/flist.h>

#ifndef YAC
#include <api/coFeedback.h>
#endif

#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

#define RAD(x) ((x) * M_PI/180.0)
#define GRAD(x)   ((x) * 180.0/M_PI)

Gate::Gate(int argc, char *argv[])
: coSimpleModule(argc, argv, "Gate" )
{
   geo = NULL;
   fprintf(stderr, "Gate::Gate()\n");

   // We talk a lot to the logfile
   //SetLogLevel(LOG_ALL);

   // start file param
   fprintf(stderr, "Gate::Gate() Init of StartFile\n");
   startFile = addFileBrowserParam("startFile","Start file");
#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE"); 
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(defaultDir)
      startFile->setValue(coCoviseConfig::getEntry("value","Module.IHS.DataPath",defaultDir).c_str(),"*.cfg");

   else
      startFile->setValue(coCoviseConfig::getEntry("value","Module.IHS.DataPath","/data/IHS").c_str(),"*.cfg");
fprintf(stderr, "Gate::Gate() Init of StartFile1\n");
   // We build the User-Menue ...
   Gate::CreateUserMenu();
fprintf(stderr, "Gate::Gate() Init of StartFile2\n");
   // the output ports
   fprintf(stderr, "Gate::Gate() SetOutPort\n");
   grid   = addOutputPort("grid","UnstructuredGrid","Computation Grid");

   blade  = addOutputPort("blade","Polygons","Blade Polygons");
   hub    = addOutputPort("hub","Polygons","Hub Polygons");
   shroud = addOutputPort("shroud","Polygons","Shroud Polygons");

   bladenormals  = addOutputPort("bladenormals","Vec3","Blade Normals");
   hubnormals    = addOutputPort("hubnormals","Vec3","Hub Normals");
   shroudnormals = addOutputPort("shroudnormals","Vec3","Shroud Normals");

   bcin = addOutputPort("bcin","Polygons","Cells at entry");
   bcout = addOutputPort("bcout","Polygons","Cells at exit");
   bcwall = addOutputPort("bcwall","Polygons","Cells at walls");
   bcperiodic = addOutputPort("bcperiodic","Polygons","Cells at periodic borders");

   boco = addOutputPort("boco", "USR_FenflossBoco", "Boundary Conditions");
   plot2d = addOutputPort("_2dplot", "Vec2", "n-Q-diagramm");

   isInitialized = 0;
}


void Gate::postInst()
{
   startFile->show();
}


void Gate::param(const char *portname, bool inMapLoading)
{
   char buf[255];

   fprintf(stderr, "Gate::param = %s\n", portname);
   if (strcmp(portname,"startFile")==0)
   {
      if (isInitialized)
      {
         sendError("We Had an input file before...");
         return;
      }
#ifndef YAC
      Covise::getname(buf, startFile->getValue());
#else
      coFileHandler::getName(buf, startFile->getValue());
#endif
      if(strlen(buf)==0)
      {
         sendError("startFile parameter incorrect");
      }
      else
      {

         fprintf(stderr, "Gate::param = ReadGeometry(%s) ...", buf);
         geo = ReadGeometry(buf);                 //cg.c: ReadGate -> ga_io.c
         if(geo ==NULL || geo->ga==NULL)
         {
         }
         else
         {
            strcpy(geo->ga->cfgfile, buf);
            fprintf(stderr, "done\n");
         }

         if (geo && geo->ga)
         {
            if (!inMapLoading)
            {
               Gate::Struct2CtrlPanel();
               isInitialized = 1;
            }
         }
      }
   }

   /*
    */
   //selfExec();
}

#ifndef YAC
void Gate::quit()
{
   // :-)
}
#else
int Gate::quit()
{
   // }:->
   return 0;
}
#endif

int Gate::compute(const char *)
{
   //time
#ifndef WIN32
   timeval time1, time2;
   gettimeofday (&time1, NULL);
#endif

   coDoPolygons *poly;
   struct covise_info *ci;
   char name[256];
   char buf[256];
   int res = -1;
   int i;
   struct ggrid *gg;
   coDoVec2 *plot;
   float *xpl,*ypl;

   fprintf(stderr, "Gate::compute(const char *) entering... \n");

   if ( (!geo) || (!geo->ga) )
   {
      sendError("Please select a parameter file first!!");
      return FAIL;
   }

   CtrlPanel2Struct();
#ifndef YAC
   Covise::getname(name,startFile->getValue());
#else
   coFileHandler::getName(name,startFile->getValue());
#endif
   strcat(name, ".new");
   res = WriteGeometry(geo, name);
   fprintf(stderr, "WriteGeometry sends: %d\n", res);

   /////////////////////////////
   // create geometry for COVISE
   if ( (ci = CreateGeometry4Covise(geo)) )
   {
      fprintf(stderr, "Gate::compute(const char *): Geometry created\n");
#ifndef YAC
      coFeedback feedback("Gate");

      feedback.addString("Blade");

      feedback.addPara(p_BladeAngle);
      feedback.addPara(p_Q);
      feedback.addPara(p_H);
      feedback.addPara(p_n);
      feedback.addPara(p_Q_opt);
      feedback.addPara(p_n_opt);
      feedback.addPara(p_NumberOfBlades);
#endif
      poly = new coDoPolygons(hub->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->lvx->num,  ci->lvx->list,
         ci->lpol->num, ci->lpol->list);
      poly->addAttribute("MATERIAL","metal metal.14");
      poly->addAttribute("vertexOrder","1");

      GenerateNormals(HUB,poly,hubnormals->getObjName());
      hub->setCurrentObject(poly);

      poly = new coDoPolygons(shroud->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->cvx->num,  ci->cvx->list,
         ci->cpol->num, ci->cpol->list);
      poly->addAttribute("MATERIAL","metal metal.14");
      poly->addAttribute("vertexOrder","1");
      GenerateNormals(SHROUD,poly,shroudnormals->getObjName());
      shroud->setCurrentObject(poly);

      poly = new coDoPolygons(blade->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->vx->num,  ci->vx->list,
         ci->pol->num, ci->pol->list);
      poly->addAttribute("MATERIAL","metal metal.14");
      poly->addAttribute("vertexOrder","1");
#ifndef YAC
      feedback.apply(poly);
#endif
      // TUI%d %cmodule \n instance \n host \n parameterName \n parent\n text \n xPos \n yPos \n floatSlider \n parameterName \n min \n max \n value)
      // ... TUI-VR-Slider fuer Parameter Q ...
#ifndef YAC
      sprintf(buf,"M%s\n%s\n%s\n" M_Q "\nNenndaten\n" M_Q "\n0\n0\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,200.,geo->ga->Q);
      poly->addAttribute("TUI0",buf);
      sprintf(buf,"M%s\n%s\n%s\n" M_H "\nNenndaten\n" M_H "\n0\n1\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,50.,geo->ga->H);
      poly->addAttribute("TUI1",buf);
      sprintf(buf,"M%s\n%s\n%s\n" M_N "\nNenndaten\n" M_N "\n0\n2\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,1000.,geo->ga->n);
      poly->addAttribute("TUI2",buf);
      sprintf(buf,"M%s\n%s\n%s\n" M_N_OPT "\nNenndaten\n" M_N_OPT "\n0\n3\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,500.,geo->ga->nopt);
      poly->addAttribute("TUI3",buf);
      sprintf(buf,"M%s\n%s\n%s\n" M_Q_OPT "\nNenndaten\n" M_Q_OPT "\n0\n4\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,100.,geo->ga->Qopt);
      poly->addAttribute("TUI4",buf);
     
      sprintf(buf,"M%s\n%s\n%s\n" M_NUMBER_OF_BLADES "\nNenndaten\n" M_NUMBER_OF_BLADES "\n0\n5\nint\n%d\n1\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),geo->ga->nob);
      poly->addAttribute("TUI5",buf);
      /*if (geo->ga->close==1)
      {
         sprintf(buf,"M%s\n%s\n%s\n"M_BLADE_ANGLE"\nNenndaten\n"M_BLADE_ANGLE"\n0\n5\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),geo->ga->beta_min,geo->ga->beta_max,geo->ga->bangle*180./M_PI);
      }
      else*/
      {
         sprintf(buf,"M%s\n%s\n%s\n" M_BLADE_ANGLE "\nNenndaten\n" M_BLADE_ANGLE "\n0\n6\nfloatSlider\n%f\n%f\n%f\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),-20.,8.,geo->ga->bangle*180./M_PI);
      }
      poly->addAttribute("TUI6",buf);

      // ... MENUE-VR-Slider fuer Parameter Q ...
      sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_Q "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,200.,geo->ga->Q);
      poly->addAttribute("SLIDER0",buf);

      // ... MENUE-VR-Slider fuer Parameter H ...
      sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_H "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,50.,geo->ga->H);
      poly->addAttribute("SLIDER1",buf);

      // ... MENUE-VR-Slider fuer Parameter n ...
      sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_N "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,1000.,geo->ga->n);
      poly->addAttribute("SLIDER2",buf);

      // ... MENUE-VR-Slider fuer Parameter nopt ...
      sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_N_OPT "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,500.,geo->ga->nopt);
      poly->addAttribute("SLIDER3",buf);

      // ... MENUE-VR-Slider fuer Parameter Qopt ...
      sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_Q_OPT "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),0.,100.,geo->ga->Qopt);
      poly->addAttribute("SLIDER4",buf);

      // ... MENUE-VR-Slider fuer Parameter blade angle ...
/*      if (geo->ga->close==1)
      {
         sprintf(buf,"M%s\n%s\n%s\nfloat\n"M_BLADE_ANGLE"\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),geo->ga->beta_min,geo->ga->beta_max,geo->ga->bangle*180./M_PI);
      }
      else */
      {
         sprintf(buf,"M%s\n%s\n%s\nfloat\n" M_BLADE_ANGLE "\n%f\n%f\n%f\nNenndaten\n",Covise::get_module(),Covise::get_instance(),Covise::get_host(),-20.,10.,geo->ga->bangle*180./M_PI);
      }
      poly->addAttribute("SLIDER6",buf);
#endif
      GenerateNormals(BLADE,poly,bladenormals->getObjName());
      blade->setCurrentObject(poly);
   }
   else
      fprintf(stderr, "Error in CreateGeometry4Covise (%s, %d)\n", __FILE__, __LINE__);

   fprintf(stderr, "Gate::compute(const char *) done\n");

   //////////////// This creates the volume grid ////////////////////

   ////////////////////////
   // if button is pushed --> create computation grid
   if (p_makeGrid->getValue())
   {
      int size[2];

      if (p_lockmakeGrid->getValue() == 0)
         p_makeGrid->setValue(0);                 // push off button

      if (geo->ga == NULL)
      {
         sendError("Cannot create grid because geo->ga is NULL!");
         return(1);
      }

      gg = CreateGGrid(geo->ga);

      fprintf(stderr, "Gate: Grid created\n");

      coDoUnstructuredGrid *unsGrd =
                                                  // name of USG object
         new coDoUnstructuredGrid(grid->  getObjName(),
         gg->e->nume,                             // number of elements
         8*gg->e->nume,                           // number of connectivities
         gg->p->nump,                             // number of coordinates
         1);                                      // does type list exist?

#ifdef YAC
      unsGrd->getHdr()->setTime( -1, 0 );
      unsGrd->getHdr()->setRealTime( 1.0 );
#endif

      int *elem,*conn,*type;
      float *xc,*yc,*zc;
      unsGrd->getAddresses(&elem, &conn, &xc, &yc, &zc);
      unsGrd->getTypeList(&type);

      printf("nelem  = %d\n", gg->e->nume);
      printf("nconn  = %d\n", 8*gg->e->nume);
      printf("nccord = %d\n", gg->p->nump);

      int **GgridConn = gg->e->e;
      for (i = 0; i < gg->e->nume; i++)
      {
         *elem = 8*i;               elem++;

         *conn = (*GgridConn)[0];   conn++;
         *conn = (*GgridConn)[1];   conn++;
         *conn = (*GgridConn)[2];   conn++;
         *conn = (*GgridConn)[3];   conn++;
         *conn = (*GgridConn)[4];   conn++;
         *conn = (*GgridConn)[5];   conn++;
         *conn = (*GgridConn)[6];   conn++;
         *conn = (*GgridConn)[7];   conn++;

         *type = TYPE_HEXAGON;      type++;

         GgridConn++;

      }

      // copy geometry coordinates to unsgrd
      memcpy(xc, gg->p->x, gg->p->nump*sizeof(float));
      memcpy(yc, gg->p->y, gg->p->nump*sizeof(float));
      memcpy(zc, gg->p->z, gg->p->nump*sizeof(float));

      // set out port
      grid->setCurrentObject(unsGrd);
      char para[256];
#ifndef YAC
      snprintf(para, 256, "%ld", p_NumberOfBlades->getValue());
#else
      snprintf(para, 256, "%d", p_NumberOfBlades->getValue());
#endif
      unsGrd->addAttribute("number_of_blades", para);
      unsGrd->addAttribute("periodic", "1");
      unsGrd->addAttribute("rotating", "0");
      unsGrd->addAttribute("revolutions", "0");
      unsGrd->addAttribute("walltext", "");

#ifndef YAC
      snprintf(para, 256, "111,passend,120,130,perio_rota,%ld,3", p_NumberOfBlades->getValue());
#else
      snprintf(para, 256, "111,passend,120,130,perio_rota,%d,3", p_NumberOfBlades->getValue());
#endif
      unsGrd->addAttribute("periotext", para);

      snprintf(para, 256, "angle_of_multi_rotation %f\nnumber_of_rotations %d", 360.0 / p_NumberOfBlades->getValue(), (int) p_NumberOfBlades->getValue());
      unsGrd->addAttribute("TRANSFORM", para);

      // boundary condition lists
      // 1. Cells at entry
      poly = new coDoPolygons(bcin->getObjName(),
         gg->p->nump,
         gg->p->x, gg->p->y, gg->p->z,
         gg->bcin->num,  gg->bcin->list,
         gg->bcinpol->num, gg->bcinpol->list);
      //poly->addAttribute("MATERIAL","metal metal.30");
      poly->addAttribute("vertexOrder","1");
      bcin->setCurrentObject(poly);

      // 2. Cells at exit
      poly = new coDoPolygons(bcout->getObjName(),
         gg->p->nump,
         gg->p->x, gg->p->y, gg->p->z,
         gg->bcout->num,  gg->bcout->list,
         gg->bcoutpol->num, gg->bcoutpol->list);
      //poly->addAttribute("MATERIAL","metal metal.30");
      poly->addAttribute("vertexOrder","1");
      bcout->setCurrentObject(poly);

      // 3. Cells at walls
      poly = new coDoPolygons(bcwall->getObjName(),
         gg->p->nump,
         gg->p->x, gg->p->y, gg->p->z,
         gg->bcwall->num,  gg->bcwall->list,
         gg->bcwallpol->num, gg->bcwallpol->list);
      //poly->addAttribute("MATERIAL","metal metal.30");
      poly->addAttribute("vertexOrder","1");
      bcwall->setCurrentObject(poly);

      // 4. Cells at periodic borders
      poly = new coDoPolygons(bcperiodic->getObjName(),
         gg->p->nump,
         gg->p->x, gg->p->y, gg->p->z,
         gg->bcperiodic->num,  gg->bcperiodic->list,
         gg->bcperiodicpol->num, gg->bcperiodicpol->list);
      //poly->addAttribute("MATERIAL","metal metal.30");
      poly->addAttribute("vertexOrder","1");
      bcperiodic->setCurrentObject(poly);

      // we had several additional info, we should send to the
      // Domaindecomposition:
      //   0. number of columns per info
      //   1. type of node
      //   2. type of element
      //   3. list of nodes with bc
      //   4. corresponding type to 3.
      //   5. wall
      //   6. balance
      //   7. pressure
      //   8. NULL

      coDistributedObject *partObj[10];
      int *data;
      float *bPtr;
#ifndef YAC
      const char *basename = boco->getObjName();
      //   0. number of columns per info
      sprintf(name,"%s_colinfo",basename);
#else
      coObjInfo name = boco->getNewObjectInfo();
#endif
      size[0] = 6;
      size[1] = 0;
      coDoIntArr *colInfo = new coDoIntArr(name,1,size);
      data = colInfo->getAddress();
      data[0] = GG_COL_NODE;                      // (=2)
      data[1] = GG_COL_ELEM;                      // (=2)
      data[2] = GG_COL_DIRICLET;                  // (=2)
      data[3] = GG_COL_WALL;                      // (=7)
      data[4] = GG_COL_BALANCE;                   // (=7)
      data[5] = GG_COL_PRESS;                     // (=6)
      partObj[0]=colInfo;

      //   1. type of node
#ifndef YAC
      sprintf(name,"%s_nodeinfo",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size[0] = GG_COL_NODE;
      size[1] = gg->p->nump;
      coDoIntArr *nodeInfo = new coDoIntArr(name,2,size);
      data = nodeInfo->getAddress();
      for (i = 0; i < gg->p->nump; i++)
      {
         *data++ = i+1;                           // may be, that we later do it really correct
         *data++ = 0;                             // same comment ;-)
      }
      partObj[1]=nodeInfo;

      //   2. type of element
#ifndef YAC
      sprintf(name,"%s_eleminfo",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size[0] = GG_COL_ELEM;
      size[1] = gg->e->nume;
      coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
      data = elemInfo->getAddress();
      for (i = 0; i < gg->e->nume; i++)
      {
         *data++ = i+1;                           // may be, that we later do it really corect
         *data++ = 0;                             // same comment ;-)
      }
      partObj[2]=elemInfo;

      //   3. list of nodes with bc (a node may appear more than one time)
      //      and its types
#ifndef YAC
      sprintf(name,"%s_diricletNodes",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size [0] = GG_COL_DIRICLET;
      size [1] = 5*gg->bcin->num;
      coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
      data = diricletNodes->getAddress();

      //   4. corresponding value to 3.
#ifndef YAC
      sprintf(name,"%s_diricletValue",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      coDoFloat *diricletValues
         = new coDoFloat(name, 5*gg->bcin->num);
      diricletValues->getAddress(&bPtr);

      float r_in = p_InletRadius->getValue() + p_grid_len_expand_in->getValue();
      float h_in = p_InletHeight->getValue();
      float q_in = p_Q->getValue();

      float v_in = (float)(q_in / (2.0*M_PI*r_in*h_in));

      printf("inlet velocity v_in=%5.2lf\n", v_in);

      for (i = 0; i < gg->bcin->num; i++)
      {
         *data++ = gg->bcin->list[i]+1;           // node-number
         *data++ = 1;                             // type of node
                                                  // u
         *bPtr++ = - gg->p->x[gg->bcin->list[i]] / r_in * v_in;
         *data++ = gg->bcin->list[i]+1;           // node-number
         *data++ = 2;                             // type of node
                                                  // v
         *bPtr++ = - gg->p->y[gg->bcin->list[i]] / r_in * v_in;
         *data++ = gg->bcin->list[i]+1;           // node-number
         *data++ = 3;                             // type of node
         *bPtr++ = 0.0;                           // w
         *data++ = gg->bcin->list[i]+1;           // node-number
         *data++ = 4;                             // type of node
         *bPtr++ = gg->k;                         // k
         *data++ = gg->bcin->list[i]+1;           // node-number
         *data++ = 5;                             // type of node
         *bPtr++ = gg->epsilon;                   // epsilon
	 // fl: no more temperature
	 //         *data++ = gg->bcin->list[i]+1;           // node-number
	 //         *data++ = 6;                             // type of node
         //*bPtr++ = gg->T;                         // temperature
      }

      partObj[3] = diricletNodes;
      partObj[4] = diricletValues;

      //   5. wall
#ifndef YAC
      sprintf(name,"%s_wallValue",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      coDoFloat *wallValues
         = new coDoFloat(name, gg->bcwallvol->num);
      wallValues->getAddress(&bPtr);

#ifndef YAC
      sprintf(name,"%s_wall",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size[0] = GG_COL_WALL;
      size[1] = gg->bcwallvol->num;
      coDoIntArr *faces = new coDoIntArr(name, 2, size );
      data = faces->getAddress();
      for (i = 0; i < gg->bcwallvol->num; i++)
      {
         *data++ = gg->bcwall->list[4*i+0]+1;
         *data++ = gg->bcwall->list[4*i+1]+1;
         *data++ = gg->bcwall->list[4*i+2]+1;
         *data++ = gg->bcwall->list[4*i+3]+1;
         *data++ = gg->bcwallvol->list[i]+1;
         *data++ = 55;
		 *data++ = 0;
      }
      partObj[5]=faces;

      //   6. balance
#ifndef YAC
      sprintf(name,"%s_balance",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size[0] = GG_COL_BALANCE;
      size[1] = gg->bcinvol->num + gg->bcoutvol->num + gg->bcperiodicval->num;

      coDoIntArr *balance = new coDoIntArr(name, 2, size );
      data=balance->getAddress();
      for (i = 0; i < gg->bcinvol->num; i++)
      {
         *data++ = gg->bcin->list[4*i+0]+1;
         *data++ = gg->bcin->list[4*i+1]+1;
         *data++ = gg->bcin->list[4*i+2]+1;
         *data++ = gg->bcin->list[4*i+3]+1;
         *data++ = gg->bcinvol->list[i]+1;
         *data++ = gg->bc_inval;
		 *data++ = 0;
      }
      for (i = 0; i < gg->bcoutvol->num; i++)
      {
         *data++ = gg->bcout->list[4*i+0]+1;
         *data++ = gg->bcout->list[4*i+1]+1;
         *data++ = gg->bcout->list[4*i+2]+1;
         *data++ = gg->bcout->list[4*i+3]+1;
         *data++ = gg->bcoutvol->list[i]+1;
         *data++ = gg->bc_outval;
		 *data++ = 0;
      }

      for (i = 0; i < gg->bcperiodicvol->num; i++)
      {
         *data++ = gg->bcperiodic->list[4*i+0]+1;
         *data++ = gg->bcperiodic->list[4*i+1]+1;
         *data++ = gg->bcperiodic->list[4*i+2]+1;
         *data++ = gg->bcperiodic->list[4*i+3]+1;
         *data++ = gg->bcperiodicvol->list[i]+1;
         *data++ = gg->bcperiodicval->list[i];
		 *data++ = 0;
      }

      partObj[6] = balance;

      //  7. pressure bc: outlet elements
#ifndef YAC
      sprintf(name,"%s_pressElems",basename);
#else
      name = boco->getNewObjectInfo();
#endif
      size[0] = GG_COL_PRESS;
      size[1] = gg->bcoutvol->num;
      coDoIntArr *pressElems = new coDoIntArr(name, 2, size );
      data=pressElems->getAddress();

      //  8. pressure bc: value for outlet elements
      /*sprintf(name,"%s_pressVal",basename);
      coDoFloat *pressValues
         = new coDoFloat(name, gg->bcoutvol->num);
		 pressValues->getAddress(&bPtr);*/
      for (i = 0; i < gg->bcoutvol->num; i++)
      {
         *data++ = gg->bcout->list[4*i+0]+1;
         *data++ = gg->bcout->list[4*i+1]+1;
         *data++ = gg->bcout->list[4*i+2]+1;
         *data++ = gg->bcout->list[4*i+3]+1;
         *data++ = gg->bcoutvol->list[i]+1;
         *data++ = 77;                            // pressure
//         *bPtr++ = -gg->bcpressval->list[i];
         //*bPtr++ = 0.0;
	  }
      partObj[7] = pressElems;
      partObj[8] = NULL;

      //partObj[9] = NULL;
      coDoSet *set = new coDoSet(boco->getObjName(),(coDistributedObject **)partObj);
      for (i = 0; i < 4; i++)                     // rei 27.10.2001 ????????? strange ...
         delete partObj[i];

      boco->setCurrentObject(set);

   }

   // refresh Q in Control Panel
   p_Q->setValue(geo->ga->Q);

   // 2D_Plot: create characterisitic diagram (plot)
   char plbuf[1000];

   int n_circles = 5;
   int n_isolines = 5;
   int steps = 100;
   
   plot = new coDoVec2(plot2d->getObjName(), 2*n_circles*steps + 4 + 2*n_isolines);
   plot->getAddresses(&xpl, &ypl);
   plot2d->setCurrentObject(plot);

   float xwmin, xwmax, ywmin, ywmax;
   xwmin = 0.0;
   ywmin = 0.0;
   float temp1 = sqrt(geo->ga->H);
   float temp2 = geo->ga->out_rad2 * geo->ga->out_rad2;
   xwmax = 2.0f *  geo->ga->out_rad2 * geo->ga->nopt / temp1;
   ywmax = 2.0f *  geo->ga->Qopt / (temp1 * temp2);
   
   int bladeang_min;
   int bladeang_max;
   
   sprintf(plbuf,"AUTOSCALE\n");
   sprintf(buf, "WORLD %f,%f,%f,%f\n", xwmin, ywmin, xwmax, ywmax );
   strcat(plbuf, buf);
   strcat(plbuf, "SETS SYMBOL 27\n");
   strcat(plbuf, "SETS LINESTYLE 0\n");
   strcat(plbuf, "title \"Muscheldiagramm\"\n");
   strcat(plbuf, "yaxis  label \"Q1'\"\n");
   strcat(plbuf, "xaxis  label \"n1'\"\n");

   plot->addAttribute("COMMANDS", plbuf);

   CreateShell(geo->ga, n_circles, steps, xpl, ypl);
   CreateIsoAngleLines(geo->ga, 2*n_circles*steps + 4, 5, xwmin, xwmax, xpl, ypl);
   
   bladeang_min = coCoviseConfig::getInt("Module.IHS.GateBladeMin",-100);
   bladeang_max = coCoviseConfig::getInt("Module.IHS.GateBladeMax",100);
   if  (bladeang_max!=-100 || bladeang_min!=100)
   {
      p_BladeAngle->setMin(bladeang_min);
      p_BladeAngle->setMax(bladeang_max);
   }
   else
   {
      if (geo->ga->close==1)
      {
         p_BladeAngle->setMin(geo->ga->beta_min);
         p_BladeAngle->setMax(geo->ga->beta_max);
      }
      else
      {
         p_BladeAngle->setMin(-100.);
         p_BladeAngle->setMax(100.);
      }
   }

   
   ///////////////////////// Free everything ////////////////////////////////
   // FreeStructGGrid(gg);
   // FreeStructGate(geo->ga);

#ifndef WIN32
   //time
   gettimeofday (&time2, NULL);
   double sec = double (time2.tv_sec - time1.tv_sec);
   double usec = 0.000001 * (time2.tv_usec - time1.tv_usec);
   printf("Gate Laufzeit: %5.2lf Sekunden\n", sec+usec);
#endif

   return SUCCESS;

}


void Gate::CreateUserMenu(void)
{

   fprintf(stderr, "Entering CreateUserMenu()\n");

   paraSwitch("types", "Select type of parameters");
   paraCase(M_GATE_DATA);

   // move this parameter to general section
   // (as soon as someone has killed the awful Covise-Bug)
   p_makeGrid = addBooleanParam(M_GENERATE_GRID, M_GENERATE_GRID);
   p_makeGrid->setValue(0);

   p_lockmakeGrid = addBooleanParam(M_LOCK_GRID, M_LOCK_GRID);
   p_lockmakeGrid->setValue(0);

   p_GeoFromFile = addBooleanParam(M_GEO_FROM_FILE, M_GEO_FROM_FILE);
   p_GeoFromFile->setValue(0);

   p_saveGrid = addBooleanParam(M_SAVE_GRID, M_SAVE_GRID);
   p_saveGrid->setValue(0);

   p_radialGate = addBooleanParam(M_RADIAL_GATE, M_RADIAL_GATE);
   p_radialGate->setValue(0);

   Gate::CreateMenuGateData();

   //paraSwitch("blade_data", "profile data");
   //paraEndSwitch();	// end of blade data

   paraEndCase();                                 // end of M_GATE_DATA

   paraCase(M_MERIDIAN_DATA);
   Gate::CreateMenuMeridianData();
   paraEndCase();                                 // end of M_MERIDIAN_DATA

   paraCase(M_BLADE_PROFILE_DATA);
   Gate::CreateMenuProfileData();
   paraEndCase();                                 // end of M_BLADE_PROFILE_DATA

   paraCase(M_GRID_DATA);

   p_grid_edge_ps = addInt32Param(M_EDGE_PS, M_EDGE_PS);
   p_grid_edge_ps->setValue(90);

   p_grid_edge_ss = addInt32Param(M_EDGE_SS, M_EDGE_SS);
   p_grid_edge_ss->setValue(110);

   p_grid_bound_layer = addFloatParam(M_BOUND_LAYER, M_BOUND_LAYER);
   p_grid_bound_layer->setValue(0.0);

   paraSwitch("subcategories_grid", "grid categories");
   paraCase(M_GRID_DATA_POINTS);
   Gate::CreateMenuGatePNumbers();
   paraEndCase();
   paraCase(M_GRID_DATA_LENGTH);
   Gate::CreateMenuGateCurveLengths();
   paraEndCase();
   paraCase(M_GRID_DATA_COMPRESSION);
   Gate::CreateMenuGateCompressions();
   paraEndCase();
   paraCase(M_GRID_DATA_SHIFT);
   Gate::CreateMenuGateShifts();
   paraEndCase();
   paraEndSwitch();

   paraEndCase();                                 // end of M_GRID_DATA

   paraEndSwitch();                               // end of "types"

}


void Gate::CreateMenuGateData(void)
{
   int bladeang_min;
   int bladeang_max;
   
   p_Q = addFloatSliderParam(M_Q, M_Q);
   p_Q->setValue(0.,200.,4.0);

   p_H = addFloatSliderParam(M_H, M_H);
   p_H->setValue(0.,50.,5.);

   p_n = addFloatSliderParam(M_N, M_N);
   p_n->setValue(0.,1000.,250.);

   p_Q_opt = addFloatSliderParam(M_Q_OPT, M_Q_OPT);
   p_Q_opt->setValue(0.,100.,4.);

   p_n_opt = addFloatSliderParam(M_N_OPT, M_N_OPT);
   p_n_opt->setValue(0.,500.,250.);

   p_NumberOfBlades = addInt32Param(M_NUMBER_OF_BLADES, M_NUMBER_OF_BLADES);
   p_NumberOfBlades->setValue(18);

   p_BladeAngle = addFloatSliderParam(M_BLADE_ANGLE, M_BLADE_ANGLE);
   bladeang_min = coCoviseConfig::getInt("Module.IHS.GateBladeMin",-100);
   bladeang_max = coCoviseConfig::getInt("Module.IHS.GateBladeMax",100);
   p_BladeAngle->setValue(bladeang_min, bladeang_max, 0.0);

}


void Gate::CreateMenuMeridianData(void)
{
   float *init = new float[2];
   // initialize init with default values
   init[0]=0.0;
   init[1]=0.0;

   p_InletHeight = addFloatParam(M_INLET_HEIGHT, M_INLET_HEIGHT);
   p_InletHeight->setValue(0.0);

   p_PivotRadius = addFloatParam(M_PIVOT_RADIUS, M_PIVOT_RADIUS);
   p_PivotRadius->setValue(1.0);

   p_InletRadius = addFloatParam(M_INLET_RADIUS, M_INLET_RADIUS);
   p_InletRadius->setValue(0.0);

   p_InletZ = addFloatParam(M_INLET_Z, M_INLET_Z);
   p_InletZ->setValue(0.0);

   p_OutletInnerRadius = addFloatParam(M_OUTLET_INNER_RADIUS, M_OUTLET_INNER_RADIUS);
   p_OutletInnerRadius->setValue(0.0);

   p_OutletOuterRadius = addFloatParam(M_OUTLET_OUTER_RADIUS, M_OUTLET_OUTER_RADIUS);
   p_OutletOuterRadius->setValue(0.0);

   p_OutletZ = addFloatParam(M_OUTLET_Z, M_OUTLET_Z);
   p_OutletZ->setValue(0.0);

   p_ShroudAB = addFloatVectorParam(M_SHROUD_RADIUS, M_SHROUD_RADIUS);
   p_ShroudAB->setValue(2, init);

   p_HubAB = addFloatVectorParam(M_HUB_RADIUS, M_HUB_RADIUS);
   p_HubAB->setValue(2, init);

   p_HubArcPoints = addInt32Param(M_HUB_ARC_POINTS, M_HUB_ARC_POINTS);
   p_HubArcPoints->setValue(0);

   delete[] init;
}


void Gate::CreateMenuProfileData(void)
{
   p_ChordLength = addFloatParam(M_CHORD_LENGTH, M_CHORD_LENGTH);
   p_ChordLength->setValue(0.0);

   p_PivotLocation = addFloatParam(M_CHORD_PIVOT, M_CHORD_PIVOT);
   p_PivotLocation->setValue(0.0);

   p_ChordAngle = addFloatParam(M_CHORD_ANGLE, M_CHORD_ANGLE);
   p_ChordAngle->setValue(0.0);

   p_ProfileThickness = addFloatParam(M_PROFILE_THICKNESS, M_PROFILE_THICKNESS);
   p_ProfileThickness->setValue(0.0);

   p_MaximumCamber = addFloatParam(M_MAXIMUM_CAMBER, M_MAXIMUM_CAMBER);
   p_MaximumCamber->setValue(0.0);

   p_ProfileShift = addFloatParam(M_PROFILE_SHIFT, M_PROFILE_SHIFT);
   p_ProfileShift->setValue(0.0);
}


void Gate::CreateMenuGatePNumbers()
{
   p_grid_n_rad = addInt32Param(M_N_RAD, M_N_RAD);
   p_grid_n_rad->setValue(0);

   p_grid_n_bound = addInt32Param(M_N_BOUND, M_N_BOUND);
   p_grid_n_bound->setValue(0);

   p_grid_n_out = addInt32Param(M_N_OUT, M_N_OUT);
   p_grid_n_out->setValue(0);

   p_grid_n_in = addInt32Param(M_N_IN, M_N_IN);
   p_grid_n_in->setValue(0);

   p_grid_n_blade_ps_back = addInt32Param(M_N_PS_BACK, M_N_PS_BACK);
   p_grid_n_blade_ps_back->setValue(0);

   p_grid_n_blade_ps_front = addInt32Param(M_N_PS_FRONT, M_N_PS_FRONT);
   p_grid_n_blade_ps_front->setValue(0);

   p_grid_n_blade_ss_back = addInt32Param(M_N_SS_BACK, M_N_SS_BACK);
   p_grid_n_blade_ss_back->setValue(0);

   p_grid_n_blade_ss_front = addInt32Param(M_N_SS_FRONT, M_N_SS_FRONT);
   p_grid_n_blade_ss_front->setValue(0);
}


void Gate::CreateMenuGateCurveLengths()
{
   p_grid_len_start_out_hub = addIntSliderParam(M_LEN_OUT_HUB, M_LEN_OUT_HUB);
   p_grid_len_start_out_hub->setValue(0, 100, 50);

   p_grid_len_start_out_shroud = addIntSliderParam(M_LEN_OUT_SHROUD, M_LEN_OUT_SHROUD);
   p_grid_len_start_out_shroud->setValue(0, 100, 50);

   p_grid_len_expand_in = addFloatParam(M_LEN_EXPAND_IN, M_LEN_EXPAND_IN);
   p_grid_len_expand_in->setValue(0.0);

   p_grid_len_expand_out = addFloatParam(M_LEN_EXPAND_OUT, M_LEN_EXPAND_OUT);
   p_grid_len_expand_out->setValue(0.0);

}


void Gate::CreateMenuGateCompressions()
{
   p_grid_comp_ps_back = addFloatParam(M_COMP_PS_BACK, M_COMP_PS_BACK);
   p_grid_comp_ps_back->setValue(0.0);

   p_grid_comp_ps_front = addFloatParam(M_COMP_PS_FRONT, M_COMP_PS_FRONT);
   p_grid_comp_ps_front->setValue(0.0);

   p_grid_comp_ss_back = addFloatParam(M_COMP_SS_BACK, M_COMP_SS_BACK);
   p_grid_comp_ss_back->setValue(0.0);

   p_grid_comp_ss_front = addFloatParam(M_COMP_SS_FRONT, M_COMP_SS_FRONT);
   p_grid_comp_ss_front->setValue(0.0);

   p_grid_comp_trail = addFloatParam(M_COMP_TRAIL, M_COMP_TRAIL);
   p_grid_comp_trail->setValue(0.0);

   p_grid_comp_out = addFloatParam(M_COMP_OUT, M_COMP_OUT);
   p_grid_comp_out->setValue(0.0);

   p_grid_comp_in = addFloatParam(M_COMP_IN, M_COMP_IN);
   p_grid_comp_in->setValue(0.0);

   p_grid_comp_bound = addFloatParam(M_COMP_BOUND, M_COMP_BOUND);
   p_grid_comp_bound->setValue(0.0);

   p_grid_comp_middle = addFloatParam(M_COMP_MIDDLE, M_COMP_MIDDLE);
   p_grid_comp_middle->setValue(0.0);

   p_grid_comp_rad = addFloatParam(M_COMP_RAD, M_COMP_RAD);
   p_grid_comp_rad->setValue(0.0);

}


void Gate::CreateMenuGateShifts()
{

   p_grid_shift_out = addFloatParam(M_SHIFT_OUT, M_SHIFT_OUT);
   p_grid_shift_out->setValue(0.0);

}


void Gate::Struct2CtrlPanel(void)
{
   // min/max values of parameters
   /*
      const float min_parm    =   0.0;
      const float max_parm    =   1.0;
   */

   fprintf(stderr, "Struct2CtrlPanel(): ...\n");

   p_GeoFromFile->setValue(geo->ga->geofromfile!=0);
   p_saveGrid->setValue(geo->ga->gr->savegrid!=0);

   //gate parameters
   p_radialGate->setValue(geo->ga->radial!=0);
   p_Q->setValue(geo->ga->Q);
   p_H->setValue(geo->ga->H);
   p_n->setValue(geo->ga->n);
   p_Q_opt->setValue(geo->ga->Qopt);
   p_n_opt->setValue(geo->ga->nopt);
   p_NumberOfBlades->setValue(geo->ga->nob);
   p_BladeAngle->setValue((float)(geo->ga->bangle*180/M_PI));
   p_PivotRadius->setValue(geo->ga->pivot_rad);

   //hub&shroud parameters
   p_InletHeight->setValue(geo->ga->in_height);
   p_InletRadius->setValue(geo->ga->in_rad);
   p_InletZ->setValue(geo->ga->in_z);
   p_OutletInnerRadius->setValue(geo->ga->out_rad1);
   p_OutletOuterRadius->setValue(geo->ga->out_rad2);
   p_OutletZ->setValue(geo->ga->out_z);
   p_ShroudAB->setValue(0, geo->ga->shroud_ab[0]);
   p_ShroudAB->setValue(1, geo->ga->shroud_ab[1]);
   p_HubAB->setValue(0, geo->ga->hub_ab[0]);
   p_HubAB->setValue(1, geo->ga->hub_ab[1]);
   p_HubArcPoints->setValue(geo->ga->num_hub_arc);

   //blade parameters
   p_ChordLength->setValue(geo->ga->chord);
   p_PivotLocation->setValue(geo->ga->pivot);
   p_ChordAngle->setValue((float)(geo->ga->angle*180/M_PI));
   p_ProfileThickness->setValue(geo->ga->p_thick);
   p_MaximumCamber->setValue(geo->ga->maxcamb);
   p_ProfileShift->setValue(geo->ga->bp_shift);

   //grid parameters
   p_grid_edge_ps->setValue(geo->ga->gr->edge_ps);
   p_grid_edge_ss->setValue(geo->ga->gr->edge_ss);
   p_grid_bound_layer->setValue(geo->ga->gr->bound_layer);
   p_grid_n_rad->setValue(geo->ga->gr->n_rad);
   p_grid_n_bound->setValue(geo->ga->gr->n_bound);
   p_grid_n_out->setValue(geo->ga->gr->n_out);
   p_grid_n_in->setValue(geo->ga->gr->n_in);
   p_grid_n_blade_ps_back->setValue(geo->ga->gr->n_blade_ps_back);
   p_grid_n_blade_ps_front->setValue(geo->ga->gr->n_blade_ps_front);
   p_grid_n_blade_ss_back->setValue(geo->ga->gr->n_blade_ss_back);
   p_grid_n_blade_ss_front->setValue(geo->ga->gr->n_blade_ss_front);
   p_grid_len_start_out_hub->setValue(geo->ga->gr->len_start_out_hub);
   p_grid_len_start_out_shroud->setValue(geo->ga->gr->len_start_out_shroud);
   p_grid_len_expand_in->setValue(geo->ga->gr->len_expand_in);
   p_grid_len_expand_out->setValue(geo->ga->gr->len_expand_out);
   p_grid_comp_ps_back->setValue(geo->ga->gr->comp_ps_back);
   p_grid_comp_ps_front->setValue(geo->ga->gr->comp_ps_front);
   p_grid_comp_ss_back->setValue(geo->ga->gr->comp_ss_back);
   p_grid_comp_ss_front->setValue(geo->ga->gr->comp_ss_front);
   p_grid_comp_trail->setValue(geo->ga->gr->comp_trail);
   p_grid_comp_out->setValue(geo->ga->gr->comp_out);
   p_grid_comp_in->setValue(geo->ga->gr->comp_in);
   p_grid_comp_bound->setValue(geo->ga->gr->comp_bound);
   p_grid_comp_middle->setValue(geo->ga->gr->comp_middle);
   p_grid_comp_rad->setValue(geo->ga->gr->comp_rad);
   p_grid_shift_out->setValue(geo->ga->gr->shift_out);

   fprintf(stderr, "Gate::Struct2CtrlPanel()... done\n");
}


void Gate::CtrlPanel2Struct(void)
{

   // read geometry from cfg-file
   geo->ga->geofromfile = p_GeoFromFile->getValue();

   // generate grid (yes / no) ?
   geo->ga->gr->savegrid = p_saveGrid->getValue();

   // gate parameters
   geo->ga->radial = p_radialGate->getValue();
   geo->ga->Q = p_Q->getValue();
   geo->ga->H = p_H->getValue();
   geo->ga->n = p_n->getValue();
   geo->ga->Qopt = p_Q_opt->getValue();
   geo->ga->nopt = p_n_opt->getValue();
   geo->ga->nob = p_NumberOfBlades->getValue();
   geo->ga->pivot_rad = p_PivotRadius->getValue();
   geo->ga->bangle = (float)(p_BladeAngle->getValue()*M_PI/180.0);

   // hub & shroud parameters
   geo->ga->in_height = p_InletHeight->getValue();
   geo->ga->in_rad = p_InletRadius->getValue();
   geo->ga->in_z = p_InletZ->getValue();
   geo->ga->out_rad1 = p_OutletInnerRadius->getValue();
   geo->ga->out_rad2 = p_OutletOuterRadius->getValue();
   geo->ga->out_z = p_OutletZ->getValue();
   geo->ga->shroud_ab[0] = p_ShroudAB->getValue(0);
   geo->ga->shroud_ab[1] = p_ShroudAB->getValue(1);
   geo->ga->hub_ab[0] = p_HubAB->getValue(0);
   geo->ga->hub_ab[1] = p_HubAB->getValue(1);
   geo->ga->num_hub_arc = p_HubArcPoints->getValue();

   // blade parameters
   geo->ga->chord = p_ChordLength->getValue();
   geo->ga->pivot = p_PivotLocation->getValue();
   geo->ga->angle = (float)(p_ChordAngle->getValue()*M_PI/180.0);
   geo->ga->p_thick = p_ProfileThickness->getValue();
   geo->ga->maxcamb = p_MaximumCamber->getValue();
   geo->ga->bp_shift = p_ProfileShift->getValue();

   //grid parameters
   geo->ga->gr->edge_ps = p_grid_edge_ps->getValue();
   geo->ga->gr->edge_ss = p_grid_edge_ss->getValue();
   geo->ga->gr->bound_layer = p_grid_bound_layer->getValue();
   geo->ga->gr->n_rad = p_grid_n_rad->getValue();
   geo->ga->gr->n_bound = p_grid_n_bound->getValue();
   geo->ga->gr->n_out = p_grid_n_out->getValue();
   geo->ga->gr->n_in = p_grid_n_in->getValue();
   geo->ga->gr->n_blade_ps_back = p_grid_n_blade_ps_back->getValue();
   geo->ga->gr->n_blade_ps_front = p_grid_n_blade_ps_front->getValue();
   geo->ga->gr->n_blade_ss_back = p_grid_n_blade_ss_back->getValue();
   geo->ga->gr->n_blade_ss_front = p_grid_n_blade_ss_front->getValue();
   geo->ga->gr->len_start_out_hub = p_grid_len_start_out_hub->getValue();
   geo->ga->gr->len_start_out_shroud = p_grid_len_start_out_shroud->getValue();
   geo->ga->gr->len_expand_in = p_grid_len_expand_in->getValue();
   geo->ga->gr->len_expand_out = p_grid_len_expand_out->getValue();
   geo->ga->gr->comp_ps_back = p_grid_comp_ps_back->getValue();
   geo->ga->gr->comp_ps_front = p_grid_comp_ps_front->getValue();
   geo->ga->gr->comp_ss_back = p_grid_comp_ss_back->getValue();
   geo->ga->gr->comp_ss_front = p_grid_comp_ss_front->getValue();
   geo->ga->gr->comp_trail = p_grid_comp_trail->getValue();
   geo->ga->gr->comp_out = p_grid_comp_out->getValue();
   geo->ga->gr->comp_in = p_grid_comp_in->getValue();
   geo->ga->gr->comp_bound = p_grid_comp_bound->getValue();
   geo->ga->gr->comp_middle = p_grid_comp_middle->getValue();
   geo->ga->gr->comp_rad = p_grid_comp_rad->getValue();
   geo->ga->gr->shift_out = p_grid_shift_out->getValue();

}


int Gate::CheckUserInput(const char *portname, struct geometry *geo)
{
   int changed;

   changed = 0;
   fprintf(stderr, "Gate::CheckUserInput() entering ..., pn=%s\n", portname);

   // contour parameters
   if (!strcmp(M_INLET_HEIGHT, portname))
   {
      changed = CheckUserFloatValue(p_InletHeight, geo->ga->in_height,
         0.0, 10.0, &(geo->ga->in_height) );
   }
   if (!strcmp(M_INLET_RADIUS, portname))
   {
      changed = CheckUserFloatValue(p_InletRadius, geo->ga->in_rad,
         geo->ga->out_rad2 + geo->ga->shroud_ab[0], 10.0, &(geo->ga->in_rad) );
   }
   if (!strcmp(M_INLET_Z, portname))
   {
      changed = CheckUserFloatValue(p_InletZ, geo->ga->in_z,
         -100., 100., &(geo->ga->in_z) );
   }
   if (!strcmp(M_OUTLET_INNER_RADIUS, portname))
   {
      changed = CheckUserFloatValue(p_OutletInnerRadius, geo->ga->out_rad1,
         0.0, 5.0, &(geo->ga->out_rad1) );
   }
   if (!strcmp(M_OUTLET_OUTER_RADIUS, portname))
   {
      changed = CheckUserFloatValue(p_OutletOuterRadius, geo->ga->out_rad2,
         geo->ga->out_rad1, 10.0, &(geo->ga->out_rad2) );
   }
   if (!strcmp(M_OUTLET_Z, portname))
   {
      changed = CheckUserFloatValue(p_OutletZ, geo->ga->out_z,
         -100.0, geo->ga->in_z, &(geo->ga->out_z) );
   }
   if (!strcmp(M_SHROUD_RADIUS, portname))
   {
      changed = CheckUserFloatVectorValue(p_ShroudAB, 0, geo->ga->shroud_ab[0],
         0.0, geo->ga->in_rad - geo->ga->out_rad2, &(geo->ga->shroud_ab[0]) );
   }
   if (!strcmp(M_SHROUD_RADIUS, portname))
   {
      changed = CheckUserFloatVectorValue(p_ShroudAB, 1, geo->ga->shroud_ab[1],
         0.0, geo->ga->in_z - geo->ga->out_z, &(geo->ga->shroud_ab[1]) );
   }
   if (!strcmp(M_HUB_RADIUS, portname))
   {
      changed = CheckUserFloatVectorValue(p_HubAB, 0, geo->ga->hub_ab[0],
         0.0, geo->ga->in_rad - geo->ga->out_rad1, &(geo->ga->hub_ab[0]) );
   }
   if (!strcmp(M_HUB_RADIUS, portname))
   {
      changed = CheckUserFloatVectorValue(p_HubAB, 1, geo->ga->hub_ab[1],
         0.0, geo->ga->in_z + geo->ga->in_height - geo->ga->out_z,
         &(geo->ga->hub_ab[1]) );
   }
   if (!strcmp(M_HUB_ARC_POINTS, portname))
   {
      changed = CheckUserIntValue(p_HubArcPoints, geo->ga->num_hub_arc,
         2, 20, &(geo->ga->num_hub_arc) );
   }

   // gate parameters
   if (!strcmp(M_Q, portname))
   {
      changed = CheckUserFloatSliderValue(p_Q, geo->ga->Q,
         0.0, 500.0, &(geo->ga->Q) );
   }
   if (!strcmp(M_H, portname))
   {
      changed = CheckUserFloatSliderValue(p_H, geo->ga->H,
         0.0, 3000.0, &(geo->ga->H) );
   }
   if (!strcmp(M_N, portname))
   {
      changed = CheckUserFloatSliderValue(p_n, geo->ga->n,
         0.0, 2000.0, &(geo->ga->n) );
   }
   if (!strcmp(M_Q_OPT, portname))
   {
      changed = CheckUserFloatSliderValue(p_Q_opt, geo->ga->Qopt,
         0.0, 500.0, &(geo->ga->Qopt) );
   }
   if (!strcmp(M_N_OPT, portname))
   {
      changed = CheckUserFloatSliderValue(p_n_opt, geo->ga->nopt,
         0.0, 2000.0, &(geo->ga->nopt) );
   }
   if (!strcmp(M_NUMBER_OF_BLADES, portname))
   {
      changed = CheckUserIntValue(p_NumberOfBlades, geo->ga->nob,
         5, 30, &(geo->ga->nob) );
   }
   if (!strcmp(M_PIVOT_RADIUS, portname))
   {
      changed = CheckUserFloatValue(p_PivotRadius, geo->ga->pivot_rad,
         geo->ga->out_rad2 + geo->ga->shroud_ab[0], geo->ga->in_rad, &(geo->ga->pivot_rad) );
   }

   // blade parameters
   if (!strcmp(M_CHORD_PIVOT, portname))
   {
      changed = CheckUserFloatValue(p_PivotLocation, geo->ga->pivot,
         0.0, geo->ga->chord, &(geo->ga->pivot) );
   }

   // grid parameters
   // check to be filled in here

   return changed;

}

#ifndef YAC
coDistributedObject *Gate::GenerateNormals(int part, coDoPolygons *poly, const char *out_name)
#else
coDistributedObject *Gate::GenerateNormals(int part, coDoPolygons *poly, coObjInfo out_name)
#endif
{
   coDoVec3 *normals;

   float l,x1,x2,y1,y2,z1,z2;
   float *x,*y,*z,*U,*V,*W,*NU,*NV,*NW,*F_Normals_U,*F_Normals_V,*F_Normals_W;
   int i,n,num_n,n0,n1,n2;
   int *vl,*pl;
   int *nl,*nli,numpoly,numcoord;

   numpoly=poly->getNumPolygons();
   int numvert=poly->getNumVertices();
   numcoord=poly->getNumPoints();

   poly->getAddresses(&x,&y,&z,&vl,&pl);
   poly->getNeighborList(&num_n,&nl,&nli);

   normals = new coDoVec3(out_name, numcoord);
   normals->getAddresses(&NU,&NV,&NW);

   U=F_Normals_U = new float[numpoly];
   V=F_Normals_V = new float[numpoly];
   W=F_Normals_W = new float[numpoly];

   for(i=0;i<numpoly;i++)
   {
      // find out number of corners
      int no_corners;
      if(i<numpoly-1)
      {
         no_corners = pl[i+1] - pl[i];
      }
      else
      {
         no_corners = numvert - pl[i];
      }
      if (no_corners!=3)
      {
         sendError("sorry, this is only working with triangles!");
         return(NULL);
      }

      l=0.0;

      n0=vl[pl[i]];
      n1=vl[pl[i]+1];
      n2=vl[pl[i]+2];

      x1=x[n1]-x[n0];
      y1=y[n1]-y[n0];
      z1=z[n1]-z[n0];
      x2=x[n2]-x[n0];
      y2=y[n2]-y[n0];
      z2=z[n2]-z[n0];
      *U=y1*z2-y2*z1;
      *V=x2*z1-x1*z2;
      *W=x1*y2-x2*y1;
      l=sqrt(*U * *U+*V * *V+*W * *W);

      if(l!=0.0)
      {
         *U/=l;
         *V/=l;
         *W/=l;
      }
      else
      {
         *U=0.0;
         *V=0.0;
         *W=0.0;
      }
      U++;
      V++;
      W++;
   }

   for(i=0;i<numcoord;i++)
   {
      *NU=*NV=*NW=0;
      for(n=nli[i];n<nli[i+1];n++)
      {
         *NU+=F_Normals_U[nl[n]];
         *NV+=F_Normals_V[nl[n]];
         *NW+=F_Normals_W[nl[n]];
      }
      float l=sqrt(*NU * *NU+*NV * *NV+*NW * *NW);
      if(l>0.0)
      {
         *NU /= l;
         *NV /= l;
         *NW /= l;
      }
      NU++;
      NV++;
      NW++;
   }

   if (part==BLADE)
      bladenormals->setCurrentObject(normals);
   if (part==HUB)
      hubnormals->setCurrentObject(normals);
   if (part==SHROUD)
      shroudnormals->setCurrentObject(normals);

   delete[] F_Normals_U;
   delete[] F_Normals_V;
   delete[] F_Normals_W;

   return(0);
}

#ifdef YAC
void Gate::paramChanged(coParam *param) {

   this->param(param->getName(), false);
}
#endif

MODULE_MAIN(VISiT, Gate)
