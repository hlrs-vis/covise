// rei, Die Sep 14 08:58:24 MEST 1999
#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <api/coFeedback.h>
#include <do/coDoSet.h>
#include <do/coDoIntArr.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include "DraftTube.h"
#include <General/include/geo.h>
#include <General/include/log.h>
#include <DraftTube/include/tube.h>
#include <DraftTube/include/tgrid.h>
#include <General/include/cov.h>
#include <General/include/CreateFileNameParam.h>

DraftTube::DraftTube(int argc, char *argv[])
: coModule(argc, argv, "Geometry Generator")
{
   //char buf[256];
   char *pfn;

   // loglevel and debug files ...
   SetDebugPath(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(),getenv(ENV_IHS_DEBPATH));
   SetDebugLevel(0);
   if (getenv(ENV_IHS_DEBUGLEVEL))
   {
      SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
   }
   else
      dprintf(0, "WARNING: %s is not set. (now setting to 0)\n",
         ENV_IHS_DEBUGLEVEL);
   if ((pfn = CreateFileNameParam(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(), ENV_IHS_DEBPATH, coCoviseConfig::getEntry("value","Module.IHS.DebFile","DraftTube.deb").c_str(), CFNP_NORM)) != NULL)
   {
      fprintf(stderr, "### pfn=%s\n", pfn);
      dopen(pfn);
      free(pfn);
   }
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**                                                                  **\n");
   dprintf(0, "** %-64.64s **\n", "Draft-Tube module");
   dprintf(0, "** %-64.64s **\n", "(c) 1999-2003 by University of Stuttgart - IHS");
   dprintf(0, "**                                                                  **\n");
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**************************************************\n");

   paramFile = addFileBrowserParam(PARAM_FILE, PARAM_FILE);


   std::string dataPath; 
#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE");
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(defaultDir)
      dataPath=coCoviseConfig::getEntry("value","Module.IHS.DataPath",defaultDir);
   else
      dataPath=coCoviseConfig::getEntry("value","Module.IHS.DataPath","/data/IHS");

   if ((pfn = CreateFileNameParam(dataPath.c_str(), "IHS_DATAPATH", "nofile", CFNP_NORM)) != NULL)
   {
      dprintf(0, "Startpath: %s\n", pfn);
      paramFile->setValue(pfn,"*.cfg");
      free(pfn);
   }
   else
   {
      dprintf(0, "WARNING: pfn (%s, %d) ist NULL !\n", __FILE__, __LINE__);
   }

   // first selection: GEOMETRY || GRID ??
   paraSwitch("paraType", "Select type of parameters");
   paraCase(P_GEOMETRY);
   paraSwitch("Geometry", "Geometry parameters");
   paraCase(P_GEOMETRYCS);
   CreateMenu_GeometryCrossSection();
   paraEndCase();

   paraCase(P_GEOMETRYCSA);
   CreateMenu_AreaCrossSection();
   paraEndCase();
   paraEndSwitch();                               // ("Geometry", "Geometry parameters")
   paraEndCase();

   paraCase(P_GRID);
   CreateMenu_GridCrossSection();
   paraEndCase();
   paraEndSwitch();                               // "paraType", "Select type of parameters"

   // the output ports
   surf  = addOutputPort("surface","Polygons","Surface Polygons");
   cross = addOutputPort("cross","Polygons","Cross Section Polygons");
   grid  = addOutputPort("grid","UnstructuredGrid","Computation grid");
   boco  = addOutputPort("boco","USR_FenflossBoco","Boundary Conditions");
   bc_in = addOutputPort("bc_in","Polygons","Cells at entry");

   dprintf(1, "DraftTube::DraftTube() done\n");
}


void DraftTube::postInst()
{
   dprintf(1, "DraftTube::postInst() done\n");
}


void DraftTube::param(const char *portname, bool inMapLoading)
{
   int i, j;
   char buf[255];

   dprintf(1, "DraftTube::param(%s): START\n", portname);
   if (strcmp(portname, PARAM_FILE)==0)
   {
      float Amin, Amax;

      if (geo && geo->tu->cs_num)
      {
         sendError("We Had an input file before...");
         dprintf(1, "DraftTube::param(): END, line=%d\n", __LINE__);
         return;
      }

      Covise::getname(buf,paramFile->getValue());
      geo = ReadGeometry(buf);
      if (!geo)
      {
         return;
      }
      DumpTube(geo->tu);

      // Initialisation of the CtrlPanel
      for (j = 0; j < 4; j++)
         p_numi[j]->setValue(geo->tu->c_el[j]);
      p_numo->setValue(geo->tu->c_el_o);

      for (i=0; i<geo->tu->cs_num; i++)
      {
         p_elem[i]->setValue(geo->tu->cs[i]->c_nume);
         for (j = 0; j < 8; j++)
         {
            p_part[i][j]->setValue(geo->tu->cs[i]->c_part[j]);
         }

         for (j = 0; j < 4; j++)
         {
            p_ab[i][j]->setValue(0, geo->tu->cs[i]->c_a[j]);
            p_ab[i][j]->setValue(1, geo->tu->cs[i]->c_b[j]);
         }
         p_m[i]->setValue(geo->tu->cs[i]->c_m_x, geo->tu->cs[i]->c_m_y,
            geo->tu->cs[i]->c_m_z);
         p_hw[i]->setValue(0, geo->tu->cs[i]->c_height);
         p_hw[i]->setValue(1, geo->tu->cs[i]->c_width);
      }

      // now we setup the area-Menu ...
      CalcValuesForCSArea();
      Amin = p_cs_area[0]->getValue() * 0.6f;
      Amax = p_cs_area[MAX_CROSS-1]->getValue() * 1.4f;
      for (i = 0; i< geo->tu->cs_num; i++)
      {
         p_cs_area[i]->setMin(Amin);
         p_cs_area[i]->setMax(Amax);
      }

      dprintf(2, "DraftTube::param(): hiding vs from %d until %d\n", geo->tu->cs_num, MAX_CROSS);
      for (i = geo->tu->cs_num; i < MAX_CROSS; i++)
      {
         p_cs_area[i]->disable();
         p_cs_area[i]->hide();
      }
      dprintf(1, "DraftTube::param(): END, line=%d\n", __LINE__);
      return;
   }
   // ------ PushButton for Grid Generation
   if ( !strcmp(portname,p_makeGrid->getName()) && !inMapLoading )
   {
      if (!p_makeGrid->getValue()) return;

      // push in : create Grids in compute() Call-Back
      selfExec();
   }
   else if ( !strcmp(portname,p_ip_start->getName()) && !inMapLoading )
   {
      if (!p_ip_start->getValue()) return;

      // push in : interpolate in compute() Call-Back
      selfExec();
   }
   if (!inMapLoading)
      CheckUserInput(portname, geo);
   dprintf(1, "DraftTube::param(): END, line=%d\n", __LINE__);
}


void DraftTube::quit()
{

   dprintf(1, "application->quit()\n");

}


int DraftTube::compute(const char *)
{
   coDoPolygons *poly;
   struct covise_info *ci;
   struct tgrid *tg;
   char name[256];
   int list[1];
   const char *basename;
   int i, j;

   dprintf(1, "Now entering DraftTube::compute\n");

   if (!geo)
   {
      sendError("Please select first a parmeter file !!");
      return FAIL;
   }
   /////////////////////////////
   // First check, wether we have to interpolate ...
   if (p_ip_start->getValue())
   {
      p_ip_start->setValue(0);                    // push off button
      InterpolateAreas();
      CalcValuesForCSArea();
   }

   /////////////////////////////
   // Copy values from COVISE menu to geometry struct ...
   for (i=0; i<geo->tu->cs_num; i++)
   {
      p_m[i]->getValue(geo->tu->cs[i]->c_m_x,
         geo->tu->cs[i]->c_m_y, geo->tu->cs[i]->c_m_z);
      geo->tu->cs[i]->c_height = p_hw[i]->getValue(0);
      geo->tu->cs[i]->c_width  = p_hw[i]->getValue(1);
      for (j = 0; j < 4; j++)
      {
         geo->tu->cs[i]->c_a[j] = p_ab[i][j]->getValue(0);
         geo->tu->cs[i]->c_b[j] = p_ab[i][j]->getValue(1);
      }
   }

   Covise::getname(name,paramFile->getValue());
   strcat(name, ".new");
   WriteGeometry(geo, name);

   /////////////////////////////
   // create geometry for COVISE
   if ((ci = CreateGeometry4Covise(geo)))
   {

      dprintf(1, "DraftTube: Geometry created\n");
      // geometry surface ...
      poly = new coDoPolygons(surf->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->vx->num,  ci->vx->list,
         ci->pol->num, ci->pol->list);
      if (!poly)
      {
         dprintf(1, "Creation of coDoPolygons *poly failed (%s, %d)\n", __FILE__, __LINE__);
         exit(10);
      }
      //poly->addAttribute("MATERIAL","metal metal.30");
      //poly->addAttribute("vertexOrder","1");
      surf->setCurrentObject(poly);

      // cross sections ...
      dprintf(2, "Start of creating coDistributedObject *CrossSection[MAX_CROSS+1] ...\n");
      coDistributedObject *CrossSection[MAX_CROSS+1];
      for (i = 0; i < ci->num_cs; i++)
      {
         sprintf(name, "cs_%d", i);
         dprintf(3, "\tname=%s\n", name);
         list[0] = 0;
         CrossSection[i] = new coDoPolygons(name,
            ci->ci_cs[i]->p->nump,
            ci->ci_cs[i]->p->x, ci->ci_cs[i]->p->y, ci->ci_cs[i]->p->z,
            ci->ci_cs[i]->cvx->num, ci->ci_cs[i]->cvx->list,
            1, list);
         CrossSection[i]->addAttribute("vertexOrder","1");
         CrossSection[i]->addAttribute("COLOR","blue");

         // some stuff for 3d
         coFeedback feedback("DraftTubePlugin");
         feedback.addString(name);
         feedback.addPara(p_hw[i]);
         feedback.addPara(p_m[i]);
         feedback.addPara(p_angle[i]);
         feedback.addPara(p_angletype[i]);
         for (j = 0; j < 4; j++)
            feedback.addPara(p_ab[i][j]);
         feedback.apply(CrossSection[i]);
      }
      CrossSection[ci->num_cs] = NULL;

      basename    = cross->getObjName();
      coDoSet *set = new coDoSet((char*)basename,(coDistributedObject **)CrossSection);
                                                  // argh, schrott-Programmierung
      char *finfo = (char *)calloc(3000, sizeof(char));
      sprintf(name, "NumCS=%d;", ci->num_cs);
      strcpy(finfo, name);
      for (i = 0; i < ci->num_cs; i++)
      {
         float f[3];
         char buf[100];

         p_m[i]->getValue(f[0], f[1], f[2]);
         sprintf(buf, "%f:%f:%f;", f[0], f[1], f[2]);
         strcat(finfo, buf);
      }
      coFeedback feedback("DraftTubePlugin");
      feedback.addString(finfo);
      feedback.addPara(p_makeGrid);               // start of computation
      feedback.apply(set);
      cross->setCurrentObject(set);
      dprintf(2, "END of creating coDistributedObject *CrossSection[MAX_CROSS+1] ...\n");

      dprintf(2, "Setze bc_in_po");
      poly = new coDoPolygons(bc_in->getObjName(),
         ci->bcinnumPoints,
         ci->p->x, ci->p->y, ci->p->z,
         ci->bcinvx->num, ci->bcinvx->list,
         ci->bcinpol->num, ci->bcinpol->list);
      poly->addAttribute("vertexOrder","1");
      poly->addAttribute("COLOR","green");
      bc_in->setCurrentObject(poly);

   }
   else
   {
      dprintf(0, "Error in CreateGeometry4Covise (%s, %d)\n", __FILE__, __LINE__);
   }

   //////////////// This creates the volume grid ////////////////////

   ////////////////////////
   // if button is pushed --> create computation grid
   if (p_makeGrid->getValue())
   {
      int size[2];

      p_makeGrid->setValue(0);                    // push off button

      tg = CreateTGrid(geo->tu);
      dprintf(2, "DraftTube: Grid created\n");
      WriteTGrid(tg, "deb_gg");
      //WriteTBoundaryConditions(tg, "deb_gg");

      coDoUnstructuredGrid *unsGrd =
         new coDoUnstructuredGrid(grid->getObjName(),
         tg->e->nume,8*tg->e->nume,tg->p->nump,1);

      int *elem,*conn,*type;
      float *xc,*yc,*zc;
      unsGrd->getAddresses(&elem,&conn,&xc,&yc,&zc);
      unsGrd->getTypeList(&type);

      int **TgridConn = tg->e->e;
      for (i=0;i<tg->e->nume;i++)
      {
         *elem = 8*i;               elem++;

         *conn = (*TgridConn)[0];   conn++;
         *conn = (*TgridConn)[1];   conn++;
         *conn = (*TgridConn)[2];   conn++;
         *conn = (*TgridConn)[3];   conn++;
         *conn = (*TgridConn)[4];   conn++;
         *conn = (*TgridConn)[5];   conn++;
         *conn = (*TgridConn)[6];   conn++;
         *conn = (*TgridConn)[7];   conn++;

         *type = TYPE_HEXAGON;      type++;

         TgridConn++;

      }

      memcpy(xc,tg->p->x,tg->p->nump*sizeof(float));
      memcpy(yc,tg->p->y,tg->p->nump*sizeof(float));
      memcpy(zc,tg->p->z,tg->p->nump*sizeof(float));

      grid->setCurrentObject(unsGrd);

      // we had several additional info, we should send to the
      // Domaindecomposition:
      //   0. number of columns per info
      //   1. type of node
      //   2. type of element
      //   3. list of nodes with bc (a node may appear more than one time)
      //   4. corresponding type to 3.
      //   5. wall
      //   6. balance
      //   7. NULL

      coDistributedObject *partObj[8];
      int *data;
      float *bPtr;
      const char *basename = boco->getObjName();

      //   0. number of columns per info
      sprintf(name,"%s_colinfo",basename);
      size[0] = 5;
      size[1] = 0;
      coDoIntArr *colInfo = new coDoIntArr(name,1,size);
      data = colInfo->getAddress();
      data[0] = TG_COL_NODE;
      data[1] = TG_COL_ELEM;
      data[2] = TG_COL_DIRICLET;
      data[3] = TG_COL_WALL;
      data[4] = TG_COL_BALANCE;
      partObj[0]=colInfo;

      //   1. type of node
      sprintf(name,"%s_nodeinfo",basename);
      size[0] = TG_COL_NODE;
      size[1] = tg->p->nump;
      coDoIntArr *nodeInfo = new coDoIntArr(name,2,size);
      data = nodeInfo->getAddress();
      for (i=0;i<tg->p->nump;i++)
      {
         *data++ = i+1;                           // may be, that we later do it really correct
         *data++ = 0;                             // same comment ;-)
      }
      partObj[1]=nodeInfo;

      //   2. type of element
      sprintf(name,"%s_eleminfo",basename);
      size[0] = 2;
      size[1] = tg->e->nume*TG_COL_ELEM;
      coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
      data = elemInfo->getAddress();
      for (i=0;i<tg->p->nump;i++)
      {
         *data++ = i+1;                           // may be, that we later do it really corect
         *data++ = 0;                             // same comment ;-)
      }
      partObj[2]=elemInfo;

      //   3. list of nodes with bc (a node may appear more than one time)
      //      and its types
      sprintf(name,"%s_diricletNodes",basename);
      size [0] = TG_COL_DIRICLET;
      size [1] = 6*tg->gs[0]->p->nump;
      coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
      data = diricletNodes->getAddress();

      //   4. coreesponding value to 3.
      sprintf(name,"%s_diricletValue",basename);
      coDoFloat *diricletValues
         = new coDoFloat(name, 6*tg->gs[0]->p->nump);
      diricletValues->getAddress(&bPtr);

      for (i=0;i<tg->gs[0]->p->nump;i++)
      {
         *data++ = i + 1;                         // node-number
         *data++ = 1;                             // type of node
         *bPtr++ = 0.0;                           // u
         *data++ = i + 1;                         // node-number
         *data++ = 2;                             // type of node
         *bPtr++ = 0.0;                           // v
         *data++ = i + 1;                         // node-number
         *data++ = 3;                             // type of node
         *bPtr++ = -1.0;                          // w
         *data++ = i + 1;                         // node-number
         *data++ = 4;                             // type of node
         *bPtr++ = tg->epsilon;                   // epsilon
         *data++ = i + 1;                         // node-number
         *data++ = 5;                             // type of node
         *bPtr++ = tg->k;                         // k
         *data++ = i + 1;                         // node-number
         *data++ = 6;                             // type of node
         *bPtr++ = tg->T;                         // temperature
      }

      partObj[3] = diricletNodes;
      partObj[4] = diricletValues;

      //   5. wall
      sprintf(name,"%s_wall",basename);
      size[0] = TG_COL_WALL;
      size[1] = tg->wall->numv;
      coDoIntArr *faces = new coDoIntArr(name, 2, size );
      data=faces->getAddress();
      for (i=0;i<tg->wall->numv;i++)
      {
         *data++ = tg->wall->v[i][0]+1;
         *data++ = tg->wall->v[i][1]+1;
         *data++ = tg->wall->v[i][2]+1;
         *data++ = tg->wall->v[i][3]+1;
         *data++ = tg->wall->v[i][4]+1;
         *data++ = tg->bc_wall;                   // wall: moving|standing
      }
      partObj[5]=faces;

      //   6. balance
      sprintf(name,"%s_balance",basename);
      size[0] = TG_COL_BALANCE;
      size[1] = tg->in->numv + tg->out->numv;
      coDoIntArr *balance = new coDoIntArr(name, 2, size );
      data=balance->getAddress();
      for (i=0;i<tg->in->numv;i++)
      {
         *data++ = tg->in->v[i][0]+1;
         *data++ = tg->in->v[i][1]+1;
         *data++ = tg->in->v[i][2]+1;
         *data++ = tg->in->v[i][3]+1;
         *data++ = tg->in->v[i][4]+1;
         *data++ = tg->bc_inval;
      }
      for (i=0;i<tg->out->numv;i++)
      {
         *data++ = tg->out->v[i][0]+1;
         *data++ = tg->out->v[i][1]+1;
         *data++ = tg->out->v[i][2]+1;
         *data++ = tg->out->v[i][3]+1;
         *data++ = tg->out->v[i][4]+1;
         *data++ = tg->bc_outval;
      }
      partObj[6]=balance;

      partObj[7]=NULL;

      coDoSet *set = new coDoSet((char*)basename,(coDistributedObject **)partObj);
      for (i=0;i<4;i++)                           // rei 27.10.2001 ????????? strange ...
         delete partObj[i];

      boco->setCurrentObject(set);

      ///////////////////////// Free everything ////////////////////////////////
      FreeStructTGrid(tg);
   }
   return SUCCESS;
}


void DraftTube::CreateMenu_GeometryCrossSection()
{
   int i, j;
   char buf[255];
   char *tmp;
   float vec[3];

   vec[0] = vec[1] = vec[2] = -1;

   // sync mode for all a, b ??
   p_absync = addBooleanParam(P_ABSYNC, P_ABSYNC);
   p_absync->setValue(0);

   m_GeometryCrossSection = paraSwitch("GeCrossSection", "Select GeCross section");
   cs_labels = (char **) calloc(MAX_CROSS, sizeof(char*));
   for (i=0;i<MAX_CROSS;i++)
   {
      // create description and name
      cs_labels[i] = IndexedParameterName(GEO_SEC, i);
      paraCase(cs_labels[i]);                     // Geometry section

      // x, y, z - ccordinate of this CS
      tmp = IndexedParameterName(P_M, i);
      p_m[i] = addFloatVectorParam(tmp, tmp);
      p_m[i]->setValue(0.0,0.0,0.0);
      free(tmp);

      // Height, Width
      tmp = IndexedParameterName(P_HW, i);
      p_hw[i] = addFloatVectorParam(tmp, tmp);
      p_hw[i]->setValue(2, vec);
      free(tmp);

      // Corners AB
      for (j = 0; j < 4; j++)
      {
         // WARNING: Don't changed the string in buf
         // it is used in CheckUserInput.cpp
         sprintf(buf, "%s_%s",P_AB, direction[j]);
         tmp = IndexedParameterName(buf, i);
         p_ab[i][j] = addFloatVectorParam(tmp, tmp);
         p_ab[i][j]->setValue(2, vec);
         free(tmp);
      }

      // Angletype
      tmp = IndexedParameterName(P_ANGLETYPE, i);
      p_angletype[i] = addBooleanParam(tmp, tmp);
      p_angletype[i]->setValue(0);
      free(tmp);

      // Angle
      tmp = IndexedParameterName(P_ANGLE, i);
      p_angle[i] = addFloatParam(tmp, tmp);
      p_angle[i]->setValue(0.0);
      free(tmp);

      paraEndCase();                              // Geometry section
   }
   paraEndSwitch();                               // "GeCross section", "Select GeCross section"
}


void DraftTube::CreateMenu_AreaCrossSection()
{
   int i;
   char *tmp;
   char *selections[NUM_INTERPOL+1];

   dprintf(1, "DraftTube::CreateMenu_AreaCrossSection(): START\n");
   dprintf(5, "MAXCROSS=%d\n", MAX_CROSS);
   for (i=0;i<MAX_CROSS;i++)
   {
      // Areas of the cross sections
      tmp = IndexedParameterName(P_CS_AREA, i);
      dprintf(3, "\ti=%d(MAX_CROSS=%d), tmp=%s\n", i, MAX_CROSS, tmp);
      p_cs_area[i] = addFloatSliderParam(tmp, tmp);
      free(tmp);
   }

   // Selection List for Interpolation type
   p_ip_type = addChoiceParam(P_IP_TYPE, "Type of interpolation");
   selections[0] = strdup("Linear");
   selections[1] = NULL;
   p_ip_type->setValue(NUM_INTERPOL, selections, 0);

   // From at Cross Section ...
   p_ip_S = addInt32Param(P_IP_S, "From Cross Section");
   p_ip_S->setValue(-1);

   // to Cross Section ...
   p_ip_E = addInt32Param(P_IP_E, "Until Cross Section");
   p_ip_E->setValue(-1);

   // interpolation of a, b ???
   p_ip_ab = addBooleanParam(P_IP_AB,"Interpolate a, b");
   p_ip_ab->setValue(0);

   // interpolation of height ??
   p_ip_height = addBooleanParam(P_IP_HEIGHT,"Interpolate height");
   p_ip_height->setValue(0);

   // interpolation of width ??
   p_ip_width = addBooleanParam(P_IP_WIDTH,"Interpolate width");
   p_ip_width->setValue(0);

   // interpolation will start, when button is pushed ...
   p_ip_start = addBooleanParam(P_IP_START,"Start interpolation now");
   p_ip_start->setValue(0);
   dprintf(1, "DraftTube::CreateMenu_AreaCrossSection(): END\n");
}


void DraftTube::CreateMenu_GridCrossSection(void)
{
   int i,j;
   char buf[255];

   // Grid will be created, when button is pushed ...
   p_makeGrid = addBooleanParam("makeGrid","Make a grid now");
   p_makeGrid->setValue(0);

   // num of gridpoints in vertical direction on the right side
   strcpy(buf,"G_Points_RightVertical");
   p_numi[0] = addInt32Param(buf,buf);
   p_numi[0]->setValue(0);

   // num of gridpoints in horicontal direction on the top
   strcpy(buf,"G_Points_TopHoricontal");
   p_numi[1] = addInt32Param(buf,buf);
   p_numi[1]->setValue(0);

   // num of gridpoints in vertical direction on the left side
   strcpy(buf,"G_Points_LeftVertical");
   p_numi[2] = addInt32Param(buf,buf);
   p_numi[2]->setValue(0);

   // num of gridpoints in vertical direction on the bottom
   strcpy(buf,"G_Points_BottomHoricontal");
   p_numi[3] = addInt32Param(buf,buf);
   p_numi[3]->setValue(0);

   // number of the outer elements
   strcpy(buf,"NumberOfOuterElements");
   p_numo = addInt32Param(buf,buf);
   p_numo->setValue(0);

   paraSwitch("GrCrossSection", "Select GrCross section");
   for (i=0;i<MAX_CROSS;i++)
   {
      //  create description and name
      sprintf(buf,"GridSection_%d",i+1);
      paraCase(buf);

      // number of elements in this section
      sprintf(buf,"NumberOfElements_%d",i+1);
      p_elem[i] = addInt32Param(buf,buf);
      p_elem[i]->setValue(0);

      for (j=0;j<8;j++)
      {
         // number of elements in this section
         sprintf(buf,"OuterPart_%s_%d", sectornames[j], i+1);
         p_part[i][j] = addFloatParam(buf,buf);
         p_part[i][j]->setValue(0.0);
      }
      paraEndCase();                              // buf
   }
   paraEndSwitch();                               // "GrCross section", "Select GrCross section"
}


void DraftTube::CalcValuesForCSArea()
{
   int i;
   float min, max;
   float setVal[MAX_CROSS];

   min = p_cs_area[0]->getValue() * 0.6f;
   max = p_cs_area[MAX_CROSS-1]->getValue() * 1.4f;
   for (i = 0; i< MAX_CROSS; i++)
   {
      setVal[i] = CalcOneArea(i);
      if (min > setVal[i])
         min = setVal[i];
      if (max < setVal[i])
         max = setVal[i];
   }
   min *= 0.6f;
   max *= 1.4f;
   for (i = 0; i< MAX_CROSS; i++)
   {
      if (setVal[i] > 0.0)
         p_cs_area[i]->setValue(min, max, setVal[i]);
      else
         p_cs_area[i]->setValue(min, max, 0.0);
   }
}


float DraftTube::CalcOneArea(int ind)
{
   float A, a, b;
   int j;

   if (p_hw[ind]->getValue(0) <= 0.0 || p_hw[ind]->getValue(1) <= 0.0)
      A = -1.0;
   else
   {
      A = p_hw[ind]->getValue(0) * p_hw[ind]->getValue(1);
      for (j = 0; j < 4; j++)
      {
         a = p_ab[ind][j]->getValue(0);
         b = p_ab[ind][j]->getValue(1);
         A -= a*b;                                // Sub the corner rectangle
         A += (float)(M_PI * a * b / 4.0);                   // Add the part of the ellipse
      }
   }
   return A;
}

MODULE_MAIN(VISiT, DraftTube)
