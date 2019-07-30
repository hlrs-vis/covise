// rei, Thu Jul 19 09:41:43 DST 2001

#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
//#include "coFeedback.h"
#include "RadialRunner.h"
#include <General/include/log.h>
#include "../lib/General/include/CreateFileNameParam.h"

RadialRunner::RadialRunner(int argc, char *argv[])
#ifndef YAC
: coModule(argc, argv, "Radial Runner")
#else
: coSimpleModule(argc, argv, "Radial Runner")
#endif
{
	char buf[256];
	char *pfn;
	int i;

	geo = NULL;

	fprintf(stderr, "RadialRunner::RadialRunner()\n");

	SetDebugPath(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(),getenv(ENV_IHS_DEBPATH));
	SetDebugLevel(0);
	if (getenv(ENV_IHS_DEBUGLEVEL))
		SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
	else
		dprintf(0, "WARNING: %s is not set. (now setting to 0)\n",
				ENV_IHS_DEBUGLEVEL);
        if ((pfn = CreateFileNameParam(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(), ENV_IHS_DEBPATH, coCoviseConfig::getEntry("value","Module.IHS.DebFile","RadialRunner.deb").c_str(), CFNP_NORM)) != NULL)
   
	{
		dopen(pfn);
		free(pfn);
	}
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**									**\n");
   dprintf(0, "** %-64.64s **\n", "Radial Runner Module");
   dprintf(0, "** %-64.64s **\n", "(c) 1999-2011 by University of Stuttgart - IHS");
   dprintf(0, "**									**\n");
   dprintf(0, "**********************************************************************\n");
   dprintf(0, "**********************************************************************\n");

	// reduced modification settings
	numReducedMenuPoints = 0;
	ReducedModifyMenuPoints = (char **)calloc(MAX_MODIFY, sizeof(char*));
	// DO NOT CHANGE THE ORDER OF THE FOLLOWING LINES !!!
	// or fix CheckUserInput();
	AddOneParam(M_INLET_ANGLE);
	AddOneParam(M_OUTLET_ANGLE);
	AddOneParam(M_PROFILE_THICKNESS);
	AddOneParam(M_TE_THICKNESS);
	AddOneParam(M_TE_WRAP_ANGLE);
	AddOneParam(M_BL_WRAP_ANGLE);
	AddOneParam(M_PROFILE_SHIFT);
	AddOneParam(M_INLET_ANGLE_MODIFICATION);
	AddOneParam(M_OUTLET_ANGLE_MODIFICATION);
	AddOneParam(M_REMAINING_SWIRL);
	AddOneParam(M_BLADE_LESPLINE_PARAS);
	AddOneParam(M_BLADE_TESPLINE_PARAS);
	AddOneParam(M_CENTRE_LINE_CAMBER);
	AddOneParam(M_CENTRE_LINE_CAMBER_POSN);
	AddOneParam(M_CAMBPARA);
	AddOneParam(M_BLADE_LENGTH_FACTOR);

	fprintf(stderr, "RadialRunner::RadialRunner() Init of StartFile\n");
	startFile = addFileBrowserParam("startFile","Start_file");
   
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
		fprintf(stderr, "Startpath: %s\n", pfn);
		startFile->setValue(pfn,"*.cfg");
		free(pfn);
	}
	else
		fprintf(stderr, "WARNING: pfn ist NULL !\n");

	// WE build the User-Menue ...
	RadialRunner::CreateUserMenu();
	RadialRunner::CreatePortMenu();

	// the output ports
	fprintf(stderr, "RadialRunner::RadialRunner() SetOutPort\n");
	blade  = addOutputPort("blade","Polygons","Blade Polygons");
	hub	   = addOutputPort("hub","Polygons","Hub Polygons");
	shroud = addOutputPort("shroud","Polygons","Shroud Polygons");
	grid   = addOutputPort("grid","UnstructuredGrid","Computational grid");
	dprintf(2,"RadialRunner::RadialRunner() SetOutPort ... done!\n");

	bcin = addOutputPort("bcin","Polygons","Cells at entry");
	bcout = addOutputPort("bcout","Polygons","Outlet");
	bcwall = addOutputPort("bcwall","Polygons","Walls");
	bcblade = addOutputPort("bcblade","Polygons","Blade");
	bcperiodic = addOutputPort("bcperiodic","Polygons","Periodic borders");

	boco = addOutputPort("boco", "USR_FenflossBoco", "Boundary Conditions");

  boundaryElementFaces = addOutputPort("boundary_element_faces", "coDoSet", "boundary element faces");

	for(i = 0; i < NUM_PLOT_PORTS; i++) {
		sprintf(buf,"XMGR%s_%d",M_2DPLOT,i+1);
		plot2d[i] = addOutputPort(buf,"Vec2","plot data");
	}

	isInitialized=0;
	dprintf(2,"RadialRunner::RadialRunner() isInitialized =%d\n",
			isInitialized);
}


void RadialRunner::AddOneParam(const char *p)
{
	if (++numReducedMenuPoints > MAX_MODIFY) {
		fprintf(stderr, "Sorry, MAX_MODIFY is to small :-(\n");
		exit(1);
	}
	ReducedModifyMenuPoints[numReducedMenuPoints - 1] = strdup(p);
}


void RadialRunner::postInst()
{
	startFile->show();
}


void RadialRunner::param(const char *portname, bool inMapLoading)
{
	int j;
	char buf[255];
	int changed = 0;

	dprintf(2,"RadialRunner::param(): \n");
	dprintf(2,"portname: %s\n",portname);
	dprintf(2,"m_2DplotChoice[0]: %d\n",m_2DplotChoice[0]->getValue());
	dprintf(2,"m_2DplotChoice[1]: %d\n",m_2DplotChoice[1]->getValue());
	for(j = 0; j < MAX_ELEMENTS;j++) {
		dprintf(2,"RadialRunner::param: 0: j: %d: p_ShowConformal = %d\n",
				j,p_ShowConformal[j][0]->getValue());
	}

	fprintf(stderr,"RadialRunner::param(): %s\n", portname);fflush(stderr);
	if (strcmp(portname,"startFile")==0) {
		dprintf(2,"RadialRunner::param(): startFile\n");fflush(stderr);
		if (isInitialized) {
			sendError("We Had an input file before...");
			return;
		}
#ifndef YAC
		Covise::getname(buf,startFile->getValue());
#else
                coFileHandler::getName(buf, startFile->getValue());
#endif
		dprintf(2, "RadialRunner::param = ReadGeometry(%s) ...", buf);
		geo = ReadGeometry(buf);
		dprintf(2, "done\n");
		if (geo) {
			isInitialized=1;
			RadialRunner::Struct2CtrlPanel();
			changed = 1;
		}
	}
	dprintf(2,"RadialRunner::param(): inMapLoading = %d\n", inMapLoading);
	if (!inMapLoading) {
		dprintf(2,"RadialRunner::param(): !inMapLoading\n");
		if (CheckUserInput(portname, geo, rrg) || changed) {
#ifndef YAC
			//dprintf(2,"\n selfExec() ... \n");
			//selfExec();
			//dprintf(2,"\n selfExec() ... done!\n");
#endif
		}
	}
	dprintf(2,"RadialRunner::param() ... done!\n");
}


#ifndef YAC
void RadialRunner::quit()
{
	// :-)
}
#else
int RadialRunner::quit()
{
   return 0;
}

#endif

int RadialRunner::compute(const char *)
{
	char name[256];
	coDoPolygons *poly;
	struct covise_info *ci;

	dprintf(1," RadialRunner::compute(const char *) ... \n");

	if (!geo) {
		sendError(" Please select an input file first!");
		return FAIL;
	}

#ifndef YAC
        Covise::getname(name,startFile->getValue());
#else
        coFileHandler::getName(name, startFile->getValue());
#endif
	strcat(name, ".new");
	WriteGeometry(geo, name);

	/////////////////////////////
	// create geometry for COVISE
	if ((ci = CreateGeometry4Covise(geo))) {
		dprintf(0, "RadialRunner::compute(const char *): Geometry created\n");
		sendInfo(" Specific speed: %.4f",geo->rr->des->spec_revs);
		RadialRunner::BladeElements2CtrlPanel();
		RadialRunner::BladeElements2Reduced();
		poly = new coDoPolygons(blade->getObjName(),
							   ci->p->nump,
							   ci->p->x, ci->p->y, ci->p->z,
							   ci->vx->num,	 ci->vx->list,
							   ci->pol->num, ci->pol->list);
		if(!poly) {
			dprintf(0,"RadialRunner::compute(const char *): creating polygons failed!\n");
			return FAIL;
		}

		poly->addAttribute("MATERIAL","metal metal.30");
		poly->addAttribute("vertexOrder","1");
		blade->setCurrentObject(poly);

		poly = new coDoPolygons(hub->getObjName(),
							   ci->p->nump,
							   ci->p->x, ci->p->y, ci->p->z,
							   ci->lvx->num,  ci->lvx->list,
							   ci->lpol->num, ci->lpol->list);
		poly->addAttribute("MATERIAL","metal metal.30");
		poly->addAttribute("vertexOrder","1");
		hub->setCurrentObject(poly);

		poly = new coDoPolygons(shroud->getObjName(),
							   ci->p->nump,
							   ci->p->x, ci->p->y, ci->p->z,
							   ci->cvx->num,  ci->cvx->list,
							   ci->cpol->num, ci->cpol->list);
		poly->addAttribute("MATERIAL","metal metal.30");
		poly->addAttribute("vertexOrder","1");
		shroud->setCurrentObject(poly);

		if(p_WriteBladeData->getValue()) {
			if(PutBladeData(geo->rr)) sendError("%s",GetLastErr());
		}
	}
	else {
		dprintf(0, "Error in CreateGeometry4Covise (%s, %d)\n", __FILE__, __LINE__);
		sendError("%s",GetLastErr());
	}
	
	// **************************************************
	// create 2d-plots
	RadialRunner::CreatePlot();

	// **************************************************
	// create meridian contour 2d-plot
	// **************************************************

	// **************************************************
	//////////////// This creates the volume grid ////////////////////

	////////////////////////
	// if button is pushed --> create computation grid
	// and set attributes
	if (p_makeGrid->getValue()) {
		RadialRunner::CreateGrid();
	}
	// **************************************************

	// **************************************************

	return SUCCESS;
}

MODULE_MAIN(VISiT, RadialRunner)
