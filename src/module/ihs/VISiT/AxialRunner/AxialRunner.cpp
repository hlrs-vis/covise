#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "AxialRunner.h"
#include "../lib/General/include/log.h"
#include "../lib/General/include/CreateFileNameParam.h"
#ifdef   VATECH
#include <WriteEuler/include/writeeuler.h>
#endif                                            // VATECH

AxialRunner::AxialRunner(int argc, char *argv[])
: coModule(argc, argv, "Axial Runner")
{
	char buf[256];
	char *pfn;
	int i;

	geo  = NULL;
   rrg = NULL;
#ifdef VATECH
	lfilenameGrid  = NULL;
	lfilenameEuler = NULL;
#endif


	SetDebugPath(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(),getenv(ENV_IHS_DEBPATH));
	if (getenv(ENV_IHS_DEBUGLEVEL))
		SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
	else {
		dprintf(0, "WARNING: %s is not set. (now setting to 0)\n",
				ENV_IHS_DEBUGLEVEL);
		SetDebugLevel(0);
	}

if ((pfn = CreateFileNameParam(coCoviseConfig::getEntry("value","Module.IHS.DebPath","/tmp/").c_str(), ENV_IHS_DEBPATH, coCoviseConfig::getEntry("value","Module.IHS.DebFile","AxialRunner.deb").c_str(), CFNP_NORM)) != NULL)
	{
		dopen(pfn);
		free(pfn);
	}
	dprintf(0, "**********************************************************************\n");
	dprintf(0, "**********************************************************************\n");
	dprintf(0, "**									**\n");
	dprintf(0, "** %-64.64s **\n", "Axial-runner module");
	dprintf(0, "** %-64.64s **\n", "(c) 1999-2005 by University of Stuttgart - IHS");
	dprintf(0, "**									**\n");
	dprintf(0, "**********************************************************************\n");
	dprintf(0, "**********************************************************************\n");

	// reduced modification settings
	numReducedMenuPoints = 0;
	ReducedModifyMenuPoints = (char **)calloc(MAX_MODIFY, sizeof(char *));
	// DO NOT CHANGE THE FOLLOWING ORDER !!!
	// or fix CheckUserInput();
	AddOneParam(M_INLET_ANGLE);
	AddOneParam(M_OUTLET_ANGLE);
	AddOneParam(M_INLET_ANGLE_MODIFICATION);
	AddOneParam(M_OUTLET_ANGLE_MODIFICATION);
	AddOneParam(M_PROFILE_THICKNESS);
	AddOneParam(M_TE_THICKNESS);
	AddOneParam(M_MAXIMUM_CAMBER);
	AddOneParam(M_CAMBER_POSITION);
	AddOneParam(M_PROFILE_SHIFT);

	dprintf(2, "Init of Startfile\n");
   std::string dataPath; 
#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE");
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(!defaultDir)
      defaultDir = "/data/IHS";

   dataPath=coCoviseConfig::getEntry("value","Module.IHS.DataPath",defaultDir);
   sprintf(buf,"%s/covise/src/application/ihs/VISiT/example_data/",defaultDir);

	startFile = addFileBrowserParam("startFile","Start file");
        if ((pfn = CreateFileNameParam(dataPath.c_str(), "IHS_DATAPATH", "nofile", CFNP_NORM)) != NULL)
	{
		dprintf(3, "Startpath: %s\n", pfn);
		startFile->setValue(pfn,"*.cfg");
		free(pfn);
	}
	else
		dprintf(0, "WARNING: pfn ist NULL !\n");

	// WE build the User-Menue ...
	AxialRunner::CreateUserMenu();
	AxialRunner::CreatePortMenu();

	// the output ports
	dprintf(2, "AxialRunner::AxialRunner() SetOutPort\n");
	blade  = addOutputPort("blade","Polygons","Blade Polygons");
	hub    = addOutputPort("hub","Polygons","Hub Polygons");
	shroud = addOutputPort("shroud","Polygons","Shroud Polygons");
	grid   = addOutputPort("grid","UnstructuredGrid","Computational grid");

	bcin = addOutputPort("bcin","Polygons","Cells at entry");
	bcout = addOutputPort("bcout","Polygons","Outlet");
	bcwall = addOutputPort("bcwall","Polygons","Walls");
	bcblade = addOutputPort("bcblade","Polygons","Blade");
	bcperiodic = addOutputPort("bcperiodic","Polygons","Periodic borders");

	boco = addOutputPort("boco", "USR_FenflossBoco", "Boundary Conditions");
  
//        inletElementFaces = addOutputPort("inlet_element_faces", "IntArr", "inlet face nodes");
//        outletElementFaces = addOutputPort("outlet_element_faces", "IntArr", "outlet face nodes");
//        shroudElementFaces = addOutputPort("shroud_element_faces", "IntArr", "shroud face nodes");
//        shroudExtElementFaces = addOutputPort("shroudExt_element_faces", "IntArr", "shroudExt face nodes");
//        frictlessElementFaces = addOutputPort("frictless_element_faces", "IntArr", "frictless face nodes");
//        psbladeElementFaces = addOutputPort("psblade_element_faces", "IntArr", "psblade face nodes");
//        ssbladeElementFaces = addOutputPort("ssblade_element_faces", "IntArr", "ssblade face nodes");
//        wallElementFaces = addOutputPort("wall_element_faces", "IntArr", "wall face nodes");
//        ssleperiodicElementFaces = addOutputPort("ssleperiodic_element_faces", "IntArr", "ssleperiodic face nodes");
//        psleperiodicElementFaces = addOutputPort("psleperiodic_element_faces", "IntArr", "psleperiodic face nodes");
//        ssteperiodicElementFaces = addOutputPort("ssteperiodic_element_faces", "IntArr", "ssteperiodic face nodes");
//        psteperiodicElementFaces = addOutputPort("psteperiodic_element_faces", "IntArr", "psteperiodic face nodes");
        
        boundaryElementFaces = addOutputPort("boundary_element_faces", "coDoSet", "boundary element faces");

#ifdef   VATECH
	gridOutPort = addOutputPort("eugrid","StructuredGrid","structured grid");
	velocityOutPort = addOutputPort("velocity","Vec3","velocity data");
	relVelocityOutPort = addOutputPort("relVelocity","Vec3","Relativ velocity data");
	pressureOutPort = addOutputPort("pressure","Float","pressure data");
#endif
	for(i = 0; i < NUM_PLOT_PORTS; i++)
	{
		sprintf(buf,"XMGR%s_%d",M_2DPLOT,i+1);
		plot2d[i] =addOutputPort(buf,"Vec2","XMGR-plot data");
	}

	isInitialized = 0;
}

void AxialRunner::param(const char *portname, bool inMapLoading)
{
	char buf[255];
	int changed = 0;

	dprintf(1, "Entering AxialRunner::param = %s\n", portname);
	if (strcmp(portname,"startFile")==0)
	{
		if (isInitialized)
		{
			sendError("We Had an input file before...");
			return;
		}
		Covise::getname(buf,startFile->getValue());
		dprintf(3, "AxialRunner::param = ReadGeometry(%s) ...", buf);
		geo = ReadGeometry(buf);
		dprintf(3, "done\n");
		if (geo)
		{
			isInitialized = 1;
			AxialRunner::Struct2CtrlPanel();
			changed = 1;
		}
	}
#ifdef   VATECH
	if (!strcmp(M_EU_GRID_FILE,portname))
	{
		lfilenameGrid = strdup(filenameGrid->getValue());
		changed = 1;
	}
	if (!strcmp(M_EU_EULER_FILE,portname))
	{
		lfilenameEuler = strdup(filenameEuler->getValue());
		changed = 1;
	}
#endif                                         // VATECH
	if (!inMapLoading)
	{
		if (CheckUserInput(portname,geo,rrg) || changed)
		{
			dprintf(1, "AxialRunner::param: selfExec()\n" );
			// bitte nicht!!!!selfExec();
			dprintf(1, "AxialRunner::param: selfExec() done!\n" );
		}
	}
	dprintf(1, "Leaving AxialRunner::param = %s\n", portname);
}


void AxialRunner::quit()
{
	dprintf(1, "Entering AxialRunner::quit\n");
	// :-)
#ifdef VATECH
	if (fifoin)
		fclose(fifoin);
	if (fifofilein && *fifofilein)
		unlink(fifofilein);
	if (fifofileout && *fifofileout)
		unlink(fifofileout);
#endif                                         // VATECH
	dprintf(1, "Leaving AxialRunner::quit\n");
}


int AxialRunner::compute(const char *)
{
	char name[256];
	int res = -1;
#ifdef   VATECH
	static int EuGridChanged = 0;
#endif
	dprintf(1, "AxialRunner::compute(const char *) entering... \n");

// **********************************************************************
// VATECH-Specials
#ifdef   VATECH
	if (lfilenameGrid && *lfilenameGrid)
	{
		if (eu)  FreeStructEuler(eu);
		eu = ReadEulerGrid(lfilenameGrid,
						   -1.0*(360.0/geo->ar->nob-30.0)*M_PI/180.0);
		dprintf(3, "%s: file=%s", M_EU_GRID_FILE, lfilenameGrid);
		free(lfilenameGrid);
		lfilenameGrid = NULL;
		EuGridChanged = 1;
	}
	if (lfilenameEuler && *lfilenameEuler && eu)
	{
		ReadEulerResults(lfilenameEuler, eu, omega->getValue());
		dprintf(3, "%s: file=%s", M_EU_EULER_FILE, lfilenameEuler);
		free(lfilenameEuler);
		lfilenameEuler = NULL;
		EuGridChanged = 1;
	}

	if ((p_RotateGrid->getValue() && !multieu && eu)
		|| (EuGridChanged && p_RotateGrid->getValue() && eu))
	{
		FreeMultiEu();
		multieu = MultiRotateGrids(eu, geo->ar->nob);
		EuGridChanged = 0;
	}

	if (p_RunVATEuler->getValue())
	{
		std::string fn;
		std::string sh;
		std::string p;
		dprintf(3," AxRunner::compute p_RunVATEuler->getValue = %d\n",
				p_RunVATEuler->getValue());
		p_RunVATEuler->disable();
		p_RunVATEuler->setValue(0);                 // push off button
		filenameGrid->disable();
		filenameEuler->disable();

		p=coCoviseConfig::getEntry(EU_WORKING_PATH);
		fn=coCoviseConfig::getEntry(EU_CENTRAL_PATH);
		sh=coCoviseConfig::getEntry(EU_START_PATH);
		if (!p.empty() && !fn.empty() && !sh.empty())
		{
			char pfn[255];
			int err;

			strcpy(pfn, p.c_str());
			strcat(pfn, "/");
			strcat(pfn, fn.c_str());
			dprintf(3, "WriteEuler(%s) started ...\n", pfn);
			if ((err = WriteEuler(pfn, geo->ar)) != 0)
			{
				dprintf(0, "ERROR: err=%d (%s)\n", err, strerror(err));
			}
			dprintf(3, "WriteEuler(%s) finished\n", pfn);
			if (sh && *sh)
			{
				sprintf(pfn, "%s/%s %s %s %f %f %f %f %f %f %f &", p.c_str(), sh.c_str(),
						fifofilein, fifofileout,
						p_Head->getValue(), p_Discharge->getValue(),
						p_ProtoDiam->getValue(), p_ProtoSpeed->getValue(),
						p_nED->getValue(), p_QED->getValue(), p_alpha->getValue());
				dprintf(3, "StartEuler(%s)\n", pfn);
				system(pfn);
			}
		}
	}
	if (multieu == NULL)
	{
		if (eu)
		{
			struct EuGri *eug[2];
			eug[0] = eu;
			eug[1] = NULL;
			ShowEuler(eug);
		}
	}
	else
		ShowEuler(multieu);
#endif                                         // VATECH
// **********************************************************************

	if (!geo) {
		sendError("Please select a parameter file first!!");
		return FAIL;
	}

	Covise::getname(name,startFile->getValue());
	strcat(name, ".new");
	res = WriteGeometry(geo, name);
	dprintf(3, "WriteGeometry sends: %d\n", res);

	/////////////////////////////
	// create geometry for COVISE
	AxialRunner::CreateGeo();

	// create 2D-Plot
	AxialRunner::CreatePlot();

	dprintf(1, "AxialRunner::compute(const char *) done\n");
	// **************************************************

	// **************************************************
	//////////////// This creates the volume grid ////////////////////

	////////////////////////
	// if button is pushed --> create computation grid
	AxialRunner::CreateGrid();

	// **************************************************
 
	// **************************************************
	// Run CFD-Analysis.
	if (p_RunFENFLOSS->getValue())
	{
		char runsh[200];
		char name[200];
		float vnorm, lnorm, omega;
		vnorm = rrg->inbc->cm;
		lnorm = geo->ar->ref;
		omega = float(geo->ar->des->revs*M_PI/30.0);
		Covise::getname(name,startFile->getValue());
		strcat(name, ".new");
		p_RunFENFLOSS->setValue(0);                 // push off button
		sprintf(runsh, "%s %.2f %.2f %.2f %d %s", "runFEN.pl",
				vnorm,lnorm,omega,geo->ar->nob, name);
		if (system(runsh) == -1)
        {
            dprintf(1, "AxialRunner::system() failed\n");
        }
	}
	// end of CFD-Run.
	// **************************************************
	return SUCCESS;
}

void AxialRunner::AddOneParam(const char *p)
{
	if (++numReducedMenuPoints > MAX_MODIFY)
	{
		dprintf(0, "Sorry, MAX_MODIFY is to small :-(\n");
		Covise::sendError("Sorry, MAX_MODIFY is to small :-(\n");
		return;
	}
	ReducedModifyMenuPoints[numReducedMenuPoints - 1] = strdup(p);
}


MODULE_MAIN(VISiT, AxialRunner)
