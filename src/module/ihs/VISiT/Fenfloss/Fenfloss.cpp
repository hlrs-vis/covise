
#include <config/CoviseConfig.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>

#include "Fenfloss.h"

#include <sys/types.h>
#ifdef WIN32
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <strings.h>
#include <unistd.h>
#endif
#include <string.h>
#include <General/include/plot.h>
#include <General/include/log.h>
#include <iostream>
#include <fstream>
#include "../lib/General/include/CreateFileNameParam.h"


#ifdef HAVE_GLOBUS
#undef IOV_MAX
#include <SimulationService_client.h>
#include <globus_soap_message.h>
#include <globus_common.h>
#include <globus_io.h>
#endif

// time prints
#include <General/include/mytime.h>

//// Constructor : set up User Interface//

#ifndef  DIM
#define  DIM(x)   (sizeof(x)/sizeof(*(x)))
#endif

//#define VERBOSE
#undef  MESSAGES

#undef DUMMY

#ifdef DUMMY
ofstream testfile("testfile");
#define sendBS_Data(data,size) {  int i; \
testfile << "\n------- Send " << size/4 << " Elem:" << endl; \
for (i=0;i<size/4;i++) \
{ \
if (i%5==0) testfile << endl; \
testfile << data[i] << " "; \
} \
testfile << endl; \
}
#endif

Fenfloss::Fenfloss(int  argc,  char  *argv[])
:coSimLib(argc, argv, argv[0], "Simulation coupling")
{
	char *pfn;
	////////// set up default parameters
        
#ifdef VERBOSE

	cerr << "##############################" << endl;
	cerr << "#####   Fenfloss "              << endl;
	cerr << "#####   PID =  " << getpid()   << endl;
	cerr << "##############################" << endl;
#endif
#ifndef YAC
	set_module_description("Fenfloss Simulation");
#endif
	SetDebugPath(coCoviseConfig::getEntry("Module.IHS.DebPath").c_str(),getenv(ENV_IHS_DEBPATH));
/*
	SetDebugLevel(0);
	if (getenv(ENV_IHS_DEBUGLEVEL))
           SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
*/
        if ((pfn = CreateFileNameParam(coCoviseConfig::getEntry("value","Module.IHS.DebPath","/tmp/").c_str(), ENV_IHS_DEBPATH, coCoviseConfig::getEntry("value","Module.IHS.DebFile","Fenfloss.deb").c_str(), CFNP_NORM)) != NULL)
	{
           dopen(pfn);
           free(pfn);
	}

	// Parameters
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
	dprintf(2, "Fenfloss::Fenfloss(): dp = %s\n", dataPath.c_str());
        
        Fenfloss::CreateUserMenu();

	// Input ports : yet only parallel ones allowed
	p_grid    = addInputPort("distGrid","UnstructuredGrid","Distributed Grid");
	p_boco    = addInputPort("distBoco1","USR_DistFenflossBoco","Distributed Boundary Cond");
	p_boco2   = addInputPort("distBoco2","USR_DistFenflossBoco","Distributed Boundary Cond");
	p_boco2->setRequired(0);
	p_in_bcin = addInputPort("bcin","Polygons","Boundary surface of inlet(IN)");
	p_in_bcin->setRequired(0);

	// Output ports
	p_velocity   = addOutputPort("velocity","Vec3","Geschwindigkeit");
	p_press      = addOutputPort("p","Float","Druck");
	p_turb       = addOutputPort("turb","Float","Turbulen");
	p_out_bcin   = addOutputPort("out_bcin","Polygons","Boundary surface of inlet(OUT)");

	// not yet started
	stepNo = -1;

	// No object yet received
	d_distGridName = new char [1];
	d_distGridName[0] = '\0';
	d_distBocoName = new char [1];
	d_distBocoName[0] = '\0';

#ifdef HAVE_GLOBUS        
	globus_module_activate(GLOBUS_COMMON_MODULE);
	globus_module_activate(GLOBUS_SOAP_MESSAGE_MODULE);
#endif

}


void Fenfloss::param(const char *portname, bool inMapLoading)
{
        int connMeth;
        (void) inMapLoading;
        dprintf(1, "Fenfloss::param(): Entering Fenfloss::param(): %s\n", portname);
        // if spotpoint is changed, we have to copy the values in
	// the "transfer buffers"
        if (!strcmp(portname, p_ConnectionMethod->getName()))
        {
           dprintf(2, "p_ConnectionMethod changed ...\n");
           connMeth = p_ConnectionMethod->getValue();
           if (!strcmp(s_ConnectionMethod[connMeth], "local"))
           {
              p_User->disable();
              p_Hostname->disable();
           }
           else
           {
              p_User->enable();
              p_Hostname->enable();
           }
        }
        else if (!strcmp(portname, p_Port->getName())) {
           
           int min, max;
           int n = sscanf(p_Port->getValue(), "%d %d", &min, &max);
           
           if (!n)
              return;

           if (n == 1)
              setPorts(min, min);
           else
              setPorts(min, max);           
        }

#ifdef HAVE_GLOBUS
	else if (!strcmp(portname, p_Discovery->getName()) || 
                 !strcmp(portname, p_User->getName()))
        {
           // globus support will come back later
        }

	else if (!strcmp(portname, p_Simulations->getName())) {
           int index = p_Simulations->getValue();
           if (index < 1) return;

           char user[128];
           char host[128];
           char port[128];
           char *sim = s_Simulations[index];

           if (sim) {
              if (sscanf(sim, "%s @ %s : %s\n", user, host, port) == 3) {

                 p_User->setValue(user);
                 p_Hostname->setValue(host);
                 p_Port->setValue(port);
              }
              p_ConnectionMethod->setValue(6); // reattach
           }
        }
#endif
}


void Fenfloss::postInst()
{
	p_StartupSwitch->show();
        /*
	p_useInitial->show();
	p_stopSim->show();
        */
	p_pauseSim->show();
        /*
	p_GetSimData->show();
	p_detachSim->show();
	p_updateInterval->show();
	p_ConnectionMethod->show();
	p_Hostname->show();
        p_Port->show();
	p_simApplication->show();
	p_User->show();
        p_Discovery->show();
        p_Simulations->show();
        */
}

int Fenfloss::endIteration() {
   // called at COMM_QUIT in coSimLib::handleCommand()

   cerr << "Fenfloss:endIteration()" << endl;

   // update for draft tube sim
   // if not tube
   if (strcmp(p_StartupSwitch->getActLabel(), "tube"))
      numbc = 0;
   
   //struct commandMsg numbc_command = { NEWBC, numbc * 3 * sizeof(int) };
   // only needed for boundary condition update during simulation!
   //sendBS_Data(&numbc_command, sizeof(commandMsg));

   if (numbc > 0)
   {
      sendBS_Data(bcrad, numbc * sizeof(float));
      sendBS_Data(bcvu, numbc * sizeof(float));
      sendBS_Data(bcvm, numbc * sizeof(float));
   }

   //struct commandMsg boco2_command = { USE_BOCO2, sizeof(int) };
   //sendBS_Data(&boco2_command, sizeof(commandMsg));
   //sendBS_Data(&use_boco2, sizeof(int));
   
   if (use_boco2) {
	   for (int i = 0; i < numProc; i ++) {
		   dprintf(1, "Fenfloss-Mod(%d): Compute(): Sending proc %d bc %d \n", __LINE__, i, (int) (boco2_num_int[i] * (int) sizeof(int)));
		   if (boco2_num_int[i])
			   sendBS_Data(boco2_idata[i], boco2_num_int[i] * sizeof(int));
	   }

	   for (int i = 0; i < numProc; i ++) {
		   dprintf(1, "FenFloss-Mod(%d)   BC: Sending (cpu=%d), size = %d Bytes\n", __LINE__, i, (int) (boco2_num_float[i] * sizeof(float) * 2));
         sendBS_Data(boco2_fdata[i], boco2_num_float[i] * 2 * sizeof(float));
      }
   }
   return 0;
}

int Fenfloss::compute(const char *)
{
   dprintf(1, "Fenfloss::compute\n");
   int reattach = !strcmp(s_ConnectionMethod[p_ConnectionMethod->getValue()], "reattach");

   if (reattach) {
      p_ConnectionMethod->setValue(0);
      stepNo = -1;
   }

   // Start: gettime
   int numElem;

   if (stepNo==0)
   {
      stepNo=1;
      return STOP_PIPELINE;
   }

   // Find out, whether we have to re-start sim
   use_boco2 = 0;
   coDoSet *grid  = (coDoSet*) p_grid->getCurrentObject();
   coDoSet *boco  = (coDoSet*) p_boco->getCurrentObject();
   coDoSet *boco2 = (coDoSet*) p_boco2->getCurrentObject();

   if (reattach)
      d_distGridName = strdup(grid->getName());

   // fl: debug
   grid->getAllElements(&numProc);
   dprintf(4, " Fenfloss::compute(const char *): grid: %s, %d\n", grid->getName(), numProc);

   if (grid)
   {   
      dprintf(1, "Fenfloss::compute reset\n");
      // we have a new grid input object
      const char *gridName = grid->getName();
      if (strcmp(gridName,d_distGridName)!=0)
      {

         // sim currently running
         if (stepNo>=0)
         {

#ifndef WIN32
            //system("killall -KILL p_flow_4.8.2");
            sleep(5);
#endif
            resetSimLib();
            stepNo=-1;
         }
         delete [] d_distGridName;
         d_distGridName = strcpy ( new char[strlen(gridName)+1] , gridName );
      }
   }

   if ( (boco) && (!boco2) )
   {
      // first run! using boco bc object
      sendInfo("no boco2 object, using boco object");
   }
   if ( (boco) && (boco2) )
   {
      // data on both boco-ports!
      const char *bocoName = boco->getName();
      // check if there is new data from domain decomposition
      if ( strcmp(bocoName,d_distBocoName)!=0 )
      {
         // no new data from domain decomposition
         sendInfo("simulation coupling! using boco2 bc object");
         use_boco2 = 1;
      }
      else
      {
         // new data from domain decomposition
         sendInfo("new data from domain decomposition ...");
      }
   }

   coDoPolygons *poly_in, *poly_out = NULL;
   int NumberOfPoints;
   int NumberOfVertices, NumberOfPolygons;
   float *inx_coord, *iny_coord, *inz_coord;
   int *invertices, *inpolygons;
   static int grid_size;    // correct size of data arrays for visualization

   const coDistributedObject *const *gridArr = grid->getAllElements(&numProc);
   const coDistributedObject *const *bocoArr;

   const coDistributedObject *const **procGrid;
   const coDistributedObject *const **procBoco = 0;

   const coDoIntArr *gridDim;
   const coDoIntArr *bocoDim;

   if (stepNo<0)
   {
      //const char *dir;

      printf("Fenfloss::compute stepno < 0\n");
      // get target directory
      //dir = p_dir->getValue();

      // CHECK TYPES .. later
      if (!grid || !boco)
      {
         sendError("Data not received");
         return FAIL;
      }

      if (!boco2)
         bocoArr = boco->getAllElements(&numProc);
      else
         bocoArr = boco2->getAllElements(&numProc);
      
#ifndef DUMMY
      ///////////////////////////////////////////////////////////////////////
      // start simulation
      dprintf(0, "------------------starting simulation-------------------\n");
      PrepareSimStart(numProc);
      if (startSim(reattach))
         return -1;
      stepNo=0;
#endif

      if (!reattach) {

		 dprintf(2,"sending parameters ...\n");
         int32 command;
         do {
            recvBS_Data(&command, sizeof(command));
            char name[128];

			dprintf(4,"command: %d\n",command);
            switch (command) {
               
                case GET_SC_PARA_FLO:
                case GET_V3_PARA_FLO:
                case GET_SC_PARA_INT:
                case GET_BOOL_PARA:
                case GET_TEXT_PARA:
                   if (recvData((void*) name, 64) != 64)
                      dprintf(0, "error in initial parameters\n");
                   break;
                case GET_INITIAL_PARA_DONE:
                   break;
                default:
                   dprintf(0, "error: unsupported parameter in initialization\n");
                   break;
            }
            
            switch (command) {
                case GET_SC_PARA_FLO: {
                   coFloatParam *param = dynamic_cast<coFloatParam *>(findElem(name));
                   struct { float val ; int32 error; } ret = {0.0,0};
                   if (param)
                      ret.val = param->getValue();
                   else if (!findAttribute(grid, name, "%f", (void *) &ret.val))
                      ret.error = -1;
                   dprintf(4, "sending %f %d\n", ret.val, ret.error);
                   sendBS_Data((void*) &ret, sizeof(ret)); 
                   break;
                }
                case GET_V3_PARA_FLO: {
                   coFloatVectorParam *param = dynamic_cast<coFloatVectorParam *>(findElem(name));
                   struct { float val[3] ; int32 error; } ret;
                   ret.error = 0;
                   if (param)
                      for(int i=0; i<3; i++) {
                         ret.val[i] = param->getValue(i);
                         dprintf(4,"param[%d]=%f\n",i,ret.val[i]);
                      }
                   else
                      ret.error = -1;

                   dprintf(4, "sending %f %d\n", ret.val[0], ret.error);
                   sendBS_Data((void*) &ret, sizeof(ret)); 
                   break;
                }
                case GET_SC_PARA_INT:
                case GET_BOOL_PARA: {
                   
                   int val = 0;
                   int error = 0;
                   if (command == GET_SC_PARA_INT) {
                      coIntScalarParam *param = dynamic_cast<coIntScalarParam *>(findElem(name));
                      if (param)
                         val = param->getValue();
                      else if (!findAttribute(grid, name, "%d", (void *) &val))
                         error = -1;
                   } else if (command == GET_BOOL_PARA) {
                      coBooleanParam *param = dynamic_cast<coBooleanParam *>(findElem(name));
                      if (param)
                         val = param->getValue();
                      else if (!findAttribute(grid, name, "%d", (void *) &val))
                         error = -1;
                   }
                   
                   struct { int val; int32 error; } ret;
                   ret.val = val;
                   ret.error = error;
                   dprintf(4, "sending int para %d %d\n", ret.val, ret.error);
                   sendBS_Data((void*) &ret, sizeof(ret)); 
                   break;
                }
                case GET_TEXT_PARA: {
                   coStringParam *param = dynamic_cast<coStringParam *>(findElem(name));
                   char res[256];
                   memset(res, 0, 256);
                   if (param) {
                      const char *val = param->getValue();
                      if (val)
                         strncpy(res, val, 255);
                   } else
                      findAttribute(grid, name, "%s", (void *) res);
                   
                   sendData(res, 256);
                   break;
                }
                   
                default:
                   break;
            }
         } while (command != GET_INITIAL_PARA_DONE);         
		 dprintf(2,"sending parameters ... done!\n");
      }
      
      ///////////////////////////////////////////////////////////////////////
      // send dimensions
      int *idata,size=0;
      int i,j;

      procGrid = new const coDistributedObject *const*[numProc];
      procBoco = new const coDistributedObject *const*[numProc];
      
      for (i = 0; i < numProc; i++)
      {
         procGrid[i] = ((const coDoSet*)gridArr[i])->getAllElements(&numElem);
         gridDim = (coDoIntArr *) procGrid[i][0];
         idata = gridDim->getAddress();
         size  = gridDim->getDimension(0) * sizeof(int);
         dprintf(1, "Fenfloss-Mod(%d): Compute(): Sending grid %d \n", __LINE__, size);
         dprintf(1, "     sending %d %d %d %d\n", idata[0], idata[1], idata[2], idata[3]);
         
         if (!reattach) {
            struct commandMsg data = { GEO_DIM, size };
            sendBS_Data(&data, sizeof(data));
            if (size)
               sendBS_Data(idata,size);
         }
      }
      
      for (i = 0; i < numProc; i++)
      {
         procBoco[i] = ((const coDoSet*)bocoArr[i])->getAllElements(&numElem);
         bocoDim = (const coDoIntArr *) procBoco[i][0];
         idata = bocoDim->getAddress();
         size  = bocoDim->getDimension(0) * sizeof(int);
         dprintf(2, "Fenfloss-Mod(%d): Compute(): Sending bc %d \n", __LINE__, size);
         if (!reattach) {
            struct commandMsg data = { BOCO_DIM, size };
            sendBS_Data(&data, sizeof(data));
            
            if (size)
               sendBS_Data(idata,size);
         }
      }

      ///////////////////////////////////////////////////////////////////////
      // send grids
      float *fdata;

      for (i = 0; i < numProc; i++)
      {
         coDoFloat *floatArr = (coDoFloat *) procGrid[i][1];
         floatArr->getAddress(&fdata);
         size = floatArr->getNumPoints() * sizeof(float);
         if (size)
         {
            dprintf(1, "FenFloss-Mod(%d) GRID: Sending (cpu=%d), size = %d Bytes\n", __LINE__, i, size);

            if (!reattach) {
               struct { int command; int size; } data = { SEND_GEO, size };
               sendBS_Data(&data, sizeof(data));            
               sendBS_Data(fdata,size);
            }
         }
#ifdef DUMMY
         else
            testfile << " --- ignored one field" << endl;
#endif
         // M. Becker: flow generates additional nodes
         // if we want to be sure that data and grid-arrays have the same length,
         // we have to shorten the data arrays (at the end of ::compute)!
         if (i == 0)
         {
            coDoIntArr *intArr = (coDoIntArr *) procGrid[i][0];
            idata = intArr->getAddress();
            grid_size = idata[1];
            dprintf(1, "*************************** grid_size: %d\n", grid_size);
         }
         for (j = 2; j < 11; j++)
         {
            coDoIntArr *intArr = (coDoIntArr *) procGrid[i][j];
            idata = intArr->getAddress();
            size = intArr->getDimension(0) * sizeof(int);
            if (size)
            {
               dprintf(1, "FenFloss-Mod(%d) GRID: Sending (cpu=%d, j=%d), size = %d Bytes\n", __LINE__, i, j, size);
               dprintf(4," idata[0] = %d, idata[1] = %d, idata[2] = %d\n",
					   idata[0], idata[1], idata[2]);
               if (!reattach) {
                  sendBS_Data(idata,size);
               }
            }
#ifdef DUMMY
            else
               testfile << " --- ignored one field" << endl;
#endif
         }
      }
      
      ///////////////////////////////////////////////////////////////////////
      // send boco
      for (i = 0; i < numProc; i++)
      {
         if (!reattach) {
            struct { int command; int size; } data = { SEND_BOCO, size };
            sendBS_Data(&data, sizeof(data));
         }
         for (j = 1; j < 12; j++)
         {
            coDoIntArr *intArr = (coDoIntArr *) procBoco[i][j];
            idata = intArr->getAddress();
            size = intArr->getDimension(0) * sizeof(int);
            if (size)
            {
				// debug, fl:
				for(int k = 0; k < size/4; k++) {
					dprintf(3,"send boco(%2d,%2d): %8d\n",i,j,idata[k]);
				}
				dprintf(2, "FenFloss-Mod(%d)   BC: Sending (cpu=%d, j=%d), size = %d Bytes\n", __LINE__, i, j, size);
				if (!reattach) {
					sendBS_Data(idata,size);
				}
            }
#ifdef DUMMY
            else
               testfile << " --- ignored one field" << endl;
#endif
         }

         for (j = 12; j < 13; j++)                   // displ_wert
         {
            sendInfo("Sending RB-Data ...");
            coDoFloat *floatArr = (coDoFloat *) procBoco[i][j];
            floatArr->getAddress(&fdata);
            size = floatArr->getNumPoints() * sizeof(float);
            if (size)
            {
               dprintf(2, "FenFloss-Mod(%d)   BC: Sending (cpu=%d, j=%d), size = %d Bytes\n", __LINE__, i, j, size);
               if (!reattach) {
                  sendBS_Data(fdata,size);
               }
               /*<tmp>
               if(j == 13)
               {
                  sendInfo("Writing Diplacement-Data to file ...");
                  ofstream ofs("displwerte.debug");
                  for(int ji(0); ji<size; ji+=6)
                  {
                     ofs << fdata[ji] << " " << fdata[ji+1] << " " << fdata[ji+2] << " " << fdata[ji+3] << " " << fdata[ji+4] << " " << fdata[ji+5] << endl;
                  }
                  ofs.close();
               }
               </tmp>*/
            }
#ifdef DUMMY
            else
               testfile <<  "--- ignored one field" << endl;
#endif
         }
      }
   } /* endif(stepno < 0) */

#ifndef YAC
   executeCommands();
#endif

   numbc = 0;
   
   for (int j = 0; j < numbc && bcrad; j++)
	   dprintf(4, "j=%d: bcrad=%f, bcvu=%f, bcvm=%f\n", j, bcrad[j], bcvu[j], bcvm[j]);
   
   if (use_boco2)
   {
      // use boco2 object (from Flowmid module)!
      bocoArr = boco2->getAllElements(&numProc);

      if (!boco2_num_int) {
         boco2_num_int = new int[numProc];
         boco2_idata = new int*[numProc];
      }
         
      // size
      for (int i = 0; i < numProc; i++)
      {
         if (boco2_idata[i]) delete [] boco2_idata[i];
         procBoco[i] = ((coDoSet*)bocoArr[i])->getAllElements(&numElem);
         bocoDim = (coDoIntArr *) procBoco[i][0];
         boco2_num_int[i] = bocoDim->getDimension(0);
         boco2_idata[i] = new int[boco2_num_int[i]];
         memcpy(boco2_idata[i], bocoDim->getAddress(), boco2_num_int[i] * sizeof(int));
      }
      
      if (!boco2_num_float) {
         boco2_num_float = new int[numProc];
         boco2_fdata = new float*[numProc];
      }

      ofstream debugfile;
      debugfile.open("boco2_bcin_data.txt");
      // data
      for (int i=0;i<numProc;i++)
      {
         float *address;
         procBoco[i] = ((coDoSet*)bocoArr[i])->getAllElements(&numElem);
         coDoFloat *floatArr = (coDoFloat *) procBoco[i][13]; // displ_wert
         if (boco2_fdata[i]) delete [] boco2_fdata[i];
         boco2_num_float[i] = floatArr->getNumPoints();
         boco2_fdata[i] = new float[2 * boco2_num_float[i]]; // |displ_wert| == |pres_wert| ? 
         floatArr->getAddress(&address);
         memcpy(boco2_fdata[i], address, boco2_num_float[i] * sizeof(float));
         floatArr = (coDoFloat *) procBoco[i][14]; // pres_wert
         floatArr->getAddress(&address);
         memcpy(boco2_fdata[i] + boco2_num_float[i], address, boco2_num_float[i] * sizeof(float));
      }
      /*  
         for (int j=13;j<15;j++)                  // displ_wert, pres_wert
         {
            coDoFloat *floatArr = (coDoFloat *) procBoco[i][j];
            floatArr->getAddress(&fdata);
            size = floatArr->getNumPoints() * sizeof(float);
            if (size)
            {
               printf("FenFloss-Mod(%d)   BC: Sending (cpu=%d, j=%d), size = %d Bytes\n", __LINE__, i, j, size);
               sendBS_Data(fdata,size);

            }
            if (j==13)
               for (int k=0; k<size/4; k+=6)
               {
                  debugfile <<  k/6 << " " << fdata[k]
                            << " " << fdata[k+1]
                            << " " << fdata[k+2]
                            << " " << fdata[k+3]
                            << " " << fdata[k+4]
                            << " " << fdata[k+5]
                            << endl;
               }

            debugfile.close();

#ifdef DUMMY
            else
               testfile << " --- ignored one field" << endl;
#endif
         }
      }
      */
   }

   //////////////////////////////
   // we had to add to the spot point data, the description for
   // CollectTimeSteps and the Plot module
   // the output port for the VR Plugin
   poly_in = (coDoPolygons *)p_in_bcin->getCurrentObject();
   if (poly_in) {
      NumberOfPoints   = poly_in->getNumPoints();
      NumberOfVertices = poly_in->getNumVertices();
      NumberOfPolygons = poly_in->getNumPolygons();
      poly_in->getAddresses(&inx_coord, &iny_coord, &inz_coord, &invertices, &inpolygons);

      poly_out = new coDoPolygons(p_out_bcin->getObjName(),
                                  NumberOfPoints,
                                  inx_coord, iny_coord, inz_coord,
                                  NumberOfVertices, invertices,
                                  NumberOfPolygons, inpolygons);
      poly_out->addAttribute("vertexOrder","1");
      poly_out->addAttribute("COLOR","red");
      p_out_bcin->setCurrentObject(poly_out);
   }
#ifndef YAC
   coFeedback feedback("FenflossPlugin");
   feedback.addPara(p_updateInterval);
   feedback.addPara(p_pauseSim);
   feedback.addPara(p_GetSimData);
   feedback.addPara(p_detachSim);
   feedback.addPara(p_useInitial);
   feedback.addPara(p_stopSim);
   if (poly_in)
      feedback.apply(poly_out);
#endif

   // Flow knows now, that it has to send new simulation data
   if (p_GetSimData->getValue())
      p_GetSimData->setValue(0);                  // push off button
/*
   if (p_stopSim->getValue())
      p_stopSim->setValue(0);                     // push off button
*/
   // M. Becker 17.6.2002
   // resize data arrays to fit with original created grid for visualization

   if (p_velocity->getCurrentObject())                      // do this only if there is an object!
   {
      coDoVec3 *velo = (coDoVec3 *)p_velocity->getCurrentObject();
      velo->setSize(grid_size);

      coDoFloat *press = (coDoFloat *)p_press->getCurrentObject();
      press->setSize(grid_size);
      if (p_turb->getCurrentObject())
      {
         coDoFloat *turb = (coDoFloat *)p_turb->getCurrentObject();
         turb->setSize(grid_size);
      }
   }

   if (!p_velocity->getCurrentObject())
      return STOP_PIPELINE;
   else
      return CONTINUE_PIPELINE;
}


void Fenfloss::StopSimulation(void)
{
	char *stopshell = NULL;
	char simStop[255];

	stopshell = ConnectionString();
	sprintf(simStop, "%s '%s kill'", stopshell, SimBatchString());
	if(system(simStop)==-1)
           dprintf(1, "Fenfloss::StopSimulation: execution of %s failed\n", simStop);
	if (stopshell) free(stopshell);
}

#ifndef YAC
void Fenfloss::quit()
{
	StopSimulation();
}
#else
int Fenfloss::quit()
{
	StopSimulation();
        return 0;
}
#endif

void Fenfloss::PrepareSimStart(int numProc)
{
	char sNumNodes[30];
	char *startshell = NULL;
	char simStart[255];
	int connMeth;
	int simAppl;

	dprintf(1, "Fenfloss::PrepareSimStart(%d)\n", numProc);
	sprintf(sNumNodes, "%d", numProc);
	startshell = ConnectionString();

	connMeth = p_ConnectionMethod->getValue();
	simAppl = p_simApplication->getValue();

	setUserArg(0, startshell);

	const char *caseString = p_StartupSwitch->getActLabel();

	if (!strcmp(s_ConnectionMethod[connMeth], "echo"))
		sprintf(simStart, "echo -T \"flow_%s\" -e %s start %s %d", caseString, SimBatchString(), caseString, numProc);
#ifdef WIN32
	else if (!strcmp(s_ConnectionMethod[connMeth], "WMI"))
	{
		sprintf(simStart, "\"%s start %s %d\" \" \" \"%s\" \" %s \"",  SimBatchString(), caseString, numProc, p_Hostname->getValue(), p_User->getValue());
		fprintf(stderr,"simStart='%s'\n", simStart);
	}
#endif
	else if (!strcmp(s_ConnectionMethod[connMeth], "rdaemon"))
	{
		sprintf(simStart, "\"%s start %s %d %s\" \" \" \"%s\" \" %s \"",  SimBatchString(), caseString, numProc, s_simApplication[simAppl], p_Hostname->getValue(), p_User->getValue());
           //sprintf(simStart, "\"%s start %s %d\" \" \" \"%s\" \" %s \"",  SimBatchString(), caseString, numProc, p_Hostname->getValue(), p_User->getValue());
		   
		   dprintf(3,"\n\nsimStart='%s'\n\n", simStart);
	}
	else if (!strcmp(s_ConnectionMethod[connMeth], "globus_gram")) {
	   
	   std::string globusrun = coCoviseConfig::getEntry("value","Module.Globus.GlobusRun", "/usr/local/globus-4.0.1/bin/globusrun-ws");
	   std::string jobfactory = coCoviseConfig::getEntry("value","Module.Globus.jobfactory","/wsrf/services/ManagedJobFactoryService");
	   
           printf("globusrun: [%s]\njobfactory: [%s]\n", globusrun.c_str(), jobfactory.c_str());

	   //snprintf(simStart, sizeof(simStart), "%s -s -Ft PBS -F https://%s%s -submit -c %s start %s %d %s", globusrun, p_Hostname->getValue(), jobfactory, SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
           snprintf(simStart, sizeof(simStart), "%s -s -Ft PBS -J -F https://%s%s -submit -c %s start %s %d %s", globusrun.c_str(), p_Hostname->getValue(), jobfactory.c_str(), SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
           printf("simstart: [%s]\n", simStart);
	}
	else if (!strcmp(s_ConnectionMethod[connMeth], "reattach")) {
	   
	   snprintf(simStart, sizeof(simStart), "()"); // FIXME
	}
	// execProcessWMI: commandLine, workingdirectory, host, user, password
	else {
           sprintf(simStart, "%s start %s %d %s", SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
        }

	setUserArg(1,simStart);
	dprintf(3, "\tPrepareSimStart: startshell=%s; simStart=%s\n",
               startshell, simStart);
	if (startshell)   free(startshell);
}


const char *Fenfloss::SimBatchString()
{
        //int local = 0;
        //int connMeth;

	const char *dp;

        //connMeth = p_ConnectionMethod->getValue();
        //local = !strcmp(s_ConnectionMethod[connMeth], "local");

	dp = p_StartScript->getValue();

	return (strdup(dp));
}


char *Fenfloss::ConnectionString()
{
	int connMeth;
	char connStr[100];

	memset(connStr, 0 , sizeof(connStr));

	connMeth = p_ConnectionMethod->getValue();
#ifndef WIN32
	if (!strcmp(s_ConnectionMethod[connMeth], "local") ||
	    !strcmp(s_ConnectionMethod[connMeth], "globus_gram") ||
	    !strcmp(s_ConnectionMethod[connMeth], "reattach"))
	   *connStr = ' ';
	else
	{
	   char user[50];
	   
	   memset(user, 0, sizeof(user));
	   if (p_User->getValue() && *p_User->getValue())
	      sprintf(user, "-l %s", p_User->getValue());
	   sprintf(connStr, "%s %s %s", s_ConnectionMethod[connMeth],
		   user, p_Hostname->getValue());
	}
#else
	sprintf(connStr, "%s", s_ConnectionMethod[connMeth]);
#endif

	return strdup(connStr);
}

bool Fenfloss::findAttribute(coDistributedObject *obj, const char *name, const char *scan, void *val) {
/*
   char **names, **contents;
   int num = obj->get_all_attributes(&names, &contents);
   for (int i = 0; i < num; i++) {
	   if (!strcmp(name, names[i])) {
		   return (sscanf(contents[i], scan, val) == 1);
	   }
   }
*/
   const char *attribute = obj->getAttribute(name);
   if (attribute)
      return (sscanf(attribute, scan, val) == 1);


   return false;
}

#ifdef YAC
void Fenfloss::paramChanged(coParam *param) {

   this->param(param->getName(), false);
}
#endif


MODULE_MAIN(VISiT, Fenfloss)
