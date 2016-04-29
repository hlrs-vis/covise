#include <appl/ApplInterface.h>
#include <api/coFeedback.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include "VisDomain.h"
#include <stdlib.h>
#include <stdio.h>
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
#include <iostream>

#ifndef  DIM
#define  DIM(x)   (sizeof(x)/sizeof(*(x)))
#endif

#define VERBOSE
#undef  MESSAGES

VisDomain::VisDomain(int  argc,  char  *argv[])
: coModule(argc, argv, "Visualize Domaindecomposition")
      //:coModule(argv[0], "Simulation coupling")
{
         //const char *dp;


   // Parameters
   // Input ports : yet only parallel ones allowed
   p_grid    = addInputPort("distGrid","UnstructuredGrid","Distributed Grid");
   p_boco    = addInputPort("distBoco1","USR_DistFenflossBoco","Distributed Boundary Cond");
   p_boco2   = addInputPort("distBoco2","USR_DistFenflossBoco","Distributed Boundary Cond");
   p_boco2->setRequired(0);
   p_in_bcin = addInputPort("bcin","Polygons","Boundary surface of inlet(IN)");

   // Output ports
   p_blocknum   = addOutputPort("blocknum","Float","Blocknumbers");

}

int VisDomain::compute(const char *)
{


   // Find out, whether we have to re-start sim
   coDoSet *grid  = (coDoSet*) p_grid->getCurrentObject();
   
   
   //Not used and generates C4189
   //int use_boco2 = 0;
   //coDoSet *boco  = (coDoSet*) p_boco->getCurrentObject();
   //coDoSet *boco2 = (coDoSet*) p_boco2->getCurrentObject();

  /* if (grid)
   {
      // we have a new grid input object
      const char *gridName = grid->getName();
      if ( strcmp(gridName,d_distGridName)!=0 )
      {
         delete [] d_distGridName;
         d_distGridName = strcpy ( new char[strlen(gridName)+1] , gridName );
      }
   }*/
   //float *inx_coord, *iny_coord, *inz_coord;
   //int *invertices; //, *inpolygons;
   int numProc, numElem;

   const coDistributedObject *const *gridArr = grid->getAllElements(&numProc);

   const coDistributedObject *const **procGrid;

   const coDoIntArr *gridDim;
   int numcells=10;
   float *domainnumbers;



      ///////////////////////////////////////////////////////////////////////
      // send dimensions
      int *idata,size;
      int i; //,j;

      procGrid = new const coDistributedObject *const*[numProc];

      for (i=0;i<numProc;i++)
      {
         procGrid[i] = ((const coDoSet*)gridArr[i])->getAllElements(&numElem);

         gridDim = (const coDoIntArr *) procGrid[i][0];
         idata = gridDim->getAddress();
         size  = gridDim->getDimension(0) * sizeof(int);
         numcells = idata[2];
      }


   coDoFloat *res = new coDoFloat(p_blocknum->getObjName(),numcells);
   res->getAddress(&domainnumbers);
      for (i=0;i<numcells;i++)
         domainnumbers[i]=-1;

      ///////////////////////////////////////////////////////////////////////
      // send grids
      //float *fdata;
      int n;
      int cell;

      for (i=0;i<numProc;i++)
      {
         
            coDoIntArr *intArr = (coDoIntArr *) procGrid[i][0];
            idata = intArr->getAddress();
            int localnumnods = idata[6];
            localnumnods--;
            
            intArr = (coDoIntArr *) procGrid[i][3];
            idata = intArr->getAddress();
            size = intArr->getDimension(0) * sizeof(int);
            for(n=0;n<localnumnods;n++)
            {
               cell=idata[n]-1;
               if(domainnumbers[cell]>=0)
               domainnumbers[cell]=(float)numProc;
               else
               domainnumbers[cell]=(float)i;
            }
      }

/* 
         for (j=13;j<15;j++)                      // displ_wert, pres_wert
         {
            sendInfo("Sending RB-Data ...");
            coDoFloat *floatArr = (coDoFloat *) procBoco[i][j];
            floatArr->getAddress(&fdata);
            size = floatArr->getNumPoints() * sizeof(float);
            if (size)
            {
               printf("VisDomain-Mod(%d)   BC: Sending (cpu=%d, j=%d), size = %d Bytes\n", __LINE__, i, j, size);
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
            }
         }*/
      


   p_blocknum->setCurrentObject(res);
   return SUCCESS;

}

MODULE_MAIN(VISiT, VisDomain)
