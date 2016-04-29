#include <stdio.h>
#include <CreateFileNameParam.h>
#include <include/log.h>
#include "ReadVATECH.h"
#include "EuGri.h"

#define  ENV_IHS_DEBUGLEVEL      "IHS_DEBUGLEVEL"

int main(int argc, char *argv[])
{
   ReadVATECH *readeuler = NULL;

   readeuler = new ReadVATECH();
   readeuler->start(argc,argv);

   return 0;
}


ReadVATECH::ReadVATECH()
:coModule("Read VATECH Euler data")
{
   const char *dp;
   const char *df;
   char *pfn;

   dprintf(1, "Entering ReadVATECH() ...\n");
   // loglevel and debug files ...
   SetDebugLevel(0);
   if (getenv(ENV_IHS_DEBUGLEVEL))
   {
      SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
   }
   else
      dprintf(0, "WARNING: IHS_DEBUGLEVEL is not set. (now setting to 0)\n");
   dp = CoviseConfig::getEntry("IHS.DebPath");
   df = (CoviseConfig::getEntry("IHS.DebFile")
      ? CoviseConfig::getEntry("IHS.DebFile") : "ReadEuler.deb");
   if ((pfn = CreateFileNameParam(dp, "IHS_DEBPATH", df, CFNP_NORM)) != NULL)
   {
      dopen(pfn);
      free(pfn);
   }
   dprintf(0, "**************************************************\n");
   dprintf(0, "* Read-Euler module                              *\n");
   dprintf(0, "* (c) 2003 by University of stuttgart - IHS      *\n");
   dprintf(0, "**************************************************\n");

   // setup for euler grid filename
   filenameGrid = addFileBrowserParam("GridFile","Data file path");
   dp = (CoviseConfig::getEntry(COV_IHS_EULERGRIDPATH)
      ? CoviseConfig::getEntry(COV_IHS_EULERGRIDPATH) : getenv("HOME"));
   if ((pfn = CreateFileNameParam(dp, ENV_IHS_EULERGRIDPATH, "nofile",
      CFNP_ENV_ONLY)))
   {
      dprintf(0, "Startpath: %s\n", pfn);
      filenameGrid->setValue(pfn, "*.netz");
      free(pfn);
   }
   else
   {
      dprintf(0, "WARNING: No startpath for EulerGrid (%s, %s)\n",
         COV_IHS_EULERGRIDPATH, ENV_IHS_EULERGRIDPATH);
   }

   filenameEuler = addFileBrowserParam("EulerFile","Data file path");
   dp = (CoviseConfig::getEntry(COV_IHS_EULERDATAPATH)
      ? CoviseConfig::getEntry(COV_IHS_EULERDATAPATH) : getenv("HOME"));
   if ((pfn = CreateFileNameParam(dp, ENV_IHS_EULERDATAPATH, "nofile",
      CFNP_ENV_ONLY)))
   {
      dprintf(0, "Startpath: %s\n", pfn);
      filenameEuler->setValue(pfn, "*.euler");
      free(pfn);
   }
   else
   {
      dprintf(0, "WARNING: No startpath for EulerData (%s, %s)\n",
         COV_IHS_EULERDATAPATH, ENV_IHS_EULERDATAPATH);
   }

   omega     = addFloatParam("Omega", "Omega");
   omega->setValue(0.0);

   gridnorm     = addFloatParam("GridNormValue", "Grid norm");
   gridnorm->setValue(1.0);

   // the output ports
   gridOutPort     = addOutputPort("grid","coDoStructuredGrid","structured grid");
   velocityOutPort = addOutputPort("velocity","Vec3","velocity data");
   relVelocityOutPort = addOutputPort("relVelocity","Vec3","Relativ velocity data");
   pressureOutPort = addOutputPort("pressure","coDoFloat","pressure data");
   dprintf(1, "Leaving ReadVATECH() ...\n");
}


void ReadVATECH::postInst()
{
   dprintf(1, "Entering PostInst()\n");
   gridnorm->show();
   omega->show();
   dprintf(1, "Leaving PostInst()\n");
}


#ifdef   NICHT_RAUS
ReadVATECH::~ReadVATECH()
{
}
#endif

int ReadVATECH::compute(const char *)
{
   struct EuGri *eu;
   char *grid_fn;
   char *euler_fn;
   const char *gridName;
   const char *velocityName;
   const char *relVelocityName;
   const char *pressureName;
   coDoStructuredGrid *gridObj;
   coDoFloat *pressureObj;
   coDoVec3 *velocityObj;
   coDoVec3 *relVelocityObj;

   dprintf(1, "Entering compute() ...\n");
   // read the file browser parameter
   grid_fn  = (char *)filenameGrid->getValue();
   euler_fn = (char *)filenameEuler->getValue();

   // read grid and results
   if ((eu = ReadEuler(grid_fn, euler_fn, omega->getValue())) == NULL)
   {
      return STOP_PIPELINE;
   }
   if (eu->norm != gridnorm->getValue())
   {
      NormEulerGrid(eu, gridnorm->getValue());
   }

   // create the structured grid object for the grid
   gridName     = gridOutPort->getObjName();
   if(gridName != NULL && eu->x && eu->y && eu->z)
   {
      gridObj = new coDoStructuredGrid(gridName, eu->i, eu->j, eu->k, eu->x, eu->y, eu->z);
      gridOutPort->setCurrentObject(gridObj);
   }

   // create the structured data object for the pressure
   pressureName = pressureOutPort->getObjName();
   if(pressureName != NULL && eu->p)
   {
      pressureObj = new coDoFloat(pressureName, eu->i, eu->j, eu->k, eu->p);
      pressureOutPort->setCurrentObject(pressureObj);
   }

   // create the unstructured grid object for the velocity
   velocityName = velocityOutPort->getObjName();
   if(velocityName != NULL && eu->u && eu->v && eu->w)
   {
      velocityObj = new coDoVec3(velocityName, eu->i, eu->j, eu->k, eu->u, eu->v, eu->w);
      velocityOutPort->setCurrentObject(velocityObj);
   }

   // create the unstructured grid object for the relativ velocity
   relVelocityName = relVelocityOutPort->getObjName();
   if(relVelocityName != NULL && eu->ur && eu->vr && eu->wr)
   {
      relVelocityObj = new coDoVec3(relVelocityName, eu->i, eu->j, eu->k, eu->ur, eu->vr, eu->wr);
      relVelocityOutPort->setCurrentObject(relVelocityObj);
   }
   dprintf(1, "Leaving compute() ...\n");

   return CONTINUE_PIPELINE;
}
