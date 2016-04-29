#ifdef   VATECH

#include "AxialRunner.h"
#include <General/include/log.h>

void AxialRunner::CreateMenuVATCFDSwitches(void)
{
   char *pselect[] =
   {
      "dimensionless_data",
      "prototype_data"
   };

   p_RunVATEuler = addBooleanParam(M_RUN_VATEULER,M_RUN_VATEULER);
   p_RunVATEuler->setValue(0);

   omega = addFloatParam(M_OMEGA, M_OMEGA);
   omega->setValue(0.0);
   // setup for euler grid filename
   filenameGrid = addFileBrowserParam(M_EU_GRID_FILE,"Data_file_path");
   SetFileValue(filenameGrid, COV_IHS_EULERGRIDPATH,
      ENV_IHS_EULERGRIDPATH, "*.netz");

   filenameEuler = addFileBrowserParam(M_EU_EULER_FILE,"Data_file_path");
   SetFileValue(filenameEuler, COV_IHS_EULERDATAPATH,
      ENV_IHS_EULERDATAPATH, "*.euler");

   p_RotateGrid = addBooleanParam(M_ROTATE_GRID,M_ROTATE_GRID);
   p_RotateGrid->setValue(0);

   p_OpPoint = addChoiceParam(M_OPERATING_POINT,M_OPERATING_POINT);
   p_OpPoint->setValue(2,pselect,0);

   p_nED = addFloatParam(M_DIMLESS_N,M_DIMLESS_N);
   p_nED->setValue(0.0);

   p_QED = addFloatParam(M_DIMLESS_Q,M_DIMLESS_Q);
   p_QED->setValue(0.0);

   p_Head = addFloatParam(M_HEAD,M_HEAD);
   p_Head->setValue(0.0);
   p_Head->hide();

   p_Discharge = addFloatParam(M_FLOW,M_FLOW);
   p_Discharge->setValue(0.0);
   p_Discharge->hide();

   p_ProtoSpeed = addFloatParam(M_SPEED,M_SPEED);
   p_ProtoSpeed->setValue(0.0);
   p_ProtoSpeed->hide();

   p_ProtoDiam = addFloatParam(M_DIAMETER,M_DIAMETER);
   p_ProtoDiam->setValue(0.0);
   p_ProtoDiam->hide();

   p_alpha = addFloatParam(M_ALPHA,M_ALPHA);
   p_alpha->setValue(0.0);
   p_alpha->hide();
}


void AxialRunner::SetFileValue(coFileBrowserParam *f, const char *co, const char *en, const char *pat)
{
   const char *dp;
   char *pfn;

   dp = CoviseConfig::getEntry(co);
   if(dp == NULL)
   {

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
      dp = dataPath.c_str();
   }

   if ((pfn = CreateFileNameParam(dp, en, "nofile", CFNP_ENV_ONLY)))
   {
      dprintf(3, "Startpath: %s\n", pfn);
      f->setValue(pfn, pat);
      free(pfn);
   }
   else
   {
      dprintf(0, "WARNING: No startpath (%s, %s)\n", co, en);
   }
}
#endif                                            // VATECH
