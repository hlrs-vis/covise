#ifndef _FENFLOSS_H
#define _FENFLOSS_H

#ifndef YAC
#include <appl/ApplInterface.h>
#include <api/coFeedback.h>
#else
#include <util/coviseCompat.h>
#endif
#include <api/coSimLibComm.h>
#include <api/coSimLib.h>

#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include <General/include/iso.h>

using namespace covise;

#define BC_SELECTS   7
#define UD_SELECTS   7
#ifndef WIN32
#define CM_MAX    7
#else
#define CM_MAX    8
#endif

struct commandMsg {
   int command;
   int size;
};


class Fenfloss : public coSimLib
{
   COMODULE

   private:

      //////////  member functions

#ifdef YAC
      virtual void paramChanged(coParam *param);
#endif

      virtual int compute(const char *port);
      virtual void param(const char *, bool inMapLoading);
      virtual void postInst();
#ifndef YAC
      virtual void quit();
#else
      virtual int quit();
#endif
      virtual char *ConnectionString(void);
      virtual const char *SimBatchString(void);
      virtual void PrepareSimStart(int numProc);
      virtual void StopSimulation(void);
      virtual int endIteration();
	  virtual void CreateUserMenu();
      
      bool findAttribute(coDistributedObject *obj, const char *name, const char *scan, void *val);
      
      // current step number, -1 if not connected
      int stepNo;

      // connections ...
      coChoiceParam       *p_ConnectionMethod;
      coStringParam       *p_User;
      coStringParam       *p_Hostname;
      coStringParam       *p_Port;

      // globus
      coStringParam       *p_Discovery;
      coChoiceParam       *p_Simulations;

      // client simulation startup script (path&filename)
      coStringParam       *p_StartScript;

      // simulation application, needed by flow_covise startup-script
      coChoiceParam       *p_simApplication;

      // start directory of the simulation: there we find flow and flow.stf
	  // fl: not really, anymore
      coStringParam       *p_dir;

      coBooleanParam      *p_useInitial;
      coBooleanParam      *p_stopSim;
      coBooleanParam      *p_pauseSim;
      coBooleanParam      *p_detachSim;

      coBooleanParam      *p_GetSimData;
      coIntScalarParam    *p_updateInterval;
      coIntScalarParam    *p_uDelay;

	  // Simulation steering parameters
	  coIntScalarParam    *p_global_iterations;
	  coIntScalarParam    *p_num_time_steps;
	  coFloatParam        *p_timestep;
	  coFloatParam        *p_norm_vel;
	  coFloatParam        *p_norm_len;
	  coFloatParam        *p_lambda;
	  coFloatParam        *p_walldist;
	  coFloatParam        *p_density;
	  coFloatParam        *p_viscosity;
	  coFloatVectorParam  *p_relaxglob;
	  coFloatVectorParam  *p_relaxloca;

      coInputPort         *p_grid, *p_boco, *p_boco2, *p_in_bcin;
      coOutputPort        *p_velocity, *p_press, *p_turb, *p_out_bcin;

	  // additional parameters, if not already included in attributes
      coBooleanParam      *p_paramperiodic;
      coBooleanParam      *p_paramrotating;
      coStringParam       *p_paramperitext;
      coStringParam       *p_paramwalltext;
	  coFloatParam        *p_paramrevs;
	  coIntScalarParam    *p_paramnob;
	  



      int numProc;

      // boundary conditions
      int numbc;
      float *bcrad, *bcvu, *bcvm;

      // boco2 data
      int use_boco2, *boco2_num_int, *boco2_num_float;
      int **boco2_idata;
      float **boco2_fdata;

      // the name of the last distGrid port object
      char           *d_distGridName;

      // the name of the last distBoco port object
      char           *d_distBocoName;

      char *s_ConnectionMethod[CM_MAX];
      std::vector<char *> s_simApplication;
      char **s_Simulations;

   public:

      Fenfloss(int argc, char *argv[]);
      virtual ~Fenfloss() {}

};
#endif                                            // _FENFLOSS_H
