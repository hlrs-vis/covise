// create user menues

#include "Fenfloss.h"

#include <config/CoviseConfig.h>
#include <General/include/log.h>
#include <sys/types.h>
#ifndef WIN32
#include <pwd.h>
#include <unistd.h>
#endif

void Fenfloss::CreateUserMenu(void)
{
	const char *dp = NULL;
#ifndef WIN32
	uid_t myuid;
	struct passwd *mypwd;
#endif
	p_ConnectionMethod = addChoiceParam("Connection_Method", "ConnectionMethod");
	s_ConnectionMethod[0] = strdup("local");
	s_ConnectionMethod[1] = strdup("ssh");
	s_ConnectionMethod[2] = strdup("rsh");
	s_ConnectionMethod[3] = strdup("rdaemon");
	s_ConnectionMethod[4] = strdup("echo");
	s_ConnectionMethod[5] = strdup("globus_gram");
	s_ConnectionMethod[6] = strdup("reattach");
#ifdef WIN32
	s_ConnectionMethod[7] = strdup("WMI");
#endif

	p_ConnectionMethod->setValue(CM_MAX, s_ConnectionMethod, 0);

	p_Hostname = addStringParam("Hostname", "Hostname");
	p_Hostname->setValue("localhost");

	p_Port = addStringParam("Port", "Port");
	p_Port->setValue(coCoviseConfig::getEntry("value","Module.Fenfloss.PORTS","31500 31510").c_str());

	p_StartScript = addStringParam("Startup_Script_on_Client", "Startup Script on Client");
	p_StartScript->setValue(coCoviseConfig::getEntry("value","Module.IHS.StartScript","~/bin/fen_covise").c_str());

	p_simApplication = addChoiceParam("Application", "Application");

        coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries("Module.Fenfloss", "Application");
        const char ** entries = e.getValue();
        while (entries && *entries) {
           entries++;
           const char *sim = *entries++;
           if (sim) {
              s_simApplication.push_back(strdup(sim));
           }
        }

        char **sims = new char*[s_simApplication.size()];
        for (int index = 0; index < s_simApplication.size(); index ++)
           sims[index] = s_simApplication[index];

	p_simApplication->setValue(s_simApplication.size(), sims, 0);
   delete []sims;
   sims = NULL;

#ifndef WIN32
	// get user automatically
	myuid = getuid();
	while( (mypwd = getpwent()) ) {
		if (mypwd->pw_uid == myuid) {
			dprintf(2,"You are ");
			dprintf(2,"%s, ",mypwd->pw_name);
			dprintf(2,"%d\n",mypwd->pw_uid);
			break;
		}
	}
#endif

	p_User = addStringParam("User", "User");
#ifndef WIN32
	p_User->setValue(mypwd->pw_name);
#else
   p_User->setValue(getenv("USERNAME"));
#endif

	p_Discovery = addStringParam("Globus_Simulation_Discovery", "Globus Simulation Discovery");
	p_Discovery->setValue("https://localhost:8443/wsrf/services/simulation/SimulationService");

	p_Simulations = addChoiceParam("Running_Simulations", "Running Simulations");
	const char *sim[1] = { "Please insert the URI to the Simulation Discovery Service above" };
	p_Simulations->setValue(1, sim, 0);
	p_useInitial = addBooleanParam("Use_initial_solution","use Initial Solution");
	p_useInitial->setValue(0);

	p_stopSim = addBooleanParam("Stop_simulation","Stop Simulation");
	p_stopSim->setValue(0);

	p_pauseSim = addBooleanParam("Pause_simulation", "Pause Simulation");
	p_pauseSim->setValue(0);

	p_GetSimData = addBooleanParam("Get_Simulation_Data", "Get Simulation Data");
	p_GetSimData->setValue(0);

	p_detachSim = addBooleanParam("Detach_Simulation", "Detach Simulation");
	p_detachSim->setValue(0);

	p_updateInterval = addInt32Param("Update_Interval", "update Interval");
	p_updateInterval->setValue(5);

	p_uDelay     = addInt32Param("uDelay", "send first result after n interations");
	p_uDelay->setValue(2);

	// FENFLOSS iteration parameters
	p_timestep     = addFloatParam("time_step", "time_step");
	p_timestep->setValue(0.1f);
	p_norm_vel    = addFloatParam("norm_velo", "norm_velo");
	p_norm_vel->setValue(10.0f);
	p_norm_len    = addFloatParam("norm_length", "norm_length");
	p_norm_len->setValue(1.0f);
	p_lambda      = addFloatParam("lambda", "lambda");
	p_lambda->setValue(0.1f);
	p_walldist    = addFloatParam("walldist", "walldist");
	p_walldist->setValue(0.05f);
	p_density     = addFloatParam("density", "density");
	p_density->setValue(1000.0f);
	p_viscosity   = addFloatParam("viscosity", "viscosity");
	p_viscosity->setValue(1.e-5f);
	p_global_iterations = addInt32Param("global_iterations", "global_iterations");
	p_global_iterations->setValue(100);
	p_num_time_steps = addInt32Param("num_time_steps", "num_time_steps");
	p_num_time_steps->setValue(100);
	p_relaxglob   = addFloatVectorParam("global_relaxation","global_relaxation");
	p_relaxglob->setValue(0.5f,0.5f,0.5f);
	p_relaxloca   = addFloatVectorParam("local_relaxation","local_relaxation");
	p_relaxloca->setValue(0.7f,0.7f,0.7f);

	// additional parameters, if not already included in attributes
	p_paramperiodic =addBooleanParam("Coupling_geometry","param_periodic");
	p_paramperiodic->setValue(0);
	p_paramperitext =addStringParam("Setup_coupling","param_periotext");
	p_paramperitext->setValue("111,nomatch,110,120,perio_rota,99,3");

	p_paramnob  =addInt32Param("Num_periodic_parts","param_number_of_blades");
	p_paramnob->setValue(2);

	p_paramrotating =addBooleanParam("Rotating","param_rotating");
	p_paramrotating->setValue(0);
	p_paramwalltext =addStringParam("Wall_rotation","param_walltext");
	//p_paramwalltext->setValue("115,wand_omega,11,0.0,0.0,99.99");
	p_paramwalltext->setValue("");
   
	p_paramrevs =addFloatParam("RPMs","param_revolutions");
	p_paramrevs->setValue(1.0);

	p_dir   = addStringParam("directory","Starting directory");
	if(dp)
		p_dir->setValue(dp);

}
