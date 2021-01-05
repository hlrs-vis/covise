/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_EXPORT_H
#define CO_EXPORT_H

/* ---------------------------------------------------------------------- //
//                                                                        //
//                                                                        //
// Description: DLL EXPORT/IMPORT specification and type definitions      //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                             (C)2003 HLRS                               //
// Author: Uwe Woessner, Ruth Lang                                        //
// Date:  30.10.03  V1.0                                                  */

#if defined(__arm__) || defined(__APPLE__) || defined(CO_rhel3) || (defined(CO_ia64icc) && (__GNUC__ >= 4))
#define EXPORT_TEMPLATE(x)
#define EXPORT_TEMPLATE2(x, y)
#define EXPORT_TEMPLATE3(x, y, z)
#define INST_TEMPLATE1(x)
#define INST_TEMPLATE2(x, y)
#define INST_TEMPLATE3(x, y, z)
#else
#define EXPORT_TEMPLATE(x) x;
#define EXPORT_TEMPLATE2(x, y) x, y;
#define EXPORT_TEMPLATE3(x, y, z) x, y, z;
#define INST_TEMPLATE1(x) x;
#define INST_TEMPLATE2(x, y) x, y;
#define INST_TEMPLATE3(x, y, z) x, y, z;
#endif

#if defined(_WIN32) && !defined(NODLL)
#define COIMPORT __declspec(dllimport)
#define COEXPORT __declspec(dllexport)

#elif(defined(__GNUC__) && __GNUC__ >= 4 && !defined(CO_ia64icc)) || defined(__clang__)
#define COEXPORT __attribute__((visibility("default")))
#define COIMPORT COEXPORT

#else
#define COIMPORT
#define COEXPORT
#endif

#if defined(COVISE_APPL)
#define APPLEXPORT COEXPORT
#else
#define APPLEXPORT COIMPORT
#endif

#if defined(COVISE_VR_INTERACTOR)
#define VR_INTERACTOR_EXPORT COEXPORT
#else
#define VR_INTERACTOR_EXPORT COIMPORT
#endif

#if defined(COVISE_OSGVRUI)
#define OSGVRUIEXPORT COEXPORT
#else
#define OSGVRUIEXPORT COIMPORT
#endif

#if defined(SG_VRUI)
#define SGVRUIEXPORT COEXPORT
#else
#define SGVRUIEXPORT COIMPORT
#endif

#if defined(coMessages_EXPORTS)
#define COMSGEXPORT COEXPORT
#else
#define COMSGEXPORT COIMPORT
#endif

#if defined(coVRB_EXPORTS)
#define VRBEXPORT COEXPORT
#else
#define VRBEXPORT COIMPORT
#endif

#if defined(coVRBClient_EXPORTS)
#define VRBCLIENTEXPORT COEXPORT
#else
#define VRBCLIENTEXPORT COIMPORT
#endif

#if defined(coVRBServer_EXPORTS)
#define VRBSERVEREXPORT COEXPORT
#else
#define VRBSERVEREXPORT COIMPORT
#endif

#if defined(COVISE_FILE)
#define FILEEXPORT COEXPORT
#else
#define FILEEXPORT COIMPORT
#endif

#if defined(COVISE_GPU)
#define GPUEXPORT COEXPORT
#else
#define GPUEXPORT COIMPORT
#endif

#if defined(COIMAGE_EXPORT)
#define COIMAGEEXPORT COEXPORT
#else
#define COIMAGEEXPORT COIMPORT
#endif
#if defined(IMPORT_PLUGIN)
#define PLUGINEXPORT COEXPORT
#else
#define PLUGINEXPORT COIMPORT
#endif

#if defined(ROADTERRAIN_PLUGIN)
#define ROADTERRAINPLUGINEXPORT COEXPORT
#else
#define ROADTERRAINPLUGINEXPORT COIMPORT
#endif

#if defined(VEHICLE_UTIL)
#define VEHICLEUTILEXPORT COEXPORT
#else
#define VEHICLEUTILEXPORT COIMPORT
#endif

#if defined(coTrafficSimulation_EXPORTS)
#define TRAFFICSIMULATIONEXPORT COEXPORT
#else
#define TRAFFICSIMULATIONEXPORT COIMPORT
#endif

#if defined(VRML97_IMPORT_PLUGIN)
#define VRML97PLUGINEXPORT COEXPORT
#else
#define VRML97PLUGINEXPORT COIMPORT
#endif

#if defined(Vrml97Cover_EXPORTS)
#define VRML97COVEREXPORT COEXPORT
#else
#define VRML97COVEREXPORT COIMPORT
#endif

#if defined(CovisePluginUtil_EXPORTS)
#define COVISEPLUGINEXPORT COEXPORT
#else
#define COVISEPLUGINEXPORT COIMPORT
#endif

#if defined(COVISE_VRUI)
#define VRUIEXPORT COEXPORT
#else
#define VRUIEXPORT COIMPORT
#endif

#if defined(COVISE_PLUGIN_UTIL)
#define PLUGIN_UTILEXPORT COEXPORT
#else
#define PLUGIN_UTILEXPORT COIMPORT
#endif

#if defined(input_legacy_EXPORTS)
#define INPUT_LEGACY_EXPORT COEXPORT
#else
#define INPUT_LEGACY_EXPORT COIMPORT
#endif

#if defined(coOpenCOVER_EXPORTS)
#define COVEREXPORT COEXPORT
#else
#define COVEREXPORT COIMPORT
#endif

#if defined(COVISE_PFIV)
#define PFIVEXPORT COEXPORT
#else
#define PFIVEXPORT COIMPORT
#endif

#if defined(COVISE_PFOBJ)
#define PFOBJEXPORT COEXPORT
#else
#define PFOBJEXPORT COIMPORT
#endif

#if defined(COVISE_COVISE)
#define COVISEEXPORT COEXPORT
#else
#define COVISEEXPORT COIMPORT
#endif

#if defined(UTIL_EXPORTS) || defined(COVISE_UTIL)
#define UTILEXPORT COEXPORT
#else
#define UTILEXPORT COIMPORT
#endif

#if defined(COVISE_STAR)
#define STAREXPORT COEXPORT
#else
#define STAREXPORT COIMPORT
#endif

#if defined(COVISE_ENGINE)
#define ENGINEEXPORT COEXPORT
#else
#define ENGINEEXPORT COIMPORT
#endif

#if defined(COVISE_READER)
#define READEREXPORT COEXPORT
#else
#define READEREXPORT COIMPORT
#endif

#if defined(COVISE_COLORMAP)
#define CMAPEXPORT COEXPORT
#else
#define CMAPEXPORT COIMPORT
#endif

#if defined(CONFIG_EXPORT)
#define CONFIGEXPORT COEXPORT
#else
#define CONFIGEXPORT COIMPORT
#endif

#if defined(NET_EXPORT)
#define NETEXPORT COEXPORT
#else
#define NETEXPORT COIMPORT
#endif

#if defined(REG_EXPORT)
#define REGEXPORT COEXPORT
#else
#define REGEXPORT COIMPORT
#endif

/* tracker client */
#if defined(OSG_TCLIENT) && defined(_WIN32)
#define OSGTCLIENT COEXPORT
#else
#define OSGTCLIENT COIMPORT
#endif

/* backend */
#if defined(WIN32DLL_VINCEBACKEND) && defined(_WIN32)
#define WIN32_DLL_VINCEBACKEND COEXPORT
#else
#define WIN32_DLL_VINCEBACKEND COIMPORT
#endif

/* ViNCE renderer */
#if defined(VINCE_EXPORT) && defined(_WIN32)
#define VINCEEXPORT COEXPORT
#else
#define VINCEEXPORT COIMPORT
#endif

#if defined(COVISE_API) || defined(YAC_API)
#define APIEXPORT COEXPORT
#else
#define APIEXPORT COIMPORT
#endif

#if defined(COVISE_COMM) || defined(COMM_EXPORTS)
#define COMMEXPORT COEXPORT
#define COMMEXPORTONLY COEXPORT
#else
#define COMMEXPORT COIMPORT
#define COMMEXPORTONLY
#endif

#if defined(SHM_EXPORTS)
#define SHMEXPORT COEXPORT
#else
#define SHMEXPORT COIMPORT
#endif

#if defined(IMPORT_PLUGIN)
#define PLUGINEXPORT COEXPORT
#else
#define PLUGINEXPORT COIMPORT
#endif

#if defined(YAC_JAVA_API)
#define JAVAAPIEXPORT COEXPORT
#else
#define JAVAAPIEXPORT COIMPORT
#endif

#if defined(RENDERER_EXPORTS)
#define RENDEREXPORT COEXPORT
#else
#define RENDEREXPORT COIMPORT
#endif

#if defined(CRYPT_EXPORTS)
#define CRYPTEXPORT COEXPORT
#else
#define CRYPTEXPORT COIMPORT
#endif

#if defined(COVISE_THREAD) || defined(COVISE_THREADS) || defined(THREADS_EXPORTS)
#define THREADEXPORT COEXPORT
#else
#define THREADEXPORT COIMPORT
#endif

#if defined(COVISE_DMGR) || defined(DMGR_EXPORTS)
#define DMGREXPORT COEXPORT
#else
#define DMGREXPORT COIMPORT
#endif

#if defined(DO_EXPORTS) || defined(COVISE_DO)
#define DOEXPORT COEXPORT
#else
#define DOEXPORT COIMPORT
#endif

#if defined(COVISE_ALG) || defined(ALG_EXPORTS)
#define ALGEXPORT COEXPORT
#else
#define ALGEXPORT COIMPORT
#endif

#if defined(COVISE_VTK) || defined(VTK_EXPORTS)
#define VTKEXPORT COEXPORT
#else
#define VTKEXPORT COIMPORT
#endif

#if defined(COVISE_GRMSG) || defined(GRMSGEXPORT)
#define GRMSGEXPORT COEXPORT
#else
#define GRMSGEXPORT COIMPORT
#endif

#if defined(COVISE_VR_INTERACTOR)
#define VR_INTERACTOR_EXPORT COEXPORT
#else
#define VR_INTERACTOR_EXPORT COIMPORT
#endif

#if defined(COVISE_OPENVRUI)
#define OPENVRUIEXPORT COEXPORT
#else
#define OPENVRUIEXPORT COIMPORT
#endif

#if defined(CONFIG_EXPORT)
#define CONFIGEXPORT COEXPORT
#else
#define CONFIGEXPORT COIMPORT
#endif

#if defined(NET_EXPORT)
#define NETEXPORT COEXPORT
#else
#define NETEXPORT COIMPORT
#endif

#if defined(CONFIGEDITOR_EXPORT)
#define CONFIGEDITOREXPORT COEXPORT
#else
#define CONFIGEDITOREXPORT COIMPORT
#endif

#if defined(WSLIB_EXPORT)
#define WSLIBEXPORT COEXPORT
#else
#define WSLIBEXPORT COIMPORT
#endif

#if defined(UI_EXPORT)
#define UIEXPORT COEXPORT
#else
#define UIEXPORT COIMPORT
#endif

#if defined(SCA_EXPORT)
#define SCAEXPORT COEXPORT
#else
#define SCAEXPORT COIMPORT
#endif

#endif
