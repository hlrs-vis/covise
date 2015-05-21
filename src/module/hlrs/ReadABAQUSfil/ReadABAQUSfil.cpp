/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

   * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2012 HLRS  **
 **                                                                          **
 ** Description:                                                             **
 **                                                                          **
 ** Name:     ReadABAQUSfil                                                  **
 ** Category: I/0 Module                                                     **
 **                                                                          **
 ** Author: Ralf Schneider	                                             **
 **                                                                          **
 ** History:  								     **
 **                					       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

// Standard headers
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#ifdef WIN32
#include <stdint.h>
#else
#include <inttypes.h>
#endif
#include <sys/stat.h>
// COVISE data types
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
// this includes our own class's headers
#include "ReadABAQUSfil.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static const char *sigma[] = { "sigma_xx", "sigma_yy", "sigma_zz",
                               "sigma_xy", "sigma_xz", "sigma_yz" };
static const char *epsilon[] = { "epsilon_xx", "epsilon_yy", "epsilon_zz",
                                 "epsilon_xy", "epsilon_xz", "epsilon_yz" };
static const char *equivalence[] = { "von_Mises" };

ReadABAQUSfil::ReadABAQUSfil(int argc, char *argv[])
  : coSimpleModule(argc, argv, "Read ABAQUS .fil result file")
{

  // Parameters
  p_filFile = addFileBrowserParam("fil_File", "ABAQUS .fil result File");
  p_filFile->setValue("xxx.fil", "*.fil");

  const char *no_info[] = { "No info" };

  p_elemres = addChoiceParam("Element_Result", "Select tensor element results to be loaded");
  p_elemres->setValue(1, no_info, 0);

  p_telemres = addChoiceParam("Tensor_Element_Result", "Select tensor element results to be loaded");
  p_telemres->setValue(1, no_info, 0);

  p_nodalres = addChoiceParam("Nodal_Result", "Select nodal results to be loaded");
  p_nodalres->setValue(1, no_info, 0);

  p_selectedSets = addStringParam("Selected_Sets","Give numbers of element sets to be selected");
  p_selectedSets->setValue("All");

  // Ports
  p_gridOutPort    = addOutputPort("p_gridOutPort", "UnstructuredGrid", "Read unstructured grid");
  p_eresOutPort = addOutputPort("p_eresOutPort", "Float", "Loaded element scalar results");
  p_tresOutPort = addOutputPort("p_tresOutPort", "Float", "Loaded element tensor results");
  p_nresOutPort = addOutputPort("p_nresOutPort", "Float", "Loaded nodal results");

  p_SetgridOutPort  = addOutputPort("p_SetgridOutPort",  "UnstructuredGrid", "Read unstructured sets grid");
  p_SetgridResPort  = addOutputPort("p_SetgridResPort",  "Float", "Loaded sets element scalar results");
  p_SetgridTResPort = addOutputPort("p_SetgridTResPort", "Float", "Loaded sets element tensor results");
  p_SetgridnResPort = addOutputPort("p_SetgridnResPort", "Float", "Loaded sets nodal results");

  // not yet in compute call
  computeRunning = false;

  // Init .fil file storage
  fil_array = NULL;
  data_length = 0;

  // Set initial .fil filename to nothing
  fil_name = "xxx.fil";
    
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  param() is called every time a parameter is changed (really ??)
// ++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadABAQUSfil::param(const char *paramName, bool in_map_loading)
{
#ifdef WIN32
  struct _stat attribut;
#else
  struct stat attribut;
#endif
  int64_t fil_size;

  FILE *fd_fil;
  int ii, jj, kk, cp;

  int rec_struct[3000];
  int rec_length, rec_type, no_recs, trec_len;

  int file_offset = 4;

  double *tmp_d;
  int *tmp_i;

  int ii_sets, ii_labcro, ii_fup;

  const int dbg=1;
   
  // *************************************************************************
  // If param is called in case of fil file selection ************************
  if ((0 == strcmp(p_filFile->getName(), paramName)) &&
#ifdef _WIN32
      (0 == _stat(p_filFile->getValue(), &attribut)) &&
#else
      (0 == stat(p_filFile->getValue(), &attribut)) &&
#endif
      (0 != strcmp(p_filFile->getValue(), fil_name)) && (!computeRunning))
    {

      sendInfo("Searching for available results and sets. Please be patient ... ");

      // Allocate fil array and open fil file ***
      fil_size = ((int)attribut.st_size - 8) / 8;

      if ( dbg == 1 ) {
	printf("Need %ld 8-byte elements to store file \n", (long int)fil_size);
      }
      // In case a .fil file was already loaded free memory ***
      if (fil_array != NULL)
        {
	  free(fil_array);
        }

      fil_array = (int64_t *)malloc(fil_size * sizeof(int64_t));

      tmp_d = (double *)fil_array;
      tmp_i = (int *)fil_array;

      // Open ABAQUS result file *************************************
      if ((fd_fil = fopen(p_filFile->getValue(), "rb")) == NULL)
        {
	  sendError("Failed to open ABAQUS result file '%s'",
		    p_filFile->getValue());
	  return;
        }
      else
        {

	  // Skip first record control word in .fil file *****************
	  fseek(fd_fil, file_offset, SEEK_SET);

	  fread(fil_array, sizeof(int64_t), fil_size, fd_fil);

	  // Close ABAQUS result file ************************************
	  fclose(fd_fil);

	  // Eliminate record control words from fil_array ***************
	  ii = 513;
	  cp = 512;

	  while (ii < fil_size)
            {
	      for (jj = 1; jj <= 512; ++jj)
                {
		  fil_array[cp] = fil_array[ii];
		  cp = cp + 1;
		  ii = ii + 1;
                };
	      ii = ii + 1;
            };

	  data_length = cp - 1;

	  // *************************************************************
	  // Analyze record structure ************************************

	  // Init record count ************
	  for (ii = 0; ii <= 2999; ++ii)
            {
	      rec_struct[ii] = 0;
            }

	  no_recs = 0;
	  jj = 0;

	  // Count records per type ************************
	  while (jj < data_length)
            {

	      rec_length = fil_array[jj];
	      rec_type   = fil_array[jj + 1];

	      rec_struct[rec_type] = rec_struct[rec_type] + 1;

	      // Parse the job-header record *****************
	      if (rec_type == 1921)
                {

		  jobhead.version[8] = 0;
		  *(int64_t *)jobhead.version = fil_array[jj + 2];

		  // Parse date and time of job ****
		  // jobhead.date =
		  // jobhead.time =

		  jobhead.typical_el_length = float(fil_array[jj + 8]);
                }

	      jj = jj + rec_length;

	      no_recs = no_recs + 1;
            }

	  jobhead.no_nodes = rec_struct[1901];
	  jobhead.no_elems = rec_struct[1900];

	  jobhead.no_node_sets = rec_struct[1931];
	  jobhead.no_elem_sets = rec_struct[1933];

	  jobhead.no_steps = rec_struct[2000];


	  // Log ***************************************
	  printf("===========================================================\n");
	  printf("%s\n", "Rec # - No.");
	  for (ii = 0; ii <= 2999; ++ii)
            {
	      if (rec_struct[ii] > 0)
                {
		  printf("%5d - %d\n", ii, rec_struct[ii]);
                }
            }

	  // **************************************************
	  // Get set names ************************************
	  jj        =  0;
	  ii_sets   =  0;
	  ii_labcro =  0;

	  vsets.clear();
	  vsets.reserve(rec_struct[1931]+rec_struct[1933]);

	  vector<tCref> vcref;
	  vsets.reserve(rec_struct[1940]);

	  tSets set; 
	  tCref cref;

	  vsteps.reserve(rec_struct[2000]);

	  tStephead step;
	  step.active = 0;

	  while (jj < data_length)
	    {

	      rec_length = fil_array[jj];
	      rec_type = fil_array[jj + 1];

	      switch ( rec_type ) { 

	      case 1931 :
		// Found node set *********************************
		{

		  set.type = "Nodes";

		  // Get set name as Char(Len=8) and Integer ********
		  set.cname = string((char*)&fil_array[jj+2],8);
		    
		  istringstream convert(set.cname);

		  if ( !(convert >> set.cref ) ) {
		    set.cref = -1;
		    printf("Set no %d without cref. Name = %s\n",ii_sets,set.cname.c_str());
		  }
		    
		  // Get Nodes in set ***************************************
		  // Note that the number of external node numbers in the set
		  // is stored in no_elems
		  set.no_elems = rec_length - 3;
		    
		  // Check for node set continuations ***********************
		  kk = jj + rec_length;
		    
		  while ( fil_array[kk+1] == 1932 )
		    {
		      trec_len  = fil_array[kk];
		      rec_type  = fil_array[kk+1];
			
		      set.no_elems =  set.no_elems + trec_len - 2;
		      kk = kk + trec_len;
			
		    }

		  vsets.push_back(set);

		  ii_sets = ii_sets + 1;
		  break;
		    
		}
	      case 1933 :
		// Found element set ******************************
		{
		  set.type  = "Elems";

		  // Get set name as Char(Len=8) and Integer ********
		  set.cname = string((char*)&fil_array[jj+2],8);

		  istringstream convert(set.cname);

		  if ( !(convert >> set.cref ) ) {
		    set.cref = -1;
		    printf("Set no %d without cref. Name = %s\n",ii_sets,set.cname.c_str());
		  }
		    
		  // Get Nodes in set *****************************
		  set.no_elems = rec_length - 3;

		  // Check for node set continuations *********************
		  kk = jj + rec_length;

		  while ( fil_array[kk+1] == 1934 )
		    {
		      trec_len  = fil_array[kk];
		      rec_type  = fil_array[kk+1];
			
		      set.no_elems =  set.no_elems + trec_len - 2;
		      kk = kk + trec_len;
			
		    }

		  vsets.push_back(set);

		  ii_sets = ii_sets + 1;
		  break;
		}
	      case 1940 :
		// Found label cross reference ********************

		cref.cref = fil_array[jj+2];
		cref.name = "";
		  
		for  (ii = 1; ii <= rec_length-3; ++ii) {
		  cref.name += string((char*)&fil_array[jj+2+ii],8);
		}

		vcref.push_back(cref);

		break;
		
	      case 2000 :
	
		step.start = jj;
		step.active = 1;

		step.Total_time                  = float(tmp_d[jj + 2]);
		step.Step_time                   = float(tmp_d[jj + 3]);
		step.Max_creep_strainrate_ratio  = float(tmp_d[jj + 4]);
		step.Sol_dep_ampl                = float(tmp_d[jj + 5]);
		step.Procedure_type              = fil_array[jj + 6];
		step.Step_no                     = fil_array[jj + 7];
		step.Inc_no                      = fil_array[jj + 8];
		step.perturb_flag                = fil_array[jj + 9];
		step.Load_prop_factor            = float(tmp_d[jj + 10]);
		step.Frequency                   = float(tmp_d[jj + 11]);
		step.Time_inc                    = float(tmp_d[jj + 12]);
		
		break;

	      case 2001 :

		step.end = jj;

		if (step.active == 1) {
		  vsteps.push_back(step);
		}

		step.active = 0;  
		break;

	      }

	      jj = jj + rec_length;

	    }

	  // log steps and increments *****************************************
	  for (vector<tStephead>::iterator it = vsteps.begin(); it != vsteps.end(); ++it) {
	    printf("%3.3f %3.3f %3.3f %3.3f %3d %3d %3d %3d %3.3f %3.3f %3.3f\n",
	  	   (*it).Total_time  ,
	  	   (*it).Step_time   ,
	  	   (*it).Max_creep_strainrate_ratio,
	  	   (*it).Sol_dep_ampl   ,
	  	   (*it).Procedure_type ,
	  	   (*it).Step_no        ,
	  	   (*it).Inc_no         ,
	  	   (*it).perturb_flag   ,
	  	   (*it).Load_prop_factor,
	  	   (*it).Frequency       ,
	  	   (*it).Time_inc);
	  }

	  // Determine Names **************************************
	  for (vector<tSets>::iterator it = vsets.begin(); it != vsets.end(); ++it) {
	      
	    if ( (*it).cref == -1) {
	      (*it).name = (*it).cname;
	    } else {
	      (*it).name = vcref[(*it).cref-1].name;
	    }

	  }

	  // Log - Sets and their names with cross references ***************
	  printf("===========================================================");
	  printf("===========================================================\n");
	  printf("%10s %10s %9s %s\n","Set Type","C8-Cref","Int-Cref","External Set-Name");
	  printf("-----------------------------------------------------------");
	  printf("-----------------------------------------------------------\n");
	  for (vector<tSets>::iterator it = vsets.begin(); it != vsets.end(); ++it) {
	    printf("%9d %10s %10s %9d %s\n",
		   it-vsets.begin(),
		   (*it).type.c_str(),
		   (*it).cname.c_str(),
		   (*it).cref,(*it).name.c_str());
	  }

	  // **************************************************
	  // Setup drop down list for element results *********
	  int ii_choiseVals = 0;

	  if (rec_struct[11] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 6;
            }
	  if (rec_struct[21] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 6;
            }
	  // Equivalence Stresses ************
	  if (ii_choiseVals > 0)
            {
	      // Von Mises *********************
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  const char **choiseVals = new const char *[ii_choiseVals];
	  ii_choiseVals = 0;

	  // Stresses *********************************
	  if (rec_struct[11] > 0)
            {
	      for (ii = 0; ii < 6; ++ii)
                {
		  choiseVals[ii_choiseVals] = sigma[ii];
		  ii_choiseVals = ii_choiseVals + 1;
                }
            }

	  // Strains **********************************
	  if (rec_struct[21] > 0)
            {
	      for (ii = 0; ii < 6; ++ii)
                {
		  choiseVals[ii_choiseVals] = epsilon[ii];
		  ii_choiseVals = ii_choiseVals + 1;
                }
            }

	  // Equivalence Stresses *********************
	  if (ii_choiseVals > 0)
            {

	      // Von Mises ******************************
	      choiseVals[ii_choiseVals] = equivalence[0];

	      ii_choiseVals = ii_choiseVals + 1;
            }

	  p_elemres->setValue(ii_choiseVals, choiseVals, 0);

	  // **************************************************
	  // Setup drop down list for tensor element res. *****
	  ii_choiseVals = 0;

	  if (rec_struct[11] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 1;
            }
	  if (rec_struct[21] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  const char **tchoiseVals = new const char *[ii_choiseVals];
	  ii_choiseVals = 0;

	  // Stress tensor ****************************
	  if (rec_struct[11] > 0)
            {
	      tchoiseVals[ii_choiseVals] = "Stress";
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  // Reaction forces **************************
	  if (rec_struct[21] > 0)
            {
	      tchoiseVals[ii_choiseVals] = "Strain";
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  p_telemres->setValue(ii_choiseVals, tchoiseVals, 0);

	  // **************************************************
	  // Setup drop down list for nodal results ***********
	  ii_choiseVals = 0;

	  // Displacements ****************************
	  if (rec_struct[101] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 1;
            }
	  if (rec_struct[104] > 0)
            {
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  const char **nchoiseVals = new const char *[ii_choiseVals];
	  ii_choiseVals = 0;

	  // Displacements ****************************
	  if (rec_struct[101] > 0)
            {
	      nchoiseVals[ii_choiseVals] = "Displ.";
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  // Reaction forces **************************
	  if (rec_struct[104] > 0)
            {
	      nchoiseVals[ii_choiseVals] = "Reac.Force";
	      ii_choiseVals = ii_choiseVals + 1;
            }

	  p_nodalres->setValue(ii_choiseVals, nchoiseVals, 0);

	  // ***********************************************************************
	  // Look through elements to set up connection list ***********************
	  //
	  // Element types as enum see covise/src/kernel/do/coDoUnstructuredGrid.h
	  // TYPE_HEXAGON = 7,
	  // TYPE_HEXAEDER = 7,
	  // TYPE_PRISM = 6,
	  // TYPE_PYRAMID = 5,
	  // TYPE_TETRAHEDER = 4,
	  // TYPE_QUAD = 3,
	  // TYPE_TRIANGLE = 2,
	  // TYPE_BAR = 1,
	  // TYPE_NONE = 0,
	  // TYPE_POINT = 10

	  jobhead.no_conn = 0;
	  jobhead.no_sup_elems = 0;
	  jj = 0;

	  while (jj < data_length)
            {

	      rec_length = fil_array[jj];
	      rec_type = fil_array[jj + 1];

	      if (rec_type == 1900)
                {

		  switch (fil_array[jj + 3])
                    {

                    case 2314885531223470915: // "C3D8    "
		      jobhead.no_conn = jobhead.no_conn + 8;
		      break;

                    case 2314885530821735251: // "S3R     "
		      jobhead.no_conn = jobhead.no_conn + 3;
		      break;

                    case 2314885531156362051: // "C3D4    "
		      jobhead.no_conn = jobhead.no_conn + 4;
		      break;

                    case 2314885531189916483: // "C3D6    "
		      jobhead.no_conn = jobhead.no_conn + 6;
		      break;

                    case 2314885530819572546: // "B31     "
		      //jobhead.no_conn  = jobhead.no_conn  + 2;
		      jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
		      break;

                    case 2324217284263039059: // "SPRINGA "
		      //jobhead.no_conn  = jobhead.no_conn  + 2;
		      jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
		      break;

                    case 2314885531122807636: // "T3D2    "
		      //jobhead.no_conn  = jobhead.no_conn  + 2;
		      jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
		      break;

		    case 2325039727751676740: // DCOUP3D
		      //jobhead.no_conn  = jobhead.no_conn  + 2;
		      jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
		      break;

		    case 2314885530818458707: // S4
		      jobhead.no_conn = jobhead.no_conn + 4;
		      break;

                    default:

		      char temp[9];
		      temp[8] = 0;
		      *(int64_t *)temp = fil_array[jj + 3];
		      printf("%8s - %ld\n", temp, (long int)fil_array[jj + 3]);
		      printf("Module execution aborted\n");
		      sendError("While counting connections : Unknown element type '%s' '%ld'", 
				temp, (long int)fil_array[jj + 3]);
		      return;
		      break;
                    };

		  jobhead.no_sup_elems = jobhead.no_sup_elems + 1;
                };

	      jj = jj + rec_length;
            }
	  delete[] choiseVals;

	  if ((jobhead.no_elems - jobhead.no_sup_elems) > 0)
            {
	      sendWarning("Found %d unsupported elements", jobhead.no_elems - jobhead.no_sup_elems);
            }
	  
	  sendInfo("Finished. Please look at std-out for details about the .fil-file structure.");

	}
      
    }
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadABAQUSfil::compute(const char *port)
{
  (void)port;

  // We have to specify element types
  int hasTypes = 1;

  // Cast pointers for fil_array ***
  double *tmp_d;
  int *tmp_i;
  int ii_sets;
  int rec_length, rec_type, trec_len, tloc;
  int ii, jj, kk, nn;

  float *nxdataList;
  float *nydataList;
  float *nzdataList;
  float *dataList;
  float *tdataList;

  tmp_d = (double *)fil_array;
  tmp_i = (int *)fil_array;

  computeRunning = true;

  //===========================================================================

  // Declare grid Pointers ****************************************************
  int *outElemList, *outConnList, *outTypeList;
  float *outXCoord, *outYCoord, *outZCoord;

  // allocate new Unstructured grid *******************************************
  coDoUnstructuredGrid *outGrid = new coDoUnstructuredGrid(p_gridOutPort->getObjName(),
							   jobhead.no_sup_elems,
							   jobhead.no_conn,
							   jobhead.no_nodes,
							   hasTypes);
  // if object was not properly allocated *************************************
  if (!outGrid->objectOk())
    {
      sendError("Failed to create object '%s' for port '%s'",
		p_gridOutPort->getObjName(), p_gridOutPort->getName());

      return FAIL;
    }

  outGrid->getAddresses(&outElemList, &outConnList,
			&outXCoord, &outYCoord, &outZCoord);

  outGrid->getTypeList(&outTypeList);

  // ************************************************************************
  // Allocte elem_numbers array per set *************************************
  // Note that in case of a node set elem_numbers will hold the corresponding
  // external node numbers since set elemnts in this case refers to nodes
  for (vector<tSets>::iterator it = vsets.begin(); it != vsets.end(); ++it) {
    (*it).elem_numbers = (int*)malloc((*it).no_elems * sizeof(int));
  }

  // Get set numbers ********************************************************
  jj = 0;
  ii_sets = 0;

  while (jj < data_length)
    {

      rec_length  = fil_array[jj];
      rec_type    = fil_array[jj+1];

      switch ( rec_type ) { 
	 
      case 1931 :
	// Found Node Set *************************************************
	    
	// Get External Node Numbers in set *******************************
	tloc = rec_length-3;
	for(ii=0; ii<tloc; ++ii) {
	  vsets[ii_sets].elem_numbers[ii] = fil_array[jj+3+ii];
	}
	    
	// Check for node set continuations ************************
	kk = jj + rec_length;
	    
	while ( fil_array[kk+1] == 1932 ) {
	  trec_len  = fil_array[kk];
	  rec_type  = fil_array[kk+1];
	      
	  tloc=tloc+1;
	  for (ii=0; ii <= trec_len-3; ++ii) {
	    vsets[ii_sets].elem_numbers[tloc+ii] =  fil_array[kk+2+ii];
	  }
	  tloc = tloc + trec_len-3;
	  kk   = kk   + trec_len;
	}
	    
	ii_sets = ii_sets + 1;
	break;
	    
      case 1933 :
	// Found Element Set **********************************************
	    
	// Get External Element Numbers in set ****************************
	tloc = rec_length-3;
	for(ii=0; ii<tloc; ++ii) {
	  vsets[ii_sets].elem_numbers[ii] = fil_array[jj+3+ii];
	}
	    
	// Check for node set continuations ************************
	kk = jj + rec_length;

	while ( fil_array[kk+1] == 1934 ) {
	  trec_len  = fil_array[kk];
	  rec_type  = fil_array[kk+1];
	      
	  for (ii=0; ii <= trec_len-3; ++ii) {
	    vsets[ii_sets].elem_numbers[tloc+ii] =  fil_array[kk+2+ii];
	  }
	  tloc = tloc + trec_len-2;
	  kk   = kk   + trec_len;
	}
	    
	ii_sets = ii_sets + 1;
	break;
      }
	
      jj = jj + rec_length;
	
    }

  // Log - Sets and their names with cross references ***************
  // printf("===========================================================");
  // printf("===========================================================\n");
  // printf("%9s %10s %10s %9s %9s %9s %9s %s\n",
  // 	 "Count","Set Type","C8-Cref","Int-Cref","# Elems",
  // 	 "1st S-El","last S-El","External Set-Name");
  // printf("-----------------------------------------------------------");
  // printf("-----------------------------------------------------------\n");
  // for (vector<tSets>::iterator it = vsets.begin(); it != vsets.end(); ++it) {
  //   printf("%9d %10s %10s %9d %9d %9d %9d %s\n",
  // 	   it-vsets.begin(),
  // 	   (*it).type.c_str(),
  // 	   (*it).cname.c_str(),
  // 	   (*it).cref,
  // 	   (*it).no_elems,
  // 	   (*it).elem_numbers[0],(*it).elem_numbers[(*it).no_elems-1],
  // 	   (*it).name.c_str());
  // }

  // ************************************************************************
  // Load coordinates, elements and their external numbers ******************

  int ii_nodes = 0;
  int ii_Elems = 0;
  int ii_Conn = 0;

  jj = 0;

  int *ext_nn;
  ext_nn = (int *)malloc(jobhead.no_nodes * sizeof(int));
  int max_ext_nn = -1;

  while ((jj < data_length) && (fil_array[jj + 1] != 2000))
    {

      // Parse node records **********************
      if (fil_array[jj + 1] == 1901)
        {

	  ext_nn[ii_nodes] = (int)fil_array[jj + 2] - 1;
	  if (max_ext_nn < (int)fil_array[jj + 2] - 1)
	    max_ext_nn = (int)fil_array[jj + 2] - 1;

	  outXCoord[ii_nodes] = float(tmp_d[jj + 3]);
	  outYCoord[ii_nodes] = float(tmp_d[jj + 4]);
	  outZCoord[ii_nodes] = float(tmp_d[jj + 5]);

	  ii_nodes = ii_nodes + 1;
        };

      jj = jj + fil_array[jj];
    }

  // set up cross reference array for external node numbers *******************
  // WARNING !! It is assumed that the node numbers are in accending order
  int *cref_nodes;
  cref_nodes = (int *)malloc(max_ext_nn * sizeof(int));
  for (ii = 0; ii < max_ext_nn; ++ii)
    {
      cref_nodes[ii] = -1;
    }
  for (ii = 0; ii < jobhead.no_nodes; ++ii)
    {
      cref_nodes[ext_nn[ii]] = ii;
    }

  jj = 0;

  int *ext_en;
  ext_en = (int *)malloc(jobhead.no_sup_elems * sizeof(int));
  int max_ext_en = -1;

  while ((jj < data_length) && (fil_array[jj + 1] != 2000))
    {
      // Parse element records **********************
      if (fil_array[jj + 1] == 1900)
        {

	  ext_en[ii_Elems] = (int)fil_array[jj + 2];

	  outElemList[ii_Elems] = ii_Conn;

	  switch (fil_array[jj + 3])
            {

            case 2314885531223470915: // "C3D8    "
	      for (ii = 0; ii < 8; ++ii)
                {
		  outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
	      ii_Conn = ii_Conn + 8;
	      outTypeList[ii_Elems] = 7;
	      if (max_ext_en < (int)fil_array[jj + 2])
		max_ext_en = (int)fil_array[jj + 2];
	      break;

            case 2314885530821735251: // "S3R     "
	      for (ii = 0; ii < 3; ++ii)
                {
		  outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
	      ii_Conn = ii_Conn + 3;
	      outTypeList[ii_Elems] = 2;
	      if (max_ext_en < (int)fil_array[jj + 2])
		max_ext_en = (int)fil_array[jj + 2];
	      break;

            case 2314885531156362051: // "C3D4    "
	      for (ii = 0; ii < 4; ++ii)
                {
		  outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
	      ii_Conn = ii_Conn + 4;
	      outTypeList[ii_Elems] = 4;
	      if (max_ext_en < (int)fil_array[jj + 2])
		max_ext_en = (int)fil_array[jj + 2];
	      break;

            case 2314885531189916483: // "C3D6    "
	      for (ii = 0; ii < 6; ++ii)
                {
		  outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
	      ii_Conn = ii_Conn + 6;
	      outTypeList[ii_Elems] = 6;
	      if (max_ext_en < (int)fil_array[jj + 2])
		max_ext_en = (int)fil_array[jj + 2];
	      break;

            case 2314885530819572546: // "B31     "
	      // for(ii=0; ii<2; ++ii) {
	      //   outConnList[ii_Conn+ii]=fil_array[jj+4+ii]-(int64_t)1;
	      // }
	      // ii_Conn  = ii_Conn  + 2;
	      // outTypeList[ii_Elems] = 1;
	      ii_Elems = ii_Elems - 1;
	      break;

            case 2324217284263039059: // "SPRINGA "
	      // for(ii=0; ii<2; ++ii) {
	      //   outConnList[ii_Conn+ii]=fil_array[jj+4+ii]-(int64_t)1;
	      // }
	      // ii_Conn  = ii_Conn  + 2;
	      // outTypeList[ii_Elems] = 1;
	      ii_Elems = ii_Elems - 1;
	      break;

            case 2314885531122807636: // "T3D2    "
	      ii_Elems = ii_Elems - 1;
	      break;

	    case 2325039727751676740: // DCOUP3D
	      ii_Elems = ii_Elems - 1;
	      break;

	    case 2314885530818458707: // S4
	      for (ii = 0; ii < 4; ++ii)
                {
		  outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
	      ii_Conn = ii_Conn + 4;
	      outTypeList[ii_Elems] = 3;
	      if (max_ext_en < (int)fil_array[jj + 2])
		max_ext_en = (int)fil_array[jj + 2];
	      break;

            default:

	      sendError("While reading Elements : Unknown element type '%ld'", (long int)fil_array[jj + 3]);

	      char temp[9];
	      temp[8] = 0;
	      *(int64_t *)temp = fil_array[jj + 3];
	      printf("%8s - %ld\n", temp, (long int)fil_array[jj + 3]);

	      return FAIL;

	      break;
            };

	  ii_Elems = ii_Elems + 1;
        };

      jj = jj + fil_array[jj];
    }

  // set up cross reference array for external elem numbers *******************
  // WARNING !! It is assumed that the elem numbers are in accending order
  int *cref_elems;
  cref_elems = (int *)malloc((max_ext_en + 1) * sizeof(int));
  for (ii = 0; ii < max_ext_en + 1; ++ii)
    {
      cref_elems[ii] = -1;
    }
  for (ii = 0; ii < jobhead.no_sup_elems; ++ii)
    {
      cref_elems[ext_en[ii]] = ii;
    }

  // **************************************************************************
  // FINALLY WE HAVE THE FULL MESH LISTS LOADED :
  // ext_nn      : External node numbers
  // outXCoord   : Physical x-coordinate list
  // outYCoord   : Physical x-coordinate list
  // outZCoord   : Physical x-coordinate list
  // cref_nodes  : Cross reference from external node numbers to implicit
  //               node numbering strating from 0
  // ext_en      : External Element numbers
  // outTypeList : Element types (According to covise element type definition)
  // outConnList : Topology of elements (Node numbers belonging to each element)
  // outElemList : Topology pointer 
  // cref_elems  : Cross reference from external element numbers to implicit
  //               element numbering strating from 0
  // vsets       : Nodeset and Elementset parameters and external numbers of
  //             : contained set elements (in elem_numbers).
  // **************************************************************************

  // **************************************************************************
  // Load element data ********************************************************

  coDoSet *outERes = new coDoSet(p_eresOutPort->getObjName(),0);
  coDoSet *outTRes = new coDoSet(p_tresOutPort->getObjName(),0);
  coDoSet *outNRes = new coDoSet(p_nresOutPort->getObjName(),0);

  if (vsteps.size() > 1) {

    char ts[100];
    sprintf(ts, "1 %d", vsteps.size());
    outERes->addAttribute("TIMESTEP", ts);
    outTRes->addAttribute("TIMESTEP", ts);
    outNRes->addAttribute("TIMESTEP", ts);

  }
  
  sendInfo("Requested element scalar result: %s",p_elemres->getActLabel());
  sendInfo("Requested element tensor result: %s",p_telemres->getActLabel());
  sendInfo("Requested nodal result: %s",p_nodalres->getActLabel());
    
  const char *activeLabel  = p_elemres->getActLabel();

  int sel_tres;
  const char *activetLabel = p_telemres->getActLabel();
  if (strcmp(activetLabel, "Stress") == 0) {
    sel_tres = 11;
  } else if (strcmp(activetLabel, "Strain") == 0) {
    sel_tres = 21;
  }
  
  const char *activeNLabel = p_nodalres->getActLabel();

  int sel_nres;  
  if (strcmp(activeNLabel, "Displ.") == 0) {
    sel_nres = 101;
  } else if (strcmp(activeNLabel, "Reac.Force") == 0) {
    sel_nres = 104;
  }

  //=================================================================
  //=================================================================
  // Parse selected set numbers from paramerter p_selectedsets
  vector<string> st_sn;
  vector<int>    set_nums;
  char * pch;
  
  // Seperator for set numbers and number ranges is , ***************
  pch = strtok ((char*)p_selectedSets->getValue()," ,");
  while (pch != NULL)  {
    st_sn.push_back(pch);
    pch = strtok (NULL, " ,");
  }

  // Convert char numbers and ranges to integer *********************
  for (vector<string>::iterator it = st_sn.begin(); it != st_sn.end(); ++it) {
    
    // if we have a range (two numbers seperated by - without blanks)
    ii = (*it).find_first_of("-",0);
    if ( ii  != string::npos)  {
      
      istringstream convert((*it).substr(0,ii).c_str());
      
      // convert start of range to integer **************************
      if ( !(convert >> jj ) ) {
  	printf("Set no %d without cref. Name = %s\n",ii_sets,(*it).c_str());
      } else {
	
  	istringstream convert((*it).substr(ii+1).c_str());
	
  	// convert end of range to integer **************************
  	if ( !(convert >> kk ) ) {
  	  printf("Set no %d without cref. Name = %s\n",ii_sets,(*it).c_str());
  	} else {
	  
  	  // Push range to vector *************************
  	  for (ii = jj; ii <= kk; ++ii) {
  	    set_nums.push_back(ii);
  	  }
  	}
      }
      
    // If we have a single number ***********************************
    } else {
      
      istringstream convert((*it).c_str());
      
      // convert the number to integer ********************
      if ( !(convert >> ii ) ) {
  	printf("Set no %d without cref. Name = %s\n",ii_sets,(*it).c_str());
      } else {
  	// Push the number to vector **********************
  	set_nums.push_back(ii);
      }
    }
  }

  // Allocate pointer array to nodes used in a set ****************
  int* l_nn_nd = (int*)malloc(jobhead.no_nodes * sizeof(int));
  for (ii=0; ii < jobhead.no_nodes; ++ii) {
    l_nn_nd[ii] = 0;
  }

  printf("===========================================================");
  printf("===========================================================\n");

  int SetNo;

  coDoSet *outSet     = new coDoSet(p_SetgridOutPort->getObjName(),0);
  coDoSet *outResSet  = new coDoSet(p_SetgridResPort->getObjName(),0);
  coDoSet *outtResSet = new coDoSet(p_SetgridTResPort->getObjName(),0);
  coDoSet *outnResSet = new coDoSet(p_SetgridnResPort->getObjName(),0);
  
  printf("For each Step and Increment\n");

  // **************************************************************************
  // For each Step and Increment **********************************************
  // **************************************************************************
  for (vector<tStephead>::iterator it = vsteps.begin(); it != vsteps.end(); ++it) {

    //Allocate Output Data Objects ********************************************
    string obj_name_eres = "ERes_Step_";
    obj_name_eres += std::to_string((*it).Step_no);
    obj_name_eres += "_Inc_";
    obj_name_eres += std::to_string((*it).Inc_no);

    string obj_name_tres = "TRes_Step_";
    obj_name_tres += std::to_string((*it).Step_no);
    obj_name_tres += "_Inc_";
    obj_name_tres += std::to_string((*it).Inc_no);

    printf("%d %s\n",it,obj_name_tres.c_str());

    coDoFloat  *data  = new coDoFloat (obj_name_eres.c_str(), jobhead.no_sup_elems);
    coDoTensor *tdata = new coDoTensor(obj_name_tres.c_str(), jobhead.no_sup_elems, coDoTensor::S3D);

    // if objects were not properly allocated *********************************
    if (!data->objectOk())
      {
	sendError("Failed to create object '%s' for port '%s'",
		  p_eresOutPort->getObjName(), p_eresOutPort->getName());
	return FAIL;
      }
    
    if (!tdata->objectOk())
      {
	sendError("Failed to create object '%s' for port '%s'",
		  p_tresOutPort->getObjName(), p_tresOutPort->getName());
	return FAIL;
      }
    
    data  -> getAddress(&dataList);
    tdata -> getAddress(&tdataList);

    for (ii = 0; ii < jobhead.no_sup_elems; ii++)
      {
	dataList[ii]  = 0.;
	tdataList[ii] = 0.;
      }
    
    int ii_dat = 0;

    jj = (*it).start;

    while (jj < (*it).end) {
      
      // Element results ****************************************************
      if ((fil_array[jj + 1] == 1) && (cref_elems[(int)fil_array[jj + 2]] != -1)) {
	
	// Check for element averaged result values *****************************
	if (tmp_i[(jj + 3) * 2] != 0) {
	  sendWarning("This module supports only element averaged result values");
	  sendWarning("Element results not loaded *****************************");
	  return SUCCESS;
	  ;
	}

	ii_dat = (int)fil_array[jj + 2];
	
	jj = jj + fil_array[jj];
	
	// stress results ************************************************
	if (fil_array[jj + 1] == 11) {

	  // Load full element stress tensor ********************
	  if (sel_tres == 11) {
	    tdataList[cref_elems[ii_dat]*6  ] = float(tmp_d[jj + 2]);
	    tdataList[cref_elems[ii_dat]*6+1] = float(tmp_d[jj + 3]);
	    tdataList[cref_elems[ii_dat]*6+2] = float(tmp_d[jj + 4]);
	    tdataList[cref_elems[ii_dat]*6+3] = float(tmp_d[jj + 5]);
	    tdataList[cref_elems[ii_dat]*6+4] = float(tmp_d[jj + 7]);
	    tdataList[cref_elems[ii_dat]*6+5] = float(tmp_d[jj + 6]);
	  }
	  
	  // Load selected scalar stress tensor component *******
	  if (strcmp(activeLabel, sigma[0]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 2]);
	  }
	  if (strcmp(activeLabel, sigma[1]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 3]);
	  }
	  if (strcmp(activeLabel, sigma[2]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 4]);
	  }
	  if (strcmp(activeLabel, sigma[3]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 5]);
	  }
	  if (strcmp(activeLabel, sigma[4]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 6]);
	  }
	  if (strcmp(activeLabel, sigma[5]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 7]);
	  }
	  
	  // Von Mises **********************************************
	  if (strcmp(activeLabel, equivalence[0]) == 0) {
	    dataList[cref_elems[ii_dat]] = 
	      sqrt(float(tmp_d[jj + 2]) * float(tmp_d[jj + 2]) + 
		   float(tmp_d[jj + 3]) * float(tmp_d[jj + 3]) + 
		   float(tmp_d[jj + 4]) * float(tmp_d[jj + 4]) + 
		   float(tmp_d[jj + 2]) * float(tmp_d[jj + 3]) + 
		   float(tmp_d[jj + 2]) * float(tmp_d[jj + 4]) + 
		   float(tmp_d[jj + 3]) * float(tmp_d[jj + 4]) + 
		   3. * (float(tmp_d[jj + 5]) * float(tmp_d[jj + 5]) + 
			 float(tmp_d[jj + 6]) * float(tmp_d[jj + 6]) + 
			 float(tmp_d[jj + 7]) * float(tmp_d[jj + 7])));
	  };
	};

	// Strain Tensor ******************************************************
	if (fil_array[jj + 1] == 21) {
	  
	  // Load full element strain tensor ********************
	  if (sel_tres == 21) {
	    tdataList[cref_elems[ii_dat]*6  ] = float(tmp_d[jj + 2]);
	    tdataList[cref_elems[ii_dat]*6+1] = float(tmp_d[jj + 3]);
	    tdataList[cref_elems[ii_dat]*6+2] = float(tmp_d[jj + 4]);
	    tdataList[cref_elems[ii_dat]*6+3] = float(tmp_d[jj + 5]);
	    tdataList[cref_elems[ii_dat]*6+4] = float(tmp_d[jj + 7]);
	    tdataList[cref_elems[ii_dat]*6+5] = float(tmp_d[jj + 6]);
	  }
	  
	  // Load selected scalar strain tensor component *******
	  if (strcmp(activeLabel, epsilon[0]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 2]);
	  }
	  if (strcmp(activeLabel, epsilon[1]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 3]);
	  }
	  if (strcmp(activeLabel, epsilon[2]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 4]);
	  }
	  if (strcmp(activeLabel, epsilon[3]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 5]);
	  }
	  if (strcmp(activeLabel, epsilon[4]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 6]);
	  }
	  if (strcmp(activeLabel, epsilon[5]) == 0) {
	    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 7]);
	  }
	};
      };
      
      jj = jj + fil_array[jj];
    }
    
    // ************************************************************************
    // Load nodal data ********************************************************

    //Allocate Output Data Objects ********************************************
    string obj_name_nres = "NRes_Step_";
    obj_name_nres += std::to_string((*it).Step_no);
    obj_name_nres += "_Inc_";
    obj_name_nres += std::to_string((*it).Inc_no);

    coDoVec3 *ndata = new coDoVec3(obj_name_nres.c_str(), jobhead.no_nodes);

    // if objects were not properly allocated *********************************
    if (!ndata->objectOk())
      {
	sendError("Failed to create object '%s' for port '%s'",
		  obj_name_nres.c_str(), p_nresOutPort->getName());
	return FAIL;
      }

    ndata->getAddresses(&nxdataList,&nydataList,&nzdataList);

    for (ii = 0; ii < jobhead.no_nodes; ii++) {
      nxdataList[ii] = 0.;
      nydataList[ii] = 0.;
      nzdataList[ii] = 0.;
    }

    ii_dat = 0;
    jj     = (*it).start;
    
    while (jj < (*it).end) {
      
      // Displacements ********************************************************
      if ((fil_array[jj + 1] == 101) && (sel_nres == 101)) {
	
	ii_dat = (int)fil_array[jj + 2];
	nxdataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 3]);
	nydataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 4]);
	nzdataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 5]);
	
      }
      
      // Reaction Forces ******************************************************
      if ((fil_array[jj + 1] == 104) && (sel_nres == 104)) {
	
	ii_dat = (int)fil_array[jj + 2];
	nxdataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 3]);
	nydataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 4]);
	nzdataList[cref_nodes[ii_dat-1]] = float(tmp_d[jj + 5]);
	
      }
      
      jj = jj + fil_array[jj];
    }

    outERes->addElement(data);
    outTRes->addElement(tdata);
    outNRes->addElement(ndata);
    
  }

  //=========================================================================
  //=========================================================================
  // Get Element and node Sets
  
  for (vector<int>::iterator it = set_nums.begin(); it != set_nums.end(); ++it) {

    SetNo = (*it);

    if ( ( SetNo < 0 ) || (SetNo >= vsets.size() ) ) {
      printf("Selected set no %d not in set range 0 - %d \n",SetNo, vsets.size());
      continue;
    } else if ( vsets[SetNo].type.compare("Elems") != 0) {
      printf("Selected set no %d is not an element set \n",SetNo);
      continue;
    } else {
      printf("Adding set no %d as element %d to GridSet\n",SetNo,it-set_nums.begin());
    }
    
    // Count set local connections and mark local nodes in l_nn_nd ************
    vsets[SetNo].no_conn  = 0;
    
    for (kk=0; kk < vsets[SetNo].no_elems; ++kk) {

      switch ( outTypeList[cref_elems[vsets[SetNo].elem_numbers[kk]]] ) { 
	 
      case 2 : // Trias
  	vsets[SetNo].no_conn = vsets[SetNo].no_conn + 3;
  	nn = 3;
  	break;

      case 3 : // Quads
  	vsets[SetNo].no_conn = vsets[SetNo].no_conn + 4;
  	nn = 4;
  	break;
		
      case 4 : // Tetras    
  	vsets[SetNo].no_conn = vsets[SetNo].no_conn + 4;
  	nn = 4;
  	break;
	
      case 6 : // Pentas / Wedges
  	vsets[SetNo].no_conn = vsets[SetNo].no_conn + 6;
  	nn = 6;
  	break;

      case 7 : // Hexas
  	vsets[SetNo].no_conn = vsets[SetNo].no_conn + 8;
  	nn = 8;
  	break;
		
      default:
	
  	sendError("While counting set connections : Unknown element type '%d'", 
  		  outTypeList[cref_elems[vsets[SetNo].elem_numbers[kk]]]);
  	return FAIL;
  	break;
      };
           
      for (jj = outElemList[cref_elems[vsets[SetNo].elem_numbers[kk]]]    ; 
  	   jj < outElemList[cref_elems[vsets[SetNo].elem_numbers[kk]]]+nn ; ++jj) {
  	l_nn_nd[outConnList[jj]] = 1;
      }
      
    }
    
    // Count local nodes ******************************************************
    vsets[SetNo].no_nodes = 0;
    for (kk=0; kk < jobhead.no_nodes; ++kk) {
      if (l_nn_nd[kk]) vsets[SetNo].no_nodes = vsets[SetNo].no_nodes + 1;
    }

    //printf("Countet %d Connections for SetNo %d\n",vsets[SetNo].no_conn,SetNo);
    //printf("Countet %d Nodes       for SetNo %d\n",vsets[SetNo].no_nodes,SetNo);
    
    // Construc Set-Grid ******************************************************
    // Declare grid Pointers **************************************************
    int *setElemList, *setConnList, *setTypeList;
    float *setXCoord, *setYCoord, *setZCoord;
    
    string obj_name_grid = "Grid_Set_No_";
    obj_name_grid += std::to_string(it-set_nums.begin());

    // allocate new Unstructured grid *****************************************
    coDoUnstructuredGrid *setGrid = 
      new coDoUnstructuredGrid(obj_name_grid.c_str(), //p_SetgridOutPort->getObjName(),
  			       vsets[SetNo].no_elems,
  			       vsets[SetNo].no_conn,
  			       vsets[SetNo].no_nodes,
  			       hasTypes);
    
    // if object was not properly allocated ***********************************
    if (!outGrid->objectOk()) {
      sendError("Failed to create object '%s' for port '%s'",
  		p_SetgridOutPort->getObjName(), p_SetgridOutPort->getName());
      
      return FAIL;
    }
    
    setGrid->getAddresses(&setElemList, &setConnList,
  			  &setXCoord, &setYCoord, &setZCoord);
    
    setGrid->getTypeList(&setTypeList);
    
    // Copy coordinates from global to local grid pointers ********************
    jj = 0;
    for (kk=0; kk < jobhead.no_nodes; ++kk) {
      
      if (l_nn_nd[kk]) {
  	setXCoord[jj] = outXCoord[kk];
  	setYCoord[jj] = outYCoord[kk];
  	setZCoord[jj] = outZCoord[kk];

    	l_nn_nd[kk] = jj;
  	jj = jj + 1;
	
      } else {
	l_nn_nd[kk] = -1;
      }
    }
    
    // Copy topology from global to local grid pointers ***********************
    ii = 1;
    setElemList[0] = 0;
    
    for (kk=0; kk < vsets[SetNo].no_elems; ++kk) {
      
      switch ( outTypeList[cref_elems[vsets[SetNo].elem_numbers[kk]]] ) { 
	
      case 7 : // Hexas
  	nn = 8;
  	break;
	
      case 2 : // Trias
  	nn = 3;
  	break;

      case 3 : // Quads
  	nn = 4;
  	break;
		
      case 4 : // Tetras
  	nn = 4;
  	break;
	
      case 6 : // Pentas / Wedges
  	nn = 6;
  	break;
	
      default:
	
  	sendError("While copying Set-Elements : Unknown element type '%d'",
  		  outTypeList[cref_elems[vsets[SetNo].elem_numbers[kk]]]);

  	return FAIL;
  	break;
      };
      
      setTypeList[kk] = outTypeList[cref_elems[vsets[SetNo].elem_numbers[kk]]];
      setElemList[ii] = setElemList[ii-1];
      for (jj = outElemList[cref_elems[vsets[SetNo].elem_numbers[kk]]]    ; 
  	   jj < outElemList[cref_elems[vsets[SetNo].elem_numbers[kk]]]+nn ; ++jj) {
  	setConnList[setElemList[ii]] = l_nn_nd[outConnList[jj]];
  	setElemList[ii] = setElemList[ii] + 1;
	
      }
      ii = ii + 1;
    }

    // Add Set-Mesh to GridSet Output *********************
    outSet->addElement(setGrid);

    // ************************************************************************
    // Create set results *****************************************************
    float *setedataList;  // Scalar Element results
    float *settdataList;  // Tensor Element results
    float *setnxdataList; // Nodal result x-comp.
    float *setnydataList; // Nodal result y-comp.
    float *setnzdataList; // Nodal result z-comp.

    // Set for scalar element results per element set ***************
    string obj_name_eres = "ERes_Set_No_";
    obj_name_eres += std::to_string(it-set_nums.begin());
    coDoSet *outStepEResSet  = new coDoSet(obj_name_eres.c_str(),0);

    if (!outStepEResSet->objectOk()) {
      sendError("Failed to create object '%s'", obj_name_eres.c_str());
      return FAIL;
    }

    // Set for tensor element results per element set ***************
    string obj_name_tres = "TRes_Set_No_";
    obj_name_tres += std::to_string(it-set_nums.begin());
    coDoSet *outStepTResSet  = new coDoSet(obj_name_tres.c_str(),0);

    if (!outStepTResSet->objectOk()) {
      sendError("Failed to create object '%s'", obj_name_tres.c_str());
      return FAIL;
    }

    // Set for nodal results per element set ************************
    string obj_name_nres = "NRes_Set_No_";
    obj_name_nres += std::to_string(it-set_nums.begin());
    coDoSet *outStepNResSet  = new coDoSet(obj_name_nres.c_str(),0);

    if (!outStepNResSet->objectOk()) {
      sendError("Failed to create object '%s'", obj_name_nres.c_str());
      return FAIL;
    } 
   
    // Add TIMESTEP attribute to sets *******************************
    if (vsteps.size() > 1) {

      char ts[100];
      printf("vsteps.size() %d\n",vsteps.size());
      sprintf(ts, "1 %d", vsteps.size());
      outStepTResSet->addAttribute("TIMESTEP", ts);
      outStepEResSet->addAttribute("TIMESTEP", ts);
      outStepNResSet->addAttribute("TIMESTEP", ts);
     
    }

    int step = 0;
    
    // For each Step and increment ********************************************
    for (vector<tStephead>::iterator sit = vsteps.begin(); sit != vsteps.end(); ++sit) {

      // Create unique object names *********************************
      string obj_name_steperes = "ERes_Set_No_";
      obj_name_steperes += std::to_string(it-set_nums.begin());
      obj_name_steperes += "_Step_";
      obj_name_steperes += std::to_string((*sit).Step_no);
      obj_name_steperes += "_Inc_";
      obj_name_steperes += std::to_string((*sit).Inc_no);

      string obj_name_steptres = "TRes_Set_No_";
      obj_name_steptres += std::to_string(it-set_nums.begin());
      obj_name_steptres += "_Step_";
      obj_name_steptres += std::to_string((*sit).Step_no);
      obj_name_steptres += "_Inc_";
      obj_name_steptres += std::to_string((*sit).Inc_no);
    
      string obj_name_stepnres = "NRes_Set_No_";
      obj_name_stepnres += std::to_string(it-set_nums.begin());
      obj_name_stepnres += "_Step_";
      obj_name_stepnres += std::to_string((*sit).Step_no);
      obj_name_stepnres += "_Inc_";
      obj_name_stepnres += std::to_string((*sit).Inc_no);

      // Object for increment nodal results *************************
      coDoVec3 *setndata = new coDoVec3(obj_name_stepnres.c_str(),
					vsets[SetNo].no_nodes);

      if (!setndata->objectOk()) {
	sendError("Failed to create object '%s'", obj_name_nres.c_str());
	return FAIL;
      }
    
      setndata->getAddresses(&setnxdataList,&setnydataList,&setnzdataList);

      // Pointer to global mesh nodal results for increment *********
      coDoVec3 *setStepNdata;

      // get Pointers to increment results of global Mesh ***********
      setStepNdata = (coDoVec3*)outNRes->getElement(step);
      setStepNdata->getAddresses(&nxdataList,&nydataList,&nzdataList);

      // Copy nodal results from global to local result pointers ****
      jj = 0;
      for (kk=0; kk < jobhead.no_nodes; ++kk) {
	
	if (l_nn_nd[kk] > -1) {
	  
	  setnxdataList[jj] = nxdataList[kk];
	  setnydataList[jj] = nydataList[kk];
	  setnzdataList[jj] = nzdataList[kk];
	  
	  jj = jj + 1;
	}
      }

      // Add the set results element **************
      outStepNResSet->addElement(setndata);

      //*************************************************************
      // Object for increment element scalar results ****************
      coDoFloat *setdata = new coDoFloat( obj_name_steperes.c_str(),
					 vsets[SetNo].no_elems);

      if (!setdata->objectOk()) {
  	sendError("Failed to create object '%s'", obj_name_steperes.c_str());
  	return FAIL;
      }

      setdata->getAddress(&setedataList);

      //*************************************************************
      // Object for increment element tensor results ****************
      coDoTensor *tsetdata = new coDoTensor(obj_name_steptres.c_str(),
					    vsets[SetNo].no_elems, coDoTensor::S3D);

      if (!tsetdata->objectOk()) {
  	sendError("Failed to create object '%s'", obj_name_steptres.c_str());
  	return FAIL;
      }

      tsetdata->getAddress(&settdataList);
      printf("%s\n",obj_name_steptres.c_str());
      // Pointer to global mesh nodal results for increment *********
      coDoFloat *setStepEdata;

      // get Pointers to increment results of global Mesh ***********
      setStepEdata = (coDoFloat*)outERes->getElement(step);
      setStepEdata->getAddress(&dataList);

      coDoTensor *setStepTdata;

      // get Pointers to increment results of global Mesh ***********
      setStepTdata = (coDoTensor*)outTRes->getElement(step);
      setStepTdata->getAddress(&tdataList);

      
      for (kk=0; kk < vsets[SetNo].no_elems; ++kk) {
	
	// Copy Scalar component **********************************************
	setedataList [kk    ]   = dataList [cref_elems[vsets[SetNo].elem_numbers[kk]]];

	// Copy tensor ********************************************************
	settdataList[kk*6  ] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6  ];
	settdataList[kk*6+1] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6+1];
	settdataList[kk*6+2] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6+2];
	settdataList[kk*6+3] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6+3];
	settdataList[kk*6+4] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6+4];
	settdataList[kk*6+5] = tdataList[cref_elems[vsets[SetNo].elem_numbers[kk]]*6+5];
      }

      outStepEResSet->addElement(setdata);
      outStepTResSet->addElement(tsetdata);
    

      step = step + 1;
    }

    outResSet->addElement(outStepEResSet);
    outtResSet->addElement(outStepTResSet);
    outnResSet->addElement(outStepNResSet);

  }

  printf("===========================================================");
  printf("===========================================================\n");

  // Set single grid objects to Out Ports ***************************
  if (outGrid) 
    p_gridOutPort->setCurrentObject(outGrid);
  if (outERes)
    p_eresOutPort->setCurrentObject(outERes);
  if (outTRes)
    p_tresOutPort->setCurrentObject(outTRes);
  if (outNRes)
    p_nresOutPort->setCurrentObject(outNRes);
  

  // Set set objects to Out Ports ***********************************
  if(outSet)
    p_eresOutPort->setCurrentObject(outSet);
  if(outResSet)
    p_SetgridResPort->setCurrentObject(outResSet);
  if(outtResSet)
    p_SetgridTResPort->setCurrentObject(outtResSet);
  if(outnResSet)
    p_SetgridnResPort->setCurrentObject(outnResSet);


  computeRunning = false;

  return SUCCESS;
}

MODULE_MAIN(IO, ReadABAQUSfil)
