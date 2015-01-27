/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FLE_lib.cpp" // #### including .cpp

void FLE_impl(UniSys *us,
              Unstructured *unst,
              int compVelo,
              int unsteady,
              const char *velocity_file,
              float start_time,
              //int crop_ucd, float origin_x, float origin_y, float origin_z, float voxel_size,
              //char *mode,
              int modeNr,
              int ln,
              int divT,
              float integration_time,
              float integration_length,
              int time_intervals,
              float sep_factor_min,
              int integ_steps_max,
              int forward,
              Unstructured *unst_out,
              int smoothing_range,
              int omit_boundary_cells,
              int grad_neigh_disabled,
              UniField *unif_traj)
{

// temporary data for map
#if 1 // ###### there is a BUG in Unstructured here  // ################ TODO!!!
    Unstructured *map = new Unstructured(unst_out, 3);
#else
    UCD_structure *ucd_map = ucdClone(ucd_out, 3, "ucd map");
    Unstructured *map = new Unstructured(ucd_map);
#endif

    // node is Disabled (for all operations) if outside of UCD
    bool *nodeDisabled = new bool[unst_out->nNodes];
    if (!nodeDisabled)
    {
        us->error("out of memory");
        exit(0); // ###
    }
    for (int i = 0; i < unst_out->nNodes; i++)
        nodeDisabled[i] = false;
    int nodesDisabled = 0;

    // node is finished if FSLE separation is reached
    bool *nodeFinished = NULL;
    int nodesFinished = 0;
    //if (strcmp(mode, MO_FSLE) == 0) {
    if (modeNr == 3)
    {
        nodeFinished = new bool[unst_out->nNodes];
        if (!nodeFinished)
        {
            us->error("out of memory");
            exit(0); // ###
        }
        for (int i = 0; i < unst_out->nNodes; i++)
            nodeFinished[i] = false;
    }

    // could save memory by only allocating this when necessary
    // ### could avoid it completely, it is only used for avoiding continuation
    //     of stopped trajectories
    int *trajVertCnt = new int[unst_out->nNodes];
    for (int n = 0; n < unst_out->nNodes; n++)
    {
        trajVertCnt[n] = 0;
    }

    int modeI = 0;
    //if (strcmp(mode, MO_FTLE) == 0) modeI = MODE_FTLE;
    //else if (strcmp(mode, MO_FLLE) == 0) modeI = MODE_FLLE;
    //else if (strcmp(mode, MO_FSLE) == 0) modeI = MODE_FSLE;
    //else if (strcmp(mode, MO_FMLE) == 0) modeI = MODE_FMLE;
    //else if (strcmp(mode, MO_FALE) == 0) modeI = MODE_FALE;
    if (modeNr == 1)
        modeI = MODE_FTLE;
    else if (modeNr == 2)
        modeI = MODE_FLLE;
    else if (modeNr == 3)
        modeI = MODE_FSLE;
    else if (modeNr == 4)
        modeI = MODE_FMLE;
    else if (modeNr == 5)
        modeI = MODE_FALE;

#if 0 // DELETEME
  // compute flow map (position of given node after time of advection)
  int wsteps = 0;
  computeFlowMap(unst,
                 //crop_ucd,
                 false,
                 modeI,
                 integration_time, integration_length, integ_steps_max,
                 forward, nodeDisabled, nodesDisabled, omit_boundary_cells,
                 unst_out,
                 false, // continue map
                 0.0, 0.0, wsteps,
                 NULL,
                 unsteady, velocity_file, start_time, true, true,
                 map, trajectories);
  
  
  // compute FTLE
  computeFTLE(map, nodeDisabled, unst_out);
#endif

    {
        clock_t t_start = clock();

        printf("computation started at time:\n");
        system("date");

        double map_seconds = 0.0;
        double FLE_seconds = 0.0;

        computeFLE(us,
                   unst,
                   //crop_ucd,
                   false,
                   modeI,
                   ln, divT,
                   integration_time, integration_length, sep_factor_min,
                   time_intervals,
                   integ_steps_max,
                   forward,
                   nodeDisabled, nodesDisabled,
                   grad_neigh_disabled,
                   nodeFinished, nodesFinished,
                   omit_boundary_cells,
                   NULL,
                   unsteady, velocity_file, start_time,
                   true,
                   true,
                   1, // minor transient verbosity
                   map,
                   smoothing_range,
                   //trajectories,
                   unif_traj,
                   //NULL, // trajVertCnt
                   trajVertCnt,
                   unst_out,
                   &map_seconds,
                   &FLE_seconds);

        clock_t t_end = clock();
        double seconds = (t_end - t_start) / ((double)CLOCKS_PER_SEC);

        printf("\nstatistics ------------------------------------------------------------------\n\n");

        printf("computation finished at time:\n");
        system("date");

        printf("total computation took %g seconds\n", seconds);
        printf("map computation took %g seconds\n", map_seconds);
        printf("FLE computation took %g seconds\n", FLE_seconds);
        printf("\n\n");
    }

    // output map
    // HACK TODO
    for (int n = 0; n < unst_out->nNodes; n++)
    {
        vec3 vec;
        map->getVector3(n, vec);
        unst_out->setVector3(n, OUTCOMP_MAP, vec);
    }

    delete[] nodeDisabled;
    if (nodeFinished)
        delete[] nodeFinished;
    if (trajVertCnt)
        delete[] trajVertCnt;
    delete map;
    //if (ucd_map) UCDstructure_free(ucd_map); // ########## TOCHECK
    //delete unst_out;
}
