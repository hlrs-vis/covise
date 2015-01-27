/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FLE_LIB_CPP
#define FLE_LIB_CPP

#define PATHLINES_SEQUENTIAL 1 // for sequential pathlines
#define USE_TRANSIENT_UNSTRUCTURED 1 // for using Unstructured's time support
#define USE_DATA_DICT 0 // if zero, "single-file" method is used

#define MODE_FTLE 0
#define MODE_FLLE 1
#define MODE_FSLE 2
#define MODE_FMLE 3
#define MODE_FALE 4

#define OUTCOMP_FLE 0
#define OUTCOMP_EVAL_MAX 1
#define OUTCOMP_EVAL_MED 2
#define OUTCOMP_EVAL_MIN 3
#define OUTCOMP_INTEG 4
#define OUTCOMP_MAP 5

//extern int testNode; // HACK RP

#include <climits>
#include <cfloat>
#include <cmath>

#include "unifield.h"
#ifdef WIN32
#define clock_t int
#define clock() 0
#define CLOCKS_PER_SEC 1
#endif

const int ucdOut_compNb = 6;
int ucdOut_components[ucdOut_compNb] = { 1, 1, 1, 1, 1, 3 };
int ucdOut_totalVecLen = 8;
char ucdOut_delim[2] = ".";
char ucdOut_labelsFTLE[256] = "FTLE.eigenval max.eigenval med.eigenval min.integration time.map";
char ucdOut_labelsFLLE[256] = "FLLE.eigenval max.eigenval med.eigenval min.integration length.map";
char ucdOut_labelsFSLE[256] = "FSLE.eigenval max.eigenval med.eigenval min.integration time.map";
char ucdOut_labelsFMLE[256] = "FMLE.eigenval max.eigenval med.eigenval min.integration time.map";
char ucdOut_labelsFALE[256] = "FALE.eigenval max.eigenval med.eigenval min.integration time.map";

void computeFlowMap(UniSys *us,
                    Unstructured *unst,
                    int crop_ucd,
                    int mode,
                    float integration_time,
                    float integration_length,
                    int integ_steps_max,
                    int forward,
                    bool *nodeDisabled, int &nodesDisabled,
                    bool disableBoundaryCells,
                    int outcomp_integ,
                    Unstructured *unst_out,
                    bool continueMap,
                    float lastTime,
                    float lastLength,
                    int &stepsDone,
                    std::vector<int> *nodes,
                    bool unsteady, const char *velocity_file, float start_time,
                    bool setupTransient,
                    bool destroyTransient,
                    int verboseTransient,
                    Unstructured *map,
                    //AVSfield_float *trajectories,
                    UniField *unif_traj,
                    int *trajVertCnt)
{ // outcomp_integ: if >=0, using this instead of OUTCOMP_INTEG
    // continueMap: if true, map is continued, otherwise starting at nodes
    //              if true, start_time has to be still the first total time step
    //              if true, lastTime has to be the last previous time,
    //                       measured from start_time
    //              if true, lastLength has to be the last previous length,
    //                       measured from node position
    //              if true, stepsDone inputs and outputs total cnt of steps done
    // nodes: if NULL, all nodes of unst_out are mapped
    // setupTransient: if transient mode is once set, this can be set to false,
    //                 not supported for dataDict method (TODO)
    // verboseTransient: 0 for quiet, 1 for initial, 2 for verbose
    // unif_traj: if NULL, no output of trajectories
    // trajVertCnt: if not NULL, stores vertex count of trajectories

    int outcompInteg;
    if (outcomp_integ >= 0)
        outcompInteg = outcomp_integ;
    else
        outcompInteg = OUTCOMP_INTEG;

    // sequential streamline / pathline integration -----------------------------
    if ((!unsteady) || PATHLINES_SEQUENTIAL)
    {

        // NOTE: no support for unsteady && !USE_TRANSIENT_UNSTRUCTURED

        if (unsteady)
        {
// set unst to transient mode
#if USE_DATA_DICT
            char path[1000];
            strcpy(path, velocity_file);
            unst->setTransient(dirname(path), 3, false);
#else
            if (setupTransient)
                unst->setTransientFile(velocity_file, verboseTransient);
#endif
        }

        // go
        for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
        {

            int n;
            if (nodes)
                n = (*nodes)[nIdx];
            else
                n = nIdx;

            if (nodeDisabled[n])
            {
                continue;
            }

// do not continue trajectories that already stopped
#if 0
      if (continueMap) {
        if (mode == MODE_FTLE) {
          // HACK ###
          if (unst_out->getScalar(n, outcompInteg) < lastTime) {
            printf("stopped: time=%g lastTime=%g\n",
                 unst_out->getScalar(n, outcompInteg),
                 lastTime);
            continue;
          }
        }
        else { // FLLE
          // HACK ###
          if (unst_out->getScalar(n, outcompInteg) < lastLength) {
            continue;
          }
        }
      }
#else
            // #### this is a HACK (assumes that trajectories!=NULL)
            if (continueMap && unif_traj &&
                //(I2DV(trajectories, trajVertCnt[n], n)[0] == -1)
                (unif_traj->getVectorComp(trajVertCnt[n], n, 0) == -1))
            {
                continue;
            }
#endif

            // status
            if (nodes)
            {
                if (!(nIdx % 100))
                {
                    char buf[256];
                    sprintf(buf, "mapping node %d (%d)", nIdx, n);
                    us->moduleStatus(buf, (int)((100.0 * nIdx) / nodes->size()));
                }
            }
            else
            {
                if (!(n % 100))
                {
                    char buf[256];
                    sprintf(buf, "mapping node %d", n);
                    us->moduleStatus(buf, (int)((100.0 * n) / unst_out->nNodes));
                }
            }

            // compute streamline / pathline

            // get start point
            vec3 start;
            if (continueMap)
                map->getVector3(n, start);
            else
                unst_out->getCoords(n, start);

            if (crop_ucd)
            {
#if 1
                printf("cropping not yet implemented\n");
#else
                if ((start[0] < origin_x) || (start[0] > origin_x + MAXX(map) * voxel_size) || // ### HACK (map)
                    (start[1] < origin_y) || (start[1] > origin_y + MAXY(map) * voxel_size) || // ### HACK (map)
                    (start[2] < origin_z) || (start[2] > origin_z + MAXZ(map) * voxel_size) // ### HACK (map)
                                             * ++)
                {
                    continue;
                }
#endif
            }

            if (!continueMap)
            {
                // set default to trajectory field
                //if (trajectories) I2DV(trajectories, 0, n)[0] = -1;
                if (unif_traj)
                    unif_traj->setVectorComp(0, n, 0, -1);

                // set default for "integration time" / "integration length"
                unst_out->setScalar(n, outcompInteg, 0);
            }

            //if (n == 117) {	// HACK RP
            //bool b = unst->findCell(start, start_time);	// HACK RP
            //	printf("*** findCell[117] = %d, start=(%g,%g,%g)\n", b, start[0], start[1], start[2]);	// HACK RP
            //  }	// HACK RP

            // using unst UCD for integration
            // ############### TODO: handle continueMap case (don't set nodeDisabled)
            if (!unst->findCell(start, start_time))
            {
#if 0 // ######### replaced for quick hack for avoiding holes in FTLEM etc, 2007-06-30
        nodeDisabled[n] = true;
        nodesDisabled++;
        continue;
#else
                //if (continueMap && trajectories &&
                //    (I2DV(trajectories, trajVertCnt[n], n)[0] == -1)) {
                if (continueMap && unif_traj && (unif_traj->getVectorComp(trajVertCnt[n], n, 0) == -1))
                {
                    continue; // ##################### OK?
                }
                // else if (continueMap && trajectories &&
                //         (I2DV(trajectories, trajVertCnt[n], n)[0] != -1)) {
                else if (continueMap && unif_traj && (unif_traj->getVectorComp(trajVertCnt[n], n, 0) != -1))
                {
                    //I2DV(trajectories, trajVertCnt[n] + 1, n)[0] = -1; // ############# OK?
                    unif_traj->setVectorComp(trajVertCnt[n] + 1, n, 0, -1);
                    continue;
                }
                else
                {
                    nodeDisabled[n] = true;
                    nodesDisabled++;
                    continue;
                }
#endif
            }

            // integrate
            //double dt = unst->integrate(start, forward, integration_time, 1e9,
            //                            integ_steps_max/* ### max steps*/);

            //float *x=NULL, *y=NULL, *z=NULL;
            int curTrajOffsetI = 0;
            int curTrajOffsetJ = 0;
            if (unif_traj)
            {
                if (continueMap)
                {
                    //int dims[2] = { MAXX(trajectories), MAXY(trajectories) };
                    //int dims[2] = { unif_traj->getDim(0), unif_traj->getDim(1) };
                    //x = trajectories->points + n * dims[0] + (stepsDone + 1);
                    //y = x + dims[0] * dims[1];
                    //z = y + dims[0] * dims[1];
                    curTrajOffsetI = stepsDone + 1;
                    curTrajOffsetJ = n;
                }
                else
                {
                    // store seed as first point
                    //int dims[2] = { MAXX(trajectories), MAXY(trajectories) };
                    //x = trajectories->points + n * dims[0];
                    //y = x + dims[0] * dims[1];
                    //z = y + dims[0] * dims[1];
                    curTrajOffsetI = 0;
                    curTrajOffsetJ = n;
                    //*(x++) = start[0];
                    //*(y++) = start[1];
                    //*(z++) = start[2];
                    unif_traj->setCoord(curTrajOffsetI, curTrajOffsetJ, start);
                    curTrajOffsetI++;
                    //I2DV(trajectories, 0, n)[0] = 0;
                    unif_traj->setVectorComp(0, n, 0, 0);
                }
            }

// integrate
#if 0 // DELETEME
      double itime = integration_time / integ_steps_max;
      double timeTot = 0.0;
#else
            double itime = 1e20;
            double timeTot = 0.0;
            double lengthTot = 0.0;
            if (continueMap)
            {
                if (mode == MODE_FTLE)
                {
                    timeTot = unst_out->getScalar(n, outcompInteg);
                    //timeTot = lastTime;
                }
                else
                {
                    timeTot = lastTime;
                    lengthTot = unst_out->getScalar(n, outcompInteg);
                }
            }
            double integTot = 0.0;
            double ilength = 1e20;
            if (mode == MODE_FTLE)
                itime = integration_time / integ_steps_max;
            else
                ilength = integration_length / integ_steps_max; // FLLE
#endif
            bool aborted = false;
            //bool stagnation = false;
            for (int step = 0; step < integ_steps_max; step++)
            {

                int stepTot = step;
                if (continueMap)
                {
                    stepTot = stepsDone + step;
                }

                //if (stepTot > MAXX(trajectories) - 1) {
                //  printf("#################################\n");
                //  continue;
                //}

                double ds;
                double dt = unst->integrate(start, forward, itime, // 1e9,
                                            ilength,
                                            1000000, // upper limit of steps
                                            4, false,
                                            (forward ? start_time + timeTot : start_time - timeTot),
                                            &ds);
#if 0 // DELETEME
        timeTot += dt;
#else
                if (mode == MODE_FTLE)
                {
                    timeTot += dt;
                    integTot = timeTot;
                }
                else
                {
                    timeTot += dt;
                    lengthTot += ds;
                    integTot = lengthTot;
                }
#endif

                if (((mode == MODE_FTLE) && (dt == 0.0)) || ((mode == MODE_FLLE) && (ds == 0.0)))
                { // TODO ###### no -1 at end in case of FLLE?
#if 1
                    //if (trajectories) I2DV(trajectories, stepTot+1, n)[0] = -1;
                    if (unif_traj)
                        unif_traj->setVectorComp(stepTot + 1, n, 0, -1);
                    if (trajVertCnt)
                        trajVertCnt[n] = stepTot + 1;
                    aborted = true;
                    break;

#else // handles stagnation at boundaries (let integration continue)

                    bool found = true;
                    vec3 veloStag;
                    if (!unst->findCell(start, start_time + timeTot))
                    {
                        found = false;
                        printf("ERROR: stagnation handling: findCell() failed\n");
                    }
                    else
                    {
                        unst->interpolateVector3(veloStag, start_time + timeTot);
                    }
                    vec3 veloStart;
                    vec3 trueStart;
                    unst_out->getCoords(n, trueStart);
                    if (!unst->findCell(trueStart, start_time))
                    {
                        printf("ERROR: stagnation handling: findCell() of true start failed\n");
                    }
                    unst->interpolateVector3(veloStart, start_time);

                    if ((!found) || (vec3mag(veloStag) > vec3mag(veloStart) / 1000.0))
                    {
                        //if ((!found) || (vec3mag(veloStag) > 10.0)) {

                        //printf("veloStart=%g veloStag=%g -> no stagnation\n", vec3mag(veloStart), vec3mag(veloStag));

                        // no stagnation (but moved into outlet) -> abort integration
                        //if (trajectories) I2DV(trajectories, stepTot+1, n)[0] = -1;
                        if (unif_traj)
                            unif_traj->setVectorComp(stepTot + 1, n, 0, -1);
                        if (trajVertCnt)
                            trajVertCnt[n] = stepTot + 1;
                        aborted = true;
                        break;
                    }
                    else
                    {
                        if (!stagnation)
                        {
                            printf("veloStart=%g veloStag=%g -> STAGNATION at node %d\n", vec3mag(veloStart), vec3mag(veloStag), n);
                            stagnation = true;
                        }
                    }

#endif
                }
#ifdef WIN32
                if (_isnan(start[0]) || _isnan(start[1]) || _isnan(start[2]))
                {
#else
                if (std::isnan(start[0]) || std::isnan(start[1]) || std::isnan(start[2]))
                {
#endif
                    printf("got NAN position after integration\n");
                    //if (trajectories) I2DV(trajectories, stepTot+1, n)[0] = -1;
                    if (unif_traj)
                        unif_traj->setVectorComp(stepTot + 1, n, 0, -1);
                    if (trajVertCnt)
                        trajVertCnt[n] = stepTot + 1;
                    aborted = true;
                    break;
                }

#if 1 // ####################
                //if (!continueMap) { // ##############
                //if (trajectories) {
                if (unif_traj)
                {
                    //*(x++) = start[0];
                    //*(y++) = start[1];
                    //*(z++) = start[2];
                    unif_traj->setCoord(curTrajOffsetI, curTrajOffsetJ, start);
                    curTrajOffsetI++;
                    //I2DV(trajectories, stepTot+1, n)[0] = integTot;
                    unif_traj->setVectorComp(stepTot + 1, n, 0, integTot);
                }

                if (trajVertCnt)
                {
                    trajVertCnt[n] = stepTot + 2;
                }
//}
#endif
            }

            //stepsDone += integ_steps_max;

            // set "integration time" / "integration length"
            unst_out->setScalar(n, outcompInteg, integTot);

            // store map
            map->setVector3(n, start);

            // disable if node belongs to boundary cell
            if (disableBoundaryCells)
            {
                vec3 pos;
                unst_out->getCoords(n, pos);
                unst->findCell(pos);
                if (unst->domainBoundaryCell(unst->getCellIndex()))
                {
                    //nodeDisabled[n] = true;
                    //nodesDisabled++;
                    unst_out->setScalar(n, outcompInteg, 0.0); // ### HACK for disabling TODO
                }
            }
        }

        stepsDone += integ_steps_max;

        if (unsteady)
        {
            printf("total count of mmap() calls: %d (should be as low as possible)\n", unst->getTransientFileMapNb());

#if !USE_DATA_DICT
            // TODO: similar for DataDict
            if (destroyTransient)
                unst->unsetTransientFile();
#endif
        }
    }
    else
    {

        // parallel pathline integration ------------------------------------------

        // ### this mode will probably only be used for testing purpose because
        //     sequential integration is faster even in the transient case if
        //     transientFile method is used

        if ((mode == MODE_FLLE) || (mode == MODE_FSLE) || (mode == MODE_FMLE) || (mode == MODE_FALE))
            printf("computeFlowMap: ERROR: FLLE/FSLE/FMLE/FALE mode not yet implemented (TODO)\n");

        if (continueMap)
            printf("computeFlowMap: ERROR: continueMap mode not yet implemented (TODO)\n");

#if USE_TRANSIENT_UNSTRUCTURED
#if USE_DATA_DICT
        char path[1000];
        strcpy(path, velocity_file);
        unst->setTransient(dirname(path));
#else
        if (setupTransient)
            unst->setTransientFile(velocity_file, verboseTransient);
#endif
#else
        // Initialize the data dictionary
        DataDict dd;
        char path[1000];
        strcpy(path, velocity_file);
        dd.setDataDict(dirname(path), 3, true);
        dd.setByteSize(unst->nNodes * 3 * sizeof(float)); // Assume velocity is the first component in dd files
        printf("new data dict created\n");

        // Load data of start time
        NodeCompDataPtr nc;
        nc.ptrs.push_back(dd.interpolate(start_time));
        nc.stride = 3;
        //unst->nodeComponentDataPtrs[compVelo] = nc;
        //unst->selectVectorNodeData(compVelo);
        unst->nodeComponentDataPtrs[unst->getVectorNodeDataComponent()] = nc;
        unst->selectVectorNodeData(unst->getVectorNodeDataComponent());
#endif

        // initialize
        //for (int n=0; n<unst_out->nNodes; n++) {
        for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
        {
            int n;
            if (nodes)
                n = (*nodes)[nIdx];
            else
                n = nIdx;

            // seeds
            vec3 start;
            unst_out->getCoords(n, start);
            map->setVector3(n, start);

            // set default to trajectory field
            //if (trajectories) I2DV(trajectories, 0, n)[0] = -1;
            if (unif_traj)
                unif_traj->setVectorComp(0, n, 0, -1);

            // set default for "integration time" / "integration length"
            unst_out->setScalar(n, outcompInteg, 0);

            // using default unst UCD
            if (!unst->findCell(start, start_time))
            {
                nodeDisabled[n] = true;
                nodesDisabled++;
            }

            //if (trajectories) {
            if (unif_traj)
            {
                // store seed as first point
                //int dims[2] = { MAXX(trajectories), MAXY(trajectories) };
                //float *x, *y, *z;
                //x = trajectories->points + n * dims[0];
                //y = x + dims[0] * dims[1];
                //z = y + dims[0] * dims[1];
                //*(x++) = start[0];
                //*(y++) = start[1];
                //*(z++) = start[2];
                unif_traj->setCoord(0, n, start);
                //I2DV(trajectories, 0, n)[0] = 0;
                unif_traj->setVectorComp(0, n, 0, 0);
            }
        }

        integration_time *= (forward ? 1.0 : -1.0);
        //bool fwd = integration_time >= 0.0;
        bool fwd = forward;
        double interval = (integration_time) / integ_steps_max;
        double absInterval = interval >= 0 ? interval : -interval;

        for (int step = 0; step < integ_steps_max; step++)
        {

            printf("step=%d time=%g     \r", step, start_time + step * interval);
            fflush(stdout);

#if USE_TRANSIENT_UNSTRUCTURED
// TODO: set time as state inside Unstructured
#else
            //delete [] unst->nodeComponentDataPtrs[compVelo].ptrs[0];
            delete[] unst -> nodeComponentDataPtrs[unst->getVectorNodeDataComponent()].ptrs[0];

            NodeCompDataPtr nc;
            nc.ptrs.push_back(dd.interpolate(start_time + step * interval));
            nc.stride = 3;
            //unst->nodeComponentDataPtrs[compVelo] = nc;
            unst->nodeComponentDataPtrs[unst->getVectorNodeDataComponent()] = nc;

            //unst->selectVectorNodeData(compVelo);
            unst->selectVectorNodeData(unst->getVectorNodeDataComponent());
#endif

            //for (int n=0; n<unst_out->nNodes; n++) {
            for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
            {
                int n;
                if (nodes)
                    n = (*nodes)[nIdx];
                else
                    n = nIdx;

                if (nodeDisabled[n])
                {
                    continue;
                }

                // status
                if (nodes)
                {
                    if (!(nIdx % 100))
                    {
                        char buf[256];
                        sprintf(buf, "mapping node %d (%d) step %d", nIdx, n, step);
                        us->moduleStatus(buf, (int)((100.0 * nIdx) / nodes->size()));
                    }
                }
                else
                {
                    if (!(n % 100))
                    {
                        char buf[256];
                        sprintf(buf, "mapping node %d step %d", n, step);
                        us->moduleStatus(buf, (int)((100.0 * n) / unst_out->nNodes));
                    }
                }

                // compute path line

                // get start point
                vec3 start;
                //unst_out->getCoords(n, start);
                map->getVector3(n, start);

                if (crop_ucd)
                {
#if 1
                    printf("cropping not yet implemented\n");
#else
                    if ((start[0] < origin_x) || (start[0] > origin_x + MAXX(map) * voxel_size) || // ### HACK (map)
                        (start[1] < origin_y) || (start[1] > origin_y + MAXY(map) * voxel_size) || // ### HACK (map)
                        (start[2] < origin_z) || (start[2] > origin_z + MAXZ(map) * voxel_size) // ### HACK (map)
                        )
                    {
                        continue;
                    }
#endif
                }

#if 0
        // set default to trajectory field
        if (trajectories) I2DV(trajectories, 0, n)[0] = -1;
        
        // set default for "integration time" / "integration length"
        unst_out->setScalar(n, outcompInteg, 0);
        
        // using unst UCD for integration
        if (!unst->findCell(start, start_time + step * interval)) {
          nodeDisabled[n] = true;
          nodesDisabled++;
          continue;
        }
        
        // integrate
        //double dt = unst->integrate(start, forward, integration_time, 1e9,
        //                            integ_steps_max/* ### max steps*/);
        
        if (trajectories) {
          // store seed as first point
          int dims[2] = { MAXX(trajectories), MAXY(trajectories) };
          float *x, *y, *z;
          x = trajectories->points + n * dims[0];
          y = x + dims[0] * dims[1];
          z = y + dims[0] * dims[1];
          *(x++) = start[0];
          *(y++) = start[1];
          *(z++) = start[2];
          I2DV(trajectories, 0, n)[0] = 0;
        }
#endif

                // integrate
                //double itime = integration_time / integ_steps_max;
                //double timeTot = 0.0;
                double timeTot = unst_out->getScalar(n, outcompInteg);
                //for (int step=0; step<integ_steps_max; step++) {

                if (((step > 0) &&
                     //(trajectories && (I2DV(trajectories, step, n)[0] == -1))
                     (unif_traj && (unif_traj->getVectorComp(step, n, 0) == -1))) || (step >= integ_steps_max))
                {
                    //if (trajectories) I2DV(trajectories, step+1, n)[0] = -1;
                    if (unif_traj)
                        unif_traj->setVectorComp(step + 1, n, 0, -1);
                    continue;
                }

#if USE_TRANSIENT_UNSTRUCTURED
                double dt = unst->integrate(start, fwd, absInterval, 1e19,
                                            1000000 /* upper limit of steps */,
                                            4, false,
                                            start_time + step * interval);
#else
                double dt = unst->integrate(start, fwd, absInterval, 1e19,
                                            1000000 /* upper limit of steps */, 4, false);
#endif

                timeTot += dt;

                if (dt == 0.0)
                {
                    //if (trajectories) I2DV(trajectories, step+1, n)[0] = -1;
                    if (unif_traj)
                        unif_traj->setVectorComp(step + 1, n, 0, -1);
                    // break;
                    continue;
                }
#ifdef WIN32
                if (_isnan(start[0]) || _isnan(start[1]) || _isnan(start[2]))
                {
#else
                if (std::isnan(start[0]) || std::isnan(start[1]) || std::isnan(start[2]))
                {
#endif
                    printf("got NAN position after integration\n");
                    //if (trajectories) I2DV(trajectories, step+1, n)[0] = -1;
                    if (unif_traj)
                        unif_traj->setVectorComp(step + 1, n, 0, -1);
                    //break;
                    continue;
                }

                //if (trajectories) {
                if (unif_traj)
                {
                    //int dims[2] = { MAXX(trajectories), MAXY(trajectories) };
                    //float *x, *y, *z;
                    //x = trajectories->points + n * dims[0];
                    //y = x + dims[0] * dims[1];
                    //z = y + dims[0] * dims[1];
                    //x[step+1] = start[0];
                    //y[step+1] = start[1];
                    //z[step+1] = start[2];
                    unif_traj->setCoord(step + 1, n, start);
                    //I2DV(trajectories, step+1, n)[0] = timeTot;
                    unif_traj->setVectorComp(step + 1, n, 0, timeTot);
                    //}
                }

                // set "integration time" / "integration length"
                unst_out->setScalar(n, outcompInteg, timeTot);

                // store map
                map->setVector3(n, start);
            }
        }

        // disable if node belongs to boundary cell
        if (disableBoundaryCells)
        {
            for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
            {
                int n;
                if (nodes)
                    n = (*nodes)[nIdx];
                else
                    n = nIdx;

                vec3 pos;
                unst_out->getCoords(n, pos);
                unst->findCell(pos);
                if (unst->domainBoundaryCell(unst->getCellIndex()))
                {
                    //nodeDisabled[n] = true;
                    //nodesDisabled++;
                    unst_out->setScalar(n, outcompInteg, 0.0); // ### HACK for disabling TODO
                }
            }
        }

        printf("total count of mmap() calls: %d\n", unst->getTransientFileMapNb());

#if !USE_DATA_DICT
        // TODO: similar for DataDict
        if (destroyTransient)
            unst->unsetTransientFile();
#endif
    }
}

void computeFTLE_atNode(UniSys *us,
                        Unstructured *map,
                        bool *nodeDisabled,
                        bool *nodeLonely,
                        bool *nodeDisabled2,
                        float *defaultFTLE,
                        int ln, int divT,
                        float integration_size, int node,
                        bool setFLE,
                        Unstructured *unst_out,
                        int &complEigenvalCnt)
{ // nodeDisabled: disables for all operations (even gradient neighborhood)
    // nodeLonely: may be NULL
    // nodeDisabled2: may be NULL (does not disable for gradient neighborhood)
    // defaultFTLE: may be NULL
    // gradient of map must already be computed

    int n = node;

    if (nodeDisabled[n])
    {
        if (setFLE && defaultFTLE)
            unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
        //continue;
        return;
    }

    if (nodeLonely && nodeLonely[n])
    {
        if (setFLE && defaultFTLE)
            unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
        // continue;
        return;
    }

    if (nodeDisabled2 && nodeDisabled2[n])
    {
        if (setFLE && defaultFTLE)
            unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
        // continue;
        return;
    }

    // status
    if (!(n % 100))
    {
        char buf[256];
        sprintf(buf, "FTLE for node %d", n);
        us->moduleStatus(buf, (int)((100.0 * n) / map->nNodes));
    }

    // compute FTLE
    {
        fmat3 grad;
        map->getMatrix3(n, 0, Unstructured::OP_GRADIENT, grad);

        // (map gradient)T * (map gradient)
        fmat3 gradT, ATA;
        fmat3trp(grad, gradT);
        fmat3mul(gradT, grad, ATA);

        // get eigenvalues
        mat3 ATAD;
        fmat3tomat3(ATA, ATAD);
        // FIX 2007-08-15: force symmetric matrix
        mat3symm(ATAD, ATAD);
        vec3 eigenvalues;
        bool allReal = (mat3eigenvalues(ATAD, eigenvalues) == 3);
        if (!allReal)
        {
            //printf("bug: got complex eigenvalues at node %d, skipping\n", n);
            complEigenvalCnt++;
            //continue;
            return;
        }

        // sort eigenvalues
        //double evMax = max(eigenvalues[0], max(eigenvalues[1], eigenvalues[2]));
        //double evMin = min(eigenvalues[0], min(eigenvalues[1], eigenvalues[2]));
        double evalsDesc[3] = { eigenvalues[0], eigenvalues[1], eigenvalues[2] };
        if (evalsDesc[0] < evalsDesc[1])
        {
            double w = evalsDesc[0];
            evalsDesc[0] = evalsDesc[1];
            evalsDesc[1] = w;
        }
        if (evalsDesc[1] < evalsDesc[2])
        {
            double w = evalsDesc[1];
            evalsDesc[1] = evalsDesc[2];
            evalsDesc[2] = w;
        }
        if (evalsDesc[0] < evalsDesc[1])
        {
            double w = evalsDesc[0];
            evalsDesc[0] = evalsDesc[1];
            evalsDesc[1] = w;
        }

        double FLE;
        if (fabs(integration_size) > 0.0)
        {
            if (ln && divT)
                FLE = log(sqrt(evalsDesc[0])) / fabs(integration_size);
            else if (ln)
                FLE = log(sqrt(evalsDesc[0]));
            else if (divT)
                FLE = sqrt(evalsDesc[0]) / fabs(integration_size);
            else
                FLE = sqrt(evalsDesc[0]);
        }
        else
            FLE = 0.0; // ###
        //unst_out->setScalar(n, FLE);
        if (setFLE)
            unst_out->setScalar(n, OUTCOMP_FLE, FLE);
        unst_out->setScalar(n, OUTCOMP_EVAL_MAX, evalsDesc[0]);
        unst_out->setScalar(n, OUTCOMP_EVAL_MED, evalsDesc[1]);
        unst_out->setScalar(n, OUTCOMP_EVAL_MIN, evalsDesc[2]);

        //if (node == testNode) { // HACK RP
        //   printf("node=%d ev = %g t/s = %g FTLE = %g\n", node, evalsDesc[0], integration_size, FTLE);
        //}
    }
}

void computeFTLE(UniSys *us,
                 Unstructured *map,
                 bool *nodeDisabled,
                 bool gradNeighNodeDisable,
                 bool *nodeDisabled2,
                 float *defaultGrad,
                 float *defaultFTLE,
                 int smoothing_range,
                 int ln, int divT,
                 bool setFLE,
                 Unstructured *unst_out)
{ // nodeDisabled: disables for all operations (even gradient neighborhood)
// nodeDisabled2: may be NULL (does not disable for gradient neighborhood)
// defaultGrad: may be NULL
// defaultFTLE: may be NULL
// smoothing_range: >= 1
// unst_out:OUTCOMP_INTEG must already contain integration time/length!

// compute gradient of map
//map->gradient(0, false, smoothing_range); // BUG in unstructured  TODO
#if 0 // DELETEME
  map->gradient(0); // ### assuming component 0
#else

    // TODO: ######## should use high value, in order to prevent ridge creation ?
    //float defaultGrad[9] = { 0.0, 0.0, 0.0,
    //                       0.0, 0.0, 0.0,
    //                       0.0, 0.0, 0.0 }; // ##### ok?
    //float defaultGrad[9] = { FLT_MAX, FLT_MAX, FLT_MAX,
    //                       FLT_MAX, FLT_MAX, FLT_MAX,
    //                       FLT_MAX, FLT_MAX, FLT_MAX }; // ##### ok?

    //map->gradient(0, false, smoothing_range, nodesDisabled, defaultGrad); // ### assuming component 0 // ################# BUG in unstructured  TODO

    std::vector<int> lonelyNodes;

    if (gradNeighNodeDisable)
        map->gradient(0, false, smoothing_range, nodeDisabled, defaultGrad, &lonelyNodes); // ### assuming component 0
    else
        map->gradient(0, false, smoothing_range, NULL, defaultGrad, &lonelyNodes); // ### assuming component 0

#if 0
  // disable nodes that have too small neighborhood
  // ### this is not perfectly correct because these nodes could be still
  //     valid neighborhood nodes of other nearby nodes
  // ###################################################################
  //... still getting -inf (from log(0))
  //      and also check ridge detection part
  //      maybe store two different info: map_OK and FTLE_OK ?
  for (int n=0; n<(int)lonelyNodes.size(); n++) {
    nodeDisabled[lonelyNodes[n]] = true;
  }
#else

    // for disabling FLE computation at lonely nodes
    bool *lonelyNodesArr = new bool[map->nNodes];
    if (!lonelyNodesArr)
    {
        printf("out of memory\n");
        exit(1); // ###
    }
    for (int n = 0; n < map->nNodes; n++)
        lonelyNodesArr[n] = false;
    for (int ni = 0; ni < (int)lonelyNodes.size(); ni++)
    {
        lonelyNodesArr[lonelyNodes[ni]] = true;
    }
#endif

#endif

    //printf("HACK 2\n"); // HACK RP

    int complEigenvalCnt = 0;

    // ######## TODO: instead of setting to default in order to avoid ridge
    // generation, simply don't generate ridges in cells that share a node that
    // is disabled
    //float defaultFTLE = FLT_MAX; // ############ ok?  #################

    // compute FTLE
    for (int n = 0; n < map->nNodes; n++)
    {
        //if (n == testNode) { // HACK RP
        //   printf("node=%d t = %g\n", n, unst_out->getScalar(n, OUTCOMP_INTEG));
        //}

        if (nodeDisabled[n])
        {
            if (setFLE && defaultFTLE)
                unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
            continue;
        }

        if (lonelyNodesArr[n])
        {
            if (setFLE && defaultFTLE)
                unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
            continue;
        }

        if (nodeDisabled2 && nodeDisabled2[n])
        {
            if (setFLE && defaultFTLE)
                unst_out->setScalar(n, OUTCOMP_FLE, *defaultFTLE);
            continue;
        }

        computeFTLE_atNode(us, map, nodeDisabled, lonelyNodesArr, nodeDisabled2, defaultFTLE, ln, divT, unst_out->getScalar(n, OUTCOMP_INTEG), n, setFLE, unst_out, complEigenvalCnt);
    }

    if (complEigenvalCnt > 0)
        printf("numerical problem: got complex eigenvalues at %d nodes (skipped), TODO: correct instead of skipping!\n", complEigenvalCnt);

    // save memory: free gradient
    map->deleteNodeCompExtraData(0, Unstructured::OP_GRADIENT);

    delete[] lonelyNodesArr;
}

void computeFLE(UniSys *us,
                Unstructured *unst,
                int crop_ucd,
                int mode,
                int ln,
                int divT,
                float integration_time,
                float integration_length,
                float sepFactorMin,
                int time_intervals,
                int integ_steps_max,
                int forward,
                bool *nodeDisabled,
                int &nodesDisabled,
                bool grad_neigh_disabled,
                bool *nodeFinished,
                int &nodesFinished,
                bool disableBoundaryCells,
                std::vector<int> *nodes,
                bool unsteady,
                const char *velocity_file,
                float start_time,
                bool setupTransient,
                bool destroyTransient,
                int verboseTransient,
                Unstructured *map,
                int smoothing_range,
                //AVSfield_float *trajectories,
                UniField *unif_traj,
                int *trajVertCnt,
                Unstructured *unst_out,
                double *map_seconds,
                double *FLE_seconds)
{ // integration_time: used in MODE_FTLE, MODE_FSLE, MODE_FMLE, MODE_FALE
    // integration_length: used in MODE_FLLE
    // sepFactorMin: used in MODE_FSLE
    // time_intervals: used in MODE_FSLE, MODE_FMLE, MODE_FALE
    // nodesFinshed: must be !NULL for MODE_FSLE
    // nodes: if NULL, all nodes of unst_out are mapped
    // setupTransient: if transient mode is once set, this can be set to false,
    //                 not supported for dataDict method (TODO)
    // verboseTransient: 0 for quiet, 1 for initial, 2 for verbose
    // unif_traj: if NULL, no output of trajectories
    // trajVertCnt: if not NULL, stores vertex count of trajectories
    // map_seconds: if not NULL, returns time needed for map computation
    //              map_seconds must already be initialized
    // FLE_seconds: if not NULL, returns time needed for FLE computation
    //              FLE_seconds must already be initialized

    float defaultGrad[9] = { 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0 }; // ##### ok?

    // need to set FLE to FLT_MAX at disabled nodes for preventing ridge
    // generation at these locations
    float defaultFLE = FLT_MAX;

    if ((mode == MODE_FTLE) || (mode == MODE_FLLE))
    {
        clock_t m_start = clock();

        // compute flow map (position of given node after time of advection)
        int wsteps = 0;
        computeFlowMap(us,
                       unst,
                       //crop_ucd,
                       false,
                       mode,
                       integration_time, integration_length, integ_steps_max,
                       forward, nodeDisabled, nodesDisabled,
                       false, // omit_boundary_cells
                       -1, // outcomp_integ
                       unst_out,
                       false, // continue map
                       0.0, 0.0, wsteps,
                       nodes,
                       unsteady, velocity_file, start_time,
                       setupTransient, destroyTransient, verboseTransient,
                       map, unif_traj, trajVertCnt);

        clock_t m_end = clock();
        if (map_seconds)
            *map_seconds += (m_end - m_start) / ((double)CLOCKS_PER_SEC);
        clock_t f_start = clock();

        // compute FTLE
        // #### actually recomputing gradient at all nodes, not only affected nodes
        // TODO ######
        computeFTLE(us, map, nodeDisabled, grad_neigh_disabled, NULL, defaultGrad, &defaultFLE, smoothing_range, ln, divT, true, unst_out);

        clock_t f_end = clock();
        if (FLE_seconds)
            *FLE_seconds += (f_end - f_start) / ((double)CLOCKS_PER_SEC);

        printf("%d of %d nodes disabled          \n", nodesDisabled, unst_out->nNodes);
    }
    else
    {

        // compute FLE variants that require incremental map computation

        // node is Finished if FSLE separation reached,
        // but integration is continued for FSLE computation of nearby trajectories
        if ((mode == MODE_FSLE) && (!nodeFinished))
        {
            printf("computeFLE: FSLE mode needs nodeFinished\n");
            return;
        }

        if (nodes &&
            //!trajectories
            !unif_traj)
        {
            printf("computeFLE: incremental map computation needs trajectories\n");
            return;
        }

        if (nodes && !trajVertCnt)
        {
            printf("computeFLE: incremental map computation needs trajVertCnt\n");
            return;
        }

        // init
        if ((mode == MODE_FMLE) || (mode == MODE_FALE))
        {
#if 0 // #################### this has beed disabled before 2007-03-16 due to
      //                      bad effects, but it is needed for adaptive ...
      //for (int nIdx=0; nIdx<(nodes?(int)nodes->size():unst_out->nNodes); nIdx++) {
        int n;
        if (nodes) n = (*nodes)[nIdx];
        else n = nIdx;
#else
            for (int n = 0; n < unst_out->nNodes; n++)
            {
#endif

            switch (mode)
            {
            case MODE_FMLE:
#if 1
                unst_out->setScalar(n, OUTCOMP_FLE, -FLT_MAX);
#else
                    if (unst_out->getScalar(n, OUTCOMP_FLE) != defaultFLE)
                    {
                        unst_out->setScalar(n, OUTCOMP_FLE, -FLT_MAX);
                    }
#endif
                break;
            case MODE_FALE:
                unst_out->setScalar(n, OUTCOMP_FLE, 0.0);
                break;
            }
        }
    }

    // get nodes that are not treated
    bool *nodeTreated = NULL;
    if (nodes)
    {
        nodeTreated = new bool[unst_out->nNodes];
        if (!nodeTreated)
        {
            printf("out of memory\n");
            return;
        }
        for (int n = 0; n < unst_out->nNodes; n++)
        {
            nodeTreated[n] = false;
        }

        for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
        {
            int n;
            if (nodes)
                n = (*nodes)[nIdx];
            else
                n = nIdx;

            nodeTreated[n] = true;
        }
    }

    // helping data for FSLE
    double *lastSep = NULL;
    if (mode == MODE_FSLE)
    {
        lastSep = new double[unst_out->nNodes];
        if (!lastSep)
        {
            printf("out of memory\n");
            return;
        }
    }

    // #### this is a bug-fix: nodeFinished has to be initialized here
    // and therefore not subdivided and not passed here as a param
    if (mode == MODE_FSLE)
    {
        for (int n = 0; n < unst_out->nNodes; n++)
        {
            nodeFinished[n] = false;
        }
    }

    // init outside
    //if (map_seconds) *map_seconds = 0.0;
    //if (FLE_seconds) *FLE_seconds = 0.0;

    // go
    int stepsDone = 0;
    double time = 0.0;
    for (int tI = 0; tI < time_intervals; tI++)
    {

        // proceed flow map
        {

            clock_t m_start = clock();

            int steps = (tI + 1) * integ_steps_max / time_intervals - stepsDone;
            //printf("tI=%d steps=%d stepsDoneBefore=%d\n", tI, steps, stepsDone);

            // get position at not treated nodes from already computed trajectories
            if (nodes)
            {
                //int dims[2] = { MAXX(trajectories), MAXY(trajectories) };

                for (int n = 0; n < unst_out->nNodes; n++)
                {

                    if (!nodeTreated[n] && (trajVertCnt[n] > 0))
                    {

// get vertex index inside trajectories that matches tI
#if 0 // DELETEME
              int vertex = stepsDone + steps;
              if (I2DV(trajectories, vertex, n)[0] <= I2DV(trajectories, 0, n)[0]) {
                vertex = 0; // ######
              }
#else
                            int vertex = stepsDone + steps;
                            if (vertex > trajVertCnt[n] - 1)
                            {
                                vertex = trajVertCnt[n] - 1;
                                //printf("clamped node %d (oldvert=%d newvert=%d) ****\n",
                                //     n, stepsDone + steps, vertex);
                            }
#endif

                        // get position from trajectories
                        vec3 pos;
                        //float *x = trajectories->points + n * dims[0] + vertex;
                        //float *y = x + dims[0] * dims[1];
                        //float *z = y + dims[0] * dims[1];
                        //pos[0] = *x;
                        //pos[1] = *y;
                        //pos[2] = *z;
                        unif_traj->getCoord(vertex, n, pos);

                        //printf("getting trajectory node=%d disabled=%d vertex=%d at time=%g time0=%g vec=(%g,%g,%g)\n",
                        //     n, nodeDisabled[n], vertex, I2DV(trajectories, vertex, n)[0], I2DV(trajectories, 0, n)[0],
                        //   pos[0], pos[1], pos[2]);

                        // set map
                        map->setVector3(n, pos);
                    }
                }
            }

            // proceed (new) nodes
            int stepsDoneW = stepsDone;
            computeFlowMap(us,
                           unst,
                           //crop_ucd,
                           false,
                           MODE_FTLE,
                           integration_time / time_intervals,
                           0.0, //integration_length unused in FTLE mode
                           steps,
                           forward, nodeDisabled, nodesDisabled,
                           false, // omit_boundary_cells
                           -1, // outcomp_integ
                           unst_out,
                           !(tI == 0), // continue map
                           time,
                           0.0, // lastLength unused in FTLE mode
                           stepsDoneW,
                           nodes,
                           unsteady, velocity_file, start_time,
                           //(tI == 0) && setupTransient,
                           setupTransient, // ##### inefficient
                           //false, // TODO: test
                           destroyTransient, // ##### inefficient
                           verboseTransient,
                           map, unif_traj,
                           trajVertCnt);

            time += integration_time / time_intervals;
            stepsDone += steps;

//printf("stepsDoneAfter=%d\n", stepsDone);

#ifndef WIN32
            clock_t m_end = clock();
            if (map_seconds)
                *map_seconds += (m_end - m_start) / ((double)CLOCKS_PER_SEC);
#endif
        }

        // set  default value at disabled nodes
        // ### inefficient TODO
        // ####### supports FSLE, FMLE, FALE ? (TODO)
        for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
        {
            int n;
            if (nodes)
                n = (*nodes)[nIdx];
            else
                n = nIdx;

            if (nodeDisabled[n])
            {
                //unst_out->setScalar(n, OUTCOMP_FLE, 1.0 / integration_time);
                unst_out->setScalar(n, OUTCOMP_FLE, defaultFLE); // #### ok?
            }
        }

        // compute FLE
        {

            clock_t f_start = clock();

            // compute max eigenvalue for FLE
            // #### actually recomputing at all nodes instead of only affected ones
            computeFTLE(us, map, nodeDisabled, grad_neigh_disabled, nodeFinished, defaultGrad, NULL, smoothing_range, ln, divT, false, unst_out);

            //printf("EVAL_MAX[656] = %g\n", unst_out->getScalar(656, OUTCOMP_EVAL_MAX)); // HACK RP
            //printf("EVAL_MAX[117] = %g\n", unst_out->getScalar(117, OUTCOMP_EVAL_MAX)); // HACK RP
            //printf("nodeDisabled[117] = %d\n", nodeDisabled[117]); // HACK RP

            //DELETEME for (int nIdx=0; nIdx<(nodes?(int)nodes->size():unst_out->nNodes); nIdx++) {
            //int n;
            // if (nodes) n = (*nodes)[nIdx];
            // else n = nIdx;
            for (int n = 0; n < unst_out->nNodes; n++)
            {

                if (nodeDisabled[n] || ((mode == MODE_FSLE) && nodeFinished[n]))
                    continue;

                double sepFactor = sqrt(unst_out->getScalar(n, OUTCOMP_EVAL_MAX));

                //if (n==656 || n==117) {									// HACK RP
                //  printf("*** sepFactor[%d] = %g, time=%g\n", n, sepFactor, time);									// HACK RP
                //}									// HACK RP

                switch (mode)
                {
                case MODE_FSLE:
                {
                    //if (sepFactor <= sepFactorMin) {
                    if (sepFactor < sepFactorMin)
                    {
                        unst_out->setScalar(n, OUTCOMP_FLE, 1.0 / time);
                        lastSep[n] = sepFactor;
                    }
                    else
                    {
                        if (tI == 0)
                        {
                            unst_out->setScalar(n, OUTCOMP_FLE, 1.0 / integration_time);
                        }
                        else
                        {
#if 1 // linearly interpolate time inside last time step
                            // scheme:
                            // s1 = lastSepFactor
                            // s2 = sepFactor
                            // t1 = lastTime
                            // t2 = time
                            // t = t1 + (sepFactorMin - s1) / (s2 - s1) * (t2 - t1)

                            double t1 = 1.0 / unst_out->getScalar(n, OUTCOMP_FLE);
                            double t2 = time;
                            double s1 = lastSep[n];
                            double s2 = sepFactor;

                            double t = t1 + ((sepFactorMin - s1) / (s2 - s1)) * (t2 - t1);
                            unst_out->setScalar(n, OUTCOMP_FLE, 1.0 / t);
#endif
                        }

                        nodeFinished[n] = true;
                        nodesFinished++;
                    }
                }
                break;

                case MODE_FMLE:
                case MODE_FALE:
                {
                    double FLE;
                    if (ln && divT)
                        FLE = log(sepFactor) / fabs(time);
                    else if (ln)
                        FLE = log(sepFactor);
                    else if (divT)
                        FLE = sepFactor / fabs(time);
                    else
                        FLE = sepFactor;

                    if (mode == MODE_FMLE)
                    {
                        //if (unst_out->getScalar(n, OUTCOMP_FLE) == defaultFLE) {
                        //  continue;
                        // }
                        if (FLE > unst_out->getScalar(n, OUTCOMP_FLE))
                        {
                            unst_out->setScalar(n, OUTCOMP_FLE, FLE);
                        }
                    }
                    else
                    { // MODE_FALE
                        // ########### should clamp to FLT_MAX !!!!!! otherwise not
                        // set to FLT_MAX below in case of "disabled node"
                        float w = unst_out->getScalar(n, OUTCOMP_FLE) + FLE;
                        unst_out->setScalar(n, OUTCOMP_FLE, w);
                    }
                }
                break;
                }
            }

            clock_t f_end = clock();
            if (FLE_seconds)
                *FLE_seconds += (f_end - f_start) / ((double)CLOCKS_PER_SEC);
        }
        if (mode == MODE_FSLE)
            printf("t=%d: %d/%d/%d finished/enabled/total nodes          \n",
                   tI, nodesFinished, unst_out->nNodes - nodesDisabled,
                   unst_out->nNodes);
        else
            printf("t=%d: %d/%d enabled/total nodes          \n",
                   tI, unst_out->nNodes - nodesDisabled, unst_out->nNodes);
    }

    // post-process
    if (mode == MODE_FMLE)
    {
#if 1
        for (int n = 0; n < unst_out->nNodes; n++)
        {

            if (unst_out->getScalar(n, OUTCOMP_FLE) == -FLT_MAX)
                unst_out->setScalar(n, OUTCOMP_FLE, defaultFLE);
        }
#endif
    }
    else if (mode == MODE_FALE)
    {
#if 0
      //for (int nIdx=0; nIdx<(nodes?(int)nodes->size():unst_out->nNodes); nIdx++) {
        int n;
        if (nodes) n = (*nodes)[nIdx];
        else n = nIdx;
#else
            for (int n = 0; n < unst_out->nNodes; n++)
            {
#endif

#if 0
        unst_out->setScalar(n, OUTCOMP_FLE, 
                            unst_out->getScalar(n, OUTCOMP_FLE) / time_intervals);
#else
                // ###### this is a hack beacuse FTLE may be negative and hence sum = 0
                if ((unst_out->getScalar(n, OUTCOMP_FLE) == 0.0) || (unst_out->getScalar(n, OUTCOMP_FLE) == defaultFLE))
                    unst_out->setScalar(n, OUTCOMP_FLE, defaultFLE);
                else
                    unst_out->setScalar(n, OUTCOMP_FLE,
                                        unst_out->getScalar(n, OUTCOMP_FLE) / time_intervals);
#endif
    }
}

if (nodeTreated)
    delete[] nodeTreated;
if (destroyTransient)
    unst->unsetTransientFile();
if (lastSep)
    delete[] lastSep;
}
}

#endif
