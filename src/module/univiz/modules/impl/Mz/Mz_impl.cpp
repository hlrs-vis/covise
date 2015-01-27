/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../FLE/FLE_lib.cpp" // #### including .cpp
#include "Mz_lib.cpp"

#define OUTCOMP_MZ 0

#define COMP_S 0
#define COMP_S_COL0 1
#define COMP_S_COL1 2
#define COMP_S_COL2 3
#define COMP_M 4

void ABCsteady_CB(vec3 pos, vec3 out, double)
{
    float A = sqrt(3.);
    float B = sqrt(2.);
    float C = 1.;

    out[0] = A * sin(pos[2]) + C * cos(pos[1]);
    out[1] = B * sin(pos[0]) + A * cos(pos[2]);
    out[2] = C * sin(pos[1]) + B * cos(pos[0]);
}

void ABCsteady_veloGradient(vec3 pos, mat3 grad)
{
    float A = sqrt(3.);
    float B = sqrt(2.);
    float C = 1.;

    // u = A * sin(z) + C * cos(y)
    // v = B * sin(x) + A * cos(z)
    // w = C * sin(y) + B * cos(x)

    // du/dx =   0
    // du/dy = - C * sin(y)
    // du/dz =   A * cos(z)
    // dv/dx =   B * cos(x)
    // dv/dy =   0
    // dv/dz = - A * sin(z)
    // dw/dx = - B * sin(x)
    // dw/dy =   C * cos(y)
    // dw/dz =   0

    grad[0][0] = 0;
    grad[1][0] = B * cos(pos[0]);
    grad[2][0] = -B * sin(pos[0]);

    grad[0][1] = -C * sin(pos[1]);
    grad[1][1] = 0;
    grad[2][1] = C * cos(pos[1]);

    grad[0][2] = A * cos(pos[2]);
    grad[1][2] = -A * sin(pos[2]);
    grad[2][2] = 0;
}

void ABCsteady_SGradient(vec3 pos,
                         mat3 gradScol0,
                         mat3 gradScol1,
                         mat3 gradScol2)
{
    float A = sqrt(3.);
    float B = sqrt(2.);
    float C = 1.;

    // u = A * sin(z) + C * cos(y)
    // v = B * sin(x) + A * cos(z)
    // w = C * sin(y) + B * cos(x)

    // du/dx =   0
    // du/dy = - C * sin(y)
    // du/dz =   A * cos(z)
    // dv/dx =   B * cos(x)
    // dv/dy =   0
    // dv/dz = - A * sin(z)
    // dw/dx = - B * sin(x)
    // dw/dy =   C * cos(y)
    // dw/dz =   0

    // S[0][0] = 0;
    // S[1][0] = (B * cos(x) - C * sin(y)) / 2;
    // S[2][0] = (A * cos(z) - B * sin(x)) / 2;
    // S[0][1] = S[1][0];
    // S[1][1] = 0;
    // S[2][1] = (C * cos(y) - A * sin(z)) / 2;
    // S[0][2] = S[2][0];
    // S[1][2] = S[2][1];
    // S[2][2] = 0;

    // d/dx S[0][0] = 0;
    // d/dx S[1][0] = - B * sin(x) / 2;
    // d/dx S[2][0] = - B * cos(x) / 2;
    // d/dy S[0][0] = 0;
    // d/dy S[1][0] = - C * cos(y) / 2;
    // d/dy S[2][0] = 0;
    // d/dz S[0][0] = 0;
    // d/dz S[1][0] = 0;
    // d/dz S[2][0] = - A * sin(z) / 2;

    // d/dx S[0][1] = - B * sin(x) / 2;
    // d/dx S[1][1] = 0;
    // d/dx S[2][1] = 0;
    // d/dy S[0][1] = - C * cos(y) / 2;
    // d/dy S[1][1] = 0;
    // d/dy S[2][1] = - C * sin(y) / 2;
    // d/dz S[0][1] = 0;
    // d/dz S[1][1] = 0;
    // d/dz S[2][1] = - A * cos(z) / 2;

    // d/dx S[0][2] = - B * cos(x) / 2;
    // d/dx S[1][2] = 0;
    // d/dx S[2][2] = 0;
    // d/dy S[0][2] = 0;
    // d/dy S[1][2] = - C * sin(y) / 2;
    // d/dy S[2][2] = 0;
    // d/dz S[0][2] = - A * sin(z) / 2;
    // d/dz S[1][2] = - A * cos(z) / 2;
    // d/dz S[2][2] = 0;

    gradScol0[0][0] = 0;
    gradScol0[1][0] = -B * sin(pos[0]) / 2;
    gradScol0[2][0] = -B * cos(pos[0]) / 2;
    gradScol0[0][1] = 0;
    gradScol0[1][1] = -C * cos(pos[1]) / 2;
    gradScol0[2][1] = 0;
    gradScol0[0][2] = 0;
    gradScol0[1][2] = 0;
    gradScol0[2][2] = -A * sin(pos[2]) / 2;

    gradScol1[0][0] = -B * sin(pos[0]) / 2;
    gradScol1[1][0] = 0;
    gradScol1[2][0] = 0;
    gradScol1[0][1] = -C * cos(pos[1]) / 2;
    gradScol1[1][1] = 0;
    gradScol1[2][1] = -C * sin(pos[1]) / 2;
    gradScol1[0][2] = 0;
    gradScol1[1][2] = 0;
    gradScol1[2][2] = -A * cos(pos[2]) / 2;

    gradScol2[0][0] = -B * cos(pos[0]) / 2;
    gradScol2[1][0] = 0;
    gradScol2[2][0] = 0;
    gradScol2[0][1] = 0;
    gradScol2[1][1] = -C * sin(pos[1]) / 2;
    gradScol2[2][1] = 0;
    gradScol2[0][2] = -A * sin(pos[2]) / 2;
    gradScol2[1][2] = -A * cos(pos[2]) / 2;
    gradScol2[2][2] = 0;
}

void compute_M(vec3 velo,
               mat3 veloGrad,
               mat3 S,
               mat3 gradScol0,
               mat3 gradScol1,
               mat3 gradScol2,
               mat3 M)
{
    // time derivative of S
    // #### TODO unsteady: add d/dt S !!!
    mat3 Sdot;

    mat3 gradSx, gradSy, gradSz;
    vec3 c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z;
    mat3getcols(gradScol0, c0x, c0y, c0z);
    mat3getcols(gradScol1, c1x, c1y, c1z);
    mat3getcols(gradScol2, c2x, c2y, c2z);
    mat3setcols(gradSx, c0x, c1x, c2x);
    mat3setcols(gradSy, c0y, c1y, c2y);
    mat3setcols(gradSz, c0z, c1z, c2z);

    // compute grad(S) * u
    // tensor / vector multiplication
    mat3scal(gradSx, velo[0], gradSx);
    mat3scal(gradSy, velo[1], gradSy);
    mat3scal(gradSz, velo[2], gradSz);
    mat3copy(gradSx, Sdot);
    mat3add(Sdot, gradSy, Sdot);
    mat3add(Sdot, gradSz, Sdot);

    // S * grad
    mat3 SmulGrad;
    mat3mul(S, veloGrad, SmulGrad);

    // gradT * S
    mat3 gradT, gradTmulS;
    mat3trp(veloGrad, gradT);
    mat3mul(gradT, S, gradTmulS);

    mat3copy(Sdot, M);
    mat3add(M, SmulGrad, M);
    mat3add(M, gradTmulS, M);
}

bool Mz_positive_definite(UniSys *us,
                          //Unstructured *unst,
                          //bool unsteady,
                          //double time,
                          Unstructured *temp,
                          bool ABC_instead_unst,
                          vec3 pos,
                          bool strongHyperbolicity)
{
    // locate point
    //if (!temp->findCell(pos, time)) {
    if (!ABC_instead_unst && !temp->findCell(pos))
    {
        //us->error("Mz_positive_definite: could not locate cell");
        printf("Mz_positive_definite: ERROR: could not locate cell\n");
        return false;
    }

    // get S and M at point
    mat3 Sinterpol, Minterpol;
    if (ABC_instead_unst)
    {
        // evaluate

        // get velocity
        vec3 velo;
        ABCsteady_CB(pos, velo, 0.0); // time TODO

        // get velocity gradient
        mat3 veloGrad;
        ABCsteady_veloGradient(pos, veloGrad);

        // compute S
        mat3symm(veloGrad, Sinterpol);

        // get gradient of columns of S
        mat3 gradScol0, gradScol1, gradScol2;
        ABCsteady_SGradient(pos, gradScol0, gradScol1, gradScol2);

        compute_M(velo, veloGrad, Sinterpol, gradScol0, gradScol1, gradScol2, Minterpol);
    }
    else
    {
        // interpolate
        temp->selectVectorNodeData(COMP_S);
        temp->interpolateMatrix3(Sinterpol);
        temp->selectVectorNodeData(COMP_M);
        temp->interpolateMatrix3(Minterpol);
    }

    if (strongHyperbolicity)
    {
        double Smag2 = mat3magFrobeniusSqr(Sinterpol);
        double Mmag2 = mat3magFrobeniusSqr(Minterpol);

        bool hyperbolic;
        hyperbolic = (((4 * Smag2 * Smag2 - Mmag2) * Smag2 > mat3det(Minterpol)) && (mat3det(Minterpol) > 0));

        return hyperbolic;
    }
    else
    {
        return Mz_is_positive_definite(us, Sinterpol, Minterpol);
    }
}

void computeMz(UniSys *us,
               Unstructured *unst,
               int crop_ucd,
               float integration_time,
               int time_intervals,
               int integ_steps_max,
               int forward,
               int strong_hyperbolicity,
               bool ABC_instead_unst,
               bool *nodeDisabled,
               int &nodesDisabled,
               bool grad_neigh_disabled,
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
               double *Mz_seconds)
{ // nodes: if NULL, all nodes of unst_out are mapped
    // setupTransient: if transient mode is once set, this can be set to false,
    //                 not supported for dataDict method (TODO)
    // verboseTransient: 0 for quiet, 1 for initial, 2 for verbose
    // unif_traj: if NULL, no output of trajectories
    // trajVertCnt: if not NULL, stores vertex count of trajectories
    // map_seconds: if not NULL, returns time needed for map computation
    //              map_seconds must already be initialized
    // Mz_seconds: if not NULL, returns time needed for Mz computation
    //             Mz_seconds must already be initialized

    float defaultGrad[9] = { 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0 }; // ##### ok?

    // need to set Mz to FLT_MAX at disabled nodes for preventing ridge
    // generation at these locations
    float defaultMz = -FLT_MAX; // #### TODO, no ridges, need something for isovalues

    if (nodes &&
        //!trajectories
        !unif_traj)
    {
        printf("computeMz: Mz computation needs trajectories\n");
        return;
    }

    if (nodes && !trajVertCnt)
    {
        printf("computeMz: Mz computation needs trajVertCnt\n");
        return;
    }

    Unstructured *temp = NULL;
    if (!ABC_instead_unst)
    {
        int components[256] = {
            9, // S
            3, // S col 0
            3, // S col 1
            3, // S col 2
            9 // M
        };
        temp = new Unstructured(unst, 5, components);
    }

    // init
    // component Mz stores hyperbolicity time
    for (int n = 0; n < unst_out->nNodes; n++)
    {
        unst_out->setScalar(n, OUTCOMP_MZ, 0.0);
    }

    // get nodes that are treated
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
                        int vertex = stepsDone + steps;
                        if (vertex > trajVertCnt[n] - 1)
                        {
                            vertex = trajVertCnt[n] - 1;
                            //printf("clamped node %d (oldvert=%d newvert=%d) ****\n",
                            //     n, stepsDone + steps, vertex);
                        }

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
                           0.0, //integration_length unused in Mz mode
                           steps,
                           forward, nodeDisabled, nodesDisabled,
                           false, // omit_boundary_cells
                           1, // outcomp_integ
                           unst_out,
                           !(tI == 0), // continue map
                           time,
                           0.0, // lastLength unused in Mz mode
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

            clock_t m_end = clock();
            if (map_seconds)
                *map_seconds += (m_end - m_start) / ((double)CLOCKS_PER_SEC);
        }

        // set  default value at disabled nodes
        // ### inefficient TODO
        for (int nIdx = 0; nIdx < (nodes ? (int)nodes->size() : unst_out->nNodes); nIdx++)
        {
            int n;
            if (nodes)
                n = (*nodes)[nIdx];
            else
                n = nIdx;

            if (nodeDisabled[n])
            {
                unst_out->setScalar(n, OUTCOMP_MZ, defaultMz); // #### ok?
            }
        }

        // compute Mz hyperbolicity time
        {

            clock_t f_start = clock();

            // compute S and M
            if (((!unsteady && (tI == 0)) || unsteady) && !ABC_instead_unst)
            {
                // compute rate of strain matrix
                // ### TODO: inefficient: computing gradients etc. everywhere,
                //           should not do it at least in unsteady case

                // compute velocity gradient
                unst->gradient(unst->getVectorNodeDataComponent(),
                               false, smoothing_range, NULL,
                               defaultGrad, NULL);

                // compute S
                for (int n = 0; n < unst->nNodes; n++)
                {

                    if (nodeDisabled[n])
                        continue;

                    // get grad
                    fmat3 gradf;
                    mat3 grad;
                    unst->getMatrix3(n, unst->getVectorNodeDataComponent(),
                                     Unstructured::OP_GRADIENT, gradf);
                    fmat3tomat3(gradf, grad);

                    // compute S
                    {
                        //unst->newNodeCompExtraData(unst->getVectorNodeDataComponent(),
                        //                           TP_FLOAT, 3*3, OP_S);

                        mat3 S;
                        mat3symm(grad, S);
                        //unst->setMatrix3(n, unst->getVectorNodeDataComponent(), OP_S, S);
                        fmat3 Sf;
                        mat3tofmat3(S, Sf);
                        temp->setMatrix3(n, COMP_S, Sf);

                        vec3 Scol0, Scol1, Scol2;
                        mat3getcols(S, Scol0, Scol1, Scol2);
                        temp->setVector3(n, COMP_S_COL0, Scol0);
                        temp->setVector3(n, COMP_S_COL1, Scol1);
                        temp->setVector3(n, COMP_S_COL2, Scol2);
                    }
                }

                // compute M
                {
                    // compute gradient of S
                    // TODO ####: unsteady
                    temp->gradient(COMP_S_COL0,
                                   false, smoothing_range, NULL,
                                   defaultGrad, NULL);
                    temp->gradient(COMP_S_COL2,
                                   false, smoothing_range, NULL,
                                   defaultGrad, NULL);
                    temp->gradient(COMP_S_COL1,
                                   false, smoothing_range, NULL,
                                   defaultGrad, NULL);

                    // compute M
                    for (int n = 0; n < unst->nNodes; n++)
                    {

                        if (nodeDisabled[n])
                            continue;

                        // get velocity
                        vec3 velo;
                        unst->getVector3(n, velo);

                        // get gradient of S tensor
                        fmat3 gradScol0f, gradScol1f, gradScol2f;
                        mat3 gradScol0, gradScol1, gradScol2;
                        temp->getMatrix3(n, COMP_S_COL0, Unstructured::OP_GRADIENT, gradScol0f);
                        temp->getMatrix3(n, COMP_S_COL1, Unstructured::OP_GRADIENT, gradScol1f);
                        temp->getMatrix3(n, COMP_S_COL2, Unstructured::OP_GRADIENT, gradScol2f);
                        fmat3tomat3(gradScol0f, gradScol0);
                        fmat3tomat3(gradScol1f, gradScol1);
                        fmat3tomat3(gradScol2f, gradScol2);

                        // get S
                        fmat3 Sf;
                        mat3 S;
                        temp->getMatrix3(n, COMP_S, Sf);
                        fmat3tomat3(Sf, S);

                        // get grad
                        fmat3 gradf;
                        mat3 grad;
                        unst->getMatrix3(n, unst->getVectorNodeDataComponent(),
                                         Unstructured::OP_GRADIENT, gradf);
                        fmat3tomat3(gradf, grad);

                        // compute M
                        mat3 M;
                        compute_M(velo, grad, S, gradScol0, gradScol1, gradScol2, M);

                        // store M
                        fmat3 Mf;
                        mat3tofmat3(M, Mf);
                        temp->setMatrix3(n, COMP_M, Mf);
                    }
                }
            }

            for (int n = 0; n < unst_out->nNodes; n++)
            {

                if (nodeDisabled[n])
                    continue;

                // get endpoint of trajectory
                int vertex = trajVertCnt[n] - 1;
                vec3 pos;
                unif_traj->getCoord(vertex, n, pos);

                // evaluate Mz at endpoint of current trajectory
                bool hyperbolic = Mz_positive_definite(us,
                                                       temp,
                                                       ABC_instead_unst,
                                                       pos,
                                                       strong_hyperbolicity);

                //if (hyperbolic > unst_out->getScalar(n, OUTCOMP_MZ)) {
                if (hyperbolic)
                {
                    double newHTime = unst_out->getScalar(n, OUTCOMP_MZ);
                    // ############## this is a HACK (TODO), could be less if integration stopped!
                    newHTime += integration_time / time_intervals;
                    unst_out->setScalar(n, OUTCOMP_MZ, newHTime);
                }
            }

            clock_t f_end = clock();
            if (Mz_seconds)
                *Mz_seconds += (f_end - f_start) / ((double)CLOCKS_PER_SEC);
        }
        printf("t=%d          \n",
               tI);
    }

    if (nodeTreated)
        delete[] nodeTreated;
    if (destroyTransient)
        unst->unsetTransientFile();
    if (temp)
        delete temp;
}

void Mz_impl(UniSys *us,
             Unstructured *unst,
             int compVelo,
             int unsteady,
             const char *velocity_file,
             int strong_hyperbolicity,
             bool ABC_instead_unst,
             float start_time,
             float integration_time,
             int time_intervals,
             int integ_steps_max,
             int forward,
             Unstructured *unst_out,
             int smoothing_range,
             int omit_boundary_cells,
             int grad_neigh_disabled,
             UniField *unif_traj)
{
    if (ABC_instead_unst)
    {
        unst->setVector3CB(ABCsteady_CB, false);
    }
    else
    {
        unst->unsetVector3CB();
    }

    // temporary data for map
    Unstructured *map = new Unstructured(unst_out, 3);

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

    // could save memory by only allocating this when necessary
    // ### could avoid it completely, it is only used for avoiding continuation
    //     of stopped trajectories
    int *trajVertCnt = new int[unst_out->nNodes];
    for (int n = 0; n < unst_out->nNodes; n++)
    {
        trajVertCnt[n] = 0;
    }

    {
        clock_t t_start = clock();

        printf("computation started at time:\n");
        system("date");

        double map_seconds = 0.0;
        double Mz_seconds = 0.0;

        computeMz(us,
                  unst,
                  //crop_ucd,
                  false,
                  integration_time,
                  time_intervals,
                  integ_steps_max,
                  forward,
                  strong_hyperbolicity,
                  ABC_instead_unst,
                  nodeDisabled, nodesDisabled,
                  grad_neigh_disabled,
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
                  &Mz_seconds);

        clock_t t_end = clock();
        double seconds = (t_end - t_start) / ((double)CLOCKS_PER_SEC);

        printf("\nstatistics ------------------------------------------------------------------\n\n");

        printf("computation finished at time:\n");
        system("date");

        printf("total computation took %g seconds\n", seconds);
        printf("map computation took %g seconds\n", map_seconds);
        printf("Mz computation took %g seconds\n", Mz_seconds);
        printf("\n\n");
    }

    delete[] nodeDisabled;
    if (trajVertCnt)
        delete[] trajVertCnt;
    delete map;
    //if (ucd_map) UCDstructure_free(ucd_map); // ########## TOCHECK
    //delete unst_out;
}
