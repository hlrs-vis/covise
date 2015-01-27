/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

void flow_topo_impl(UniSys *us, Unstructured *unst, int compVelo, int compWallDist,
                    UniField **unif,
                    bool divide_by_walldist, bool interior_critical_points,
                    bool boundary_critical_points,
                    bool generate_seeds, int seedsPerCircle,
                    double radius, double offset)
{ // unif has to be != NULL
    if (compVelo >= 0)
        unst->selectVectorNodeData(compVelo);
    if (compWallDist >= 0)
        unst->selectWallDistNodeData(compWallDist);

    // Divide velocity by walldist if nonzero
    unst->divideVelocityByWalldist = divide_by_walldist;

    // Extrapolate data to no-slip boundaries
    //unst->extrapolateToNoSlipBoundary = false;
    unst->extrapolateToNoSlipBoundary = true;

    // Compute interior critical points
    if (interior_critical_points)
        unst->computeCriticalPoints();

    // Compute boundary critical points
    if (boundary_critical_points)
    {
        unst->computeCriticalBoundaryPoints();

        // Concatenate
        for (int i = 0; i < (int)unst->criticalBoundaryPoints.size(); i++)
        {
            unst->criticalPoints.push_back(unst->criticalBoundaryPoints[i]);
        }
    }

    // Output critical points
    int nPoints = unst->criticalPoints.size();
    printf("%d critical points\n", nPoints);
    int dims[1];

    if (generate_seeds)
    {
        nPoints *= seedsPerCircle;
        if (offset != 0)
            nPoints *= 2; // generate two circles
    }
    dims[0] = nPoints;

#if 0 // to be deleted, still here to give an example
  if (*critical_points) AVSfield_free((AVSfield*)*critical_points);
  *critical_points = (AVSfield_float *)
    AVSdata_alloc("field 1D 1-vector irregular 3-space float", dims);
  if (*critical_points == NULL) { AVSerror("allocation failed"); return(1); }
  AVSfield_set_labels((AVSfield *)*critical_points, "wall distance", ":");

  float* x = (*critical_points)->points;
  float* y = x + nPoints;
  float* z = y + nPoints;
  float* s = (*critical_points)->data;
#else

    (*unif)->freeField();
    int compVecLens[1] = { 1 };
    const char *compNames[1] = { "critical points" };
    if ((*unif)->allocField(1 /*ndims*/, dims, 3 /*nspace*/,
                            false /*regular*/, 1, compVecLens, compNames, UniField::DT_FLOAT) == false)
    {
        us->error("allocation failed");
        //return(1);
        return; // ###
    }
    //todo###: AVSfield_set_labels((AVSfield *)*critical_points, "wall distance", ":");

    int vertIdx = 0;
#endif

    if (generate_seeds)
    {
        for (int i = 0; i < (int)unst->criticalPoints.size(); i++)
        {
            CriticalPoint *cp = &(unst->criticalPoints[i]);
            //printf("cell: %d\n", cp->cell);
            vec3 axis, normal;
            vec3nrm(cp->eigenvector[0], axis);
            if (cp->allReal)
            {
                vec3cross(cp->eigenvector[1], cp->eigenvector[2], normal);
                vec3nrm(normal, normal);
            }
            else
            {
                vec3nrm(cp->eigenvector[1], normal);
            }

            // Compute center
            vec3 center;
            vec3scal(axis, offset, axis);
            vec3add(cp->coord, axis, center);

            // Compute two basis vectors a,b in rotating plane
            vec3 a, b;
#if 1
            vec3ortho(normal, b);
            vec3cross(b, normal, a);
#else
            if (cp->allReal)
            {
                // #### this is a HACK, a and b are not necessarily orthogonal !! ################
                vec3nrm(cp->eigenvector[1], a);
                vec3nrm(cp->eigenvector[2], b);
            }
            else
            {
                vec3ortho(normal, b);
                vec3cross(b, normal, a);
            }
#endif

            for (int j = 0; j < seedsPerCircle; j++)
            {
                float cosp = cos((float)j * 2. * M_PI / seedsPerCircle);
                float sinp = sin((float)j * 2. * M_PI / seedsPerCircle);

                vec3 pos;
                pos[0] = center[0] + radius * (cosp * a[0] + sinp * b[0]);
                pos[1] = center[1] + radius * (cosp * a[1] + sinp * b[1]);
                pos[2] = center[2] + radius * (cosp * a[2] + sinp * b[2]);
                (*unif)->setCoord(vertIdx, pos);

                (*unif)->setScalar(vertIdx, cp->wallDist);
                vertIdx++;
            }

            if (offset == 0)
                continue;

            vec3sub(center, axis, center);
            vec3sub(center, axis, center);

            for (int j = 0; j < seedsPerCircle; j++)
            {
                float cosp = cos((float)j * 2. * M_PI / seedsPerCircle);
                float sinp = sin((float)j * 2. * M_PI / seedsPerCircle);

                vec3 pos;
                pos[0] = center[0] + radius * (cosp * a[0] + sinp * b[0]);
                pos[1] = center[1] + radius * (cosp * a[1] + sinp * b[1]);
                pos[2] = center[2] + radius * (cosp * a[2] + sinp * b[2]);
                (*unif)->setCoord(vertIdx, pos);

                (*unif)->setScalar(vertIdx, cp->wallDist);
                vertIdx++;
            }
        }
    }
    else
    {
        for (int i = 0; i < (int)unst->criticalPoints.size(); i++)
        {
            CriticalPoint *cp = &(unst->criticalPoints[i]);

            vec3 pos;
            pos[0] = cp->coord[0];
            pos[1] = cp->coord[1];
            pos[2] = cp->coord[2];
            (*unif)->setCoord(vertIdx, pos);

            (*unif)->setScalar(vertIdx, cp->wallDist);
            vertIdx++;
        }
    }

    unst->deleteCriticalPoints();
    unst->deleteCriticalBoundaryPoints();
}
