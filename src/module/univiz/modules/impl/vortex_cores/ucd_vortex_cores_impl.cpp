/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

void ucd_vortex_cores_impl(UniSys *us, Unstructured *unst, int compV,
                           int methodNr, int variantNr,
                           int min_vertices, int max_exceptions,
                           float min_strength, float max_angle,
                           UniGeom *ug)
{
    static UCD_connectivity *ucdc = NULL;
    static float *gradient = NULL;
    static VertexList *vertexList = NULL;
    static PolylineList *polylineList = NULL;

    //int i;
    static int oldNNodes = 0;
    static int oldNCells = 0;
    static float extent;

    // decide if compV has changed
    static int lastCompV;
    bool compVChanged;
    if (us->inputChanged("ucd", 0))
    {
        compVChanged = true;
        lastCompV = compV;
    }
    else
    {
        if (compV != lastCompV)
        {
            compVChanged = true;
            lastCompV = compV;
        }
        else
        {
            compVChanged = false;
        }
    }

    /* Recompute ucd connectivity if grid has changed */
    /* Assume that the grid has changed iff nnodes or ncells has changed */
    //if ((unst->nNodes != oldNNodes) || (unst->nCells != oldNCells)) {
    // #### now always computing connectivity because got crashes with
    // multi-set datasets in Covise and ParaView, 2007-08-16
    // #### TODO: find a solution
    if (true)
    {
        if (ucdc)
            deleteUcdConnectivity(ucdc);

        us->moduleStatus("Computing connectivity", 30);
        ucdc = computeConnectivity(unst);
        oldNNodes = unst->nNodes;
        oldNCells = unst->nCells;
    }

    /* Compute extent */
    if (us->inputChanged("ucd", 0))
    {
        extent = computeExtent(unst);
    }

    /* Compute gradients */
    if (us->inputChanged("ucd", 0) || compVChanged)
    {
        if (gradient)
            free(gradient);
        us->moduleStatus("Computing gradients", 40);
        gradient = computeGradient(unst, compV, ucdc);
    }

    /* Compute core lines */
    if (us->inputChanged("ucd", 0) || compVChanged || us->parameterChanged("method") || us->parameterChanged("variant"))
    {

        /* Compute "w" field (either vorticity or acceleration) */
        float *wField;

        //if (method[0] == 'L') {                   /* Levy */
        if (methodNr == 1)
        { /* Levy */
            us->moduleStatus("Computing vorticities", 50);
            wField = computeVorticity(gradient, unst->nNodes);
        }
        else
        {
            us->moduleStatus("Computing accelerations", 50);
            wField = computeAcceleration(gradient, unst, compV);
        }

        /* Free old vertex list and polyline list */
        if (vertexList)
            deleteVertexList(vertexList);
        if (polylineList)
            deletePolylineList(polylineList);

        /* Compute core lines for all cells */
        vertexList = findParallel(us, unst, ucdc, compV,
                                  (fvec3 *)wField, (fmat3 *)gradient, variantNr, extent);

        /* Delete the "w" field */
        free(wField);

        /* Fix the order of link1 and link2 */
        polylineList = generatePolylines(vertexList);

        printf("\n");
        printf("nPolylines = %d\n", polylineList->nElems);

        /* Compute the feature quality for each vertex */
        computeFeatureQuality(vertexList);
    }

    /* Generate the output geometry */

    generateOutputGeometry(us, vertexList, polylineList,
                           min_vertices, max_exceptions, min_strength, max_angle, ug);
}
