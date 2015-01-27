/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define SUPPORT_COLORMAP 0 // ### TODO

#define PORTING 1 // REMOVEME

#include <climits>
#include <cfloat>

#include "unigeom.h"
#include "ridge_surface_lib.cpp" // ### including .cpp

bool ridge_surface_impl(UniSys *us,
                        Unstructured *unst, int compScalar, int compClipScalar,
#if SUPPORT_COLORMAP
                        Unstructured *unst2,
#endif
                        Unstructured **temp,
                        //int compGradient, int compHess, int compEigenvals, int compEigenvectExtr,
                        bool **excludeNodes,
                        float level,
                        int smoothing_range,
                        int modeNr,
                        int extremumNr,
                        bool useBisection,
                        int exclude_FLT_MAX,
                        int exclude_lonely_nodes,
                        float Hess_extr_eigenval_min,
                        float PCA_subdom_maxperc,
                        float scalar_min,
                        float scalar_max,
                        float clip_scalar_min,
                        float clip_scalar_max,
                        int min_size,
                        int filter_by_cell,
                        int combine_exceptions,
                        int max_exceptions,
                        float clip_min_x,
                        float clip_max_x,
                        float clip_min_y,
                        float clip_max_y,
                        float clip_min_z,
                        float clip_max_z,
                        int clip_lower_data,
                        int clip_higher_data,
                        int generate_normals,
                        UniGeom *ugeom)
{
    clock_t t_start = clock();

    //#include "ucd_ridge_surface.h"

    //static int nvertices[8] = {
    //  1, 2, 3, 4, 4, 5, 6, 8
    //};
    /* UCD_POINT, UCD_LINE, UCD_TRIANGLE, UCD_QUADRILATERAL,
     UCD_TETRAHEDRON, UCD_PYRAMID, UCD_PRISM, UCD_HEXAHEDRON */

    //static int edge[12][2] = { 	// hex
    //  {0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}
    //};
    //static int tedge[6][2] = {  // tet
    //  {0,1},{0,2},{0,3},{1,2},{2,3},{3,1}
    //};

    //int i, j;
    int count_hex, count_tet, count_other, count_marked, count_hit, count_tria;
    Nodeinfo *nodeinfo;
#if SUPPORT_COLORMA
    char *choice_list;
    char label[100][MAXLABELSIZE];
    float *data_start[100]; /* Pointers to each component */
    float *color_data, *surface_data;
    char str[MAXLABELSIZE];
    int s;
    Boolean valid;
    float RGB[MAXCOLORS][3];
    int ncomponents;
#endif
    static bool *clip = NULL; // NEW

#if !EXTERNAL_DATA
    // temporary Unstructured data:
    // gradient, Hessian, eigenvalues, eigenvector of smallest eigenvalue
    int compGradient = 0, compHess = 1, compEigenvals = 2, compEigenvectExtr = 3;
    //Unstructured *temp = NULL;
    //UCD_structure *ucd_temp = NULL;
    if (us->inputChanged("ucd", 0) || us->parameterChanged("smoothing range") || us->parameterChanged("exclude FLT_MAX") || us->parameterChanged("exclude lonely nds"))
    {

        if (*temp)
            delete *temp;
        //if (ucd_temp) UCDstructure_free(ucd_temp);

        int components[4] = { 3, 3 * 3, 3, 3 };
#if 1 // ### TODO: there is a BUG in Unstructured here
        *temp = new Unstructured(unst, 4, components);
#else
        char labels[256] = "gradient.Hessian.eigenvalues.extremum eigenvector";
        ucd_temp = ucdClone(ucd, 4, components, "ucd out", labels, ".");
        temp = new Unstructured(ucd_temp);
#endif

        if (*excludeNodes)
            delete[] * excludeNodes;
        *excludeNodes = NULL;
        if (exclude_FLT_MAX)
        {
            *excludeNodes = new bool[unst->nNodes];
            if (!*excludeNodes)
            {
                printf("out of memory\n");
                return 0;
            }
            int excluded = 0;
            for (int n = 0; n < unst->nNodes; n++)
            {
                if (unst->getScalar(n, compScalar) == FLT_MAX)
                {
                    (*excludeNodes)[n] = true;
                    excluded++;
                }
                else
                {
                    (*excludeNodes)[n] = false;
                }
            }
            printf("excluded %d nodes due to FLT_MAX\n", excluded);
        }

        std::vector<int> lonelyNodes;

        // compute gradient
        // #### TODO: range parameter
        //AVSmodify_parameter("status", AVS_VALUE, "Computing gradient ...", 0, 0);
        us->info("Computing gradient ...");
        //unst->gradient(compScalar, temp, compGradient, 2);
        //unst->gradient(compScalar, temp, compGradient, smoothing_range, excludeNodes);
        {
            float defaultGradS[3] = { 0.0, 0.0, 0.0 };
            unst->gradient(compScalar, *temp, compGradient, smoothing_range, *excludeNodes, defaultGradS, &lonelyNodes);
            printf("got %d lonely nodes from gradient computation\n", (int)lonelyNodes.size());
        }

        // compute Hessian
        // #### TODO: range parameter
        //AVSmodify_parameter("status", AVS_VALUE, "Computing Hessian ...", 0, 0);
        us->info("Computing Hessian ...");
        //temp->gradient(compGradient, temp, compHess, 2);
        //temp->gradient(compGradient, temp, compHess, smoothing_range, excludeNodes);
        if (*excludeNodes && exclude_lonely_nodes)
        {
            for (int n = 0; n < (int)lonelyNodes.size(); n++)
            {
                (*excludeNodes)[lonelyNodes[n]] = true;
            }
        }
        {
            float defaultGradV[3 * 3] = {
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            };
            std::vector<int> lonelyNodes2;
            (*temp)->gradient(compGradient, *temp, compHess, smoothing_range, *excludeNodes, defaultGradV, &lonelyNodes2, true /*force symmetric*/);
            printf("got %d lonely nodes from 2. gradient (Hessian) computation\n", (int)lonelyNodes2.size());

            if (*excludeNodes && exclude_lonely_nodes)
            {
                for (int n = 0; n < (int)lonelyNodes2.size(); n++)
                {
                    (*excludeNodes)[lonelyNodes2[n]] = true;
                }
            }
        }

        // compute eigenvalues of Hessian, in descending order
        //AVSmodify_parameter("status", AVS_VALUE, "Computing eigenvalues ...", 0, 0);
        us->info("Computing eigenvalues ...");
        (*temp)->realEigenvaluesSortedDesc(compHess, false, *temp, compEigenvals);

        // compute eigenvector corresponding to smallest(max)/largest(min) eigenvalue
        //AVSmodify_parameter("status", AVS_VALUE, "Computing eigenvectors ...", 0, 0);
        us->info("Computing eigenvectors ...");
        (*temp)->realEigenvectorSortedDesc(compHess, false,
                                           //(strcmp(extremum, EX_MIN) == 0 ? 0 : 2),
                                           (extremumNr == EXNR_MIN ? 0 : 2),
                                           *temp, compEigenvectExtr);

        // ##########################################################
        //if (excludeNodes) delete [] excludeNodes;
        //excludeNodes = NULL;
    }
#endif

    nodeinfo = (Nodeinfo *)calloc(unst->nNodes, sizeof(Nodeinfo));

    //GEOMobj* obj = GEOMcreate_obj(GEOM_POLYHEDRON, NULL);
    ugeom->createObj(UniGeom::GT_POLYHEDRON);

    count_tet = count_hex = count_other = count_marked = count_hit = count_tria = 0;
    int last_vertex_nr = 0; /* start at 1 */

    /* Mark nodes adjacent to intersected edges. */
    //DELETEME int* node_list = ucd->node_list;
    markNodesAtIntersectedEdges(us,
                                //ucd,
                                NULL,
                                *excludeNodes,
                                unst,
                                modeNr,
                                extremumNr,
                                filter_by_cell,
                                compScalar,
                                *temp,
                                compGradient,
                                compHess,
                                compEigenvals,
                                compEigenvectExtr,
                                useBisection,
                                level,
                                Hess_extr_eigenval_min,
                                PCA_subdom_maxperc,
                                scalar_min,
                                NULL, // scalar_min is same for all cells
                                scalar_max,
                                compClipScalar,
                                clip_scalar_min,
                                clip_scalar_max,
                                combine_exceptions,
                                //DELETEME node_list,
                                nodeinfo,
                                count_tet,
                                count_hex,
                                count_other);

#if NORMALS_FROM_GRAD

    /* Collect neighbors of marked nodes */
    // TODO:
    //AVSmodify_parameter("status", AVS_VALUE, "Collecting neighbors ...", 0, 0);
    us->moduleStatus("collecting neighbors", 30);
    node_list = ucd->node_list;
    for (i = 0; i < ucd->ncells; i++)
    {
        int type;
        int nodenr[8];

        type = ucd->cell_type[i];
        if (type == UCD_HEXAHEDRON)
        {
            nodenr[4] = *node_list++;
            nodenr[5] = *node_list++;
            nodenr[7] = *node_list++;
            nodenr[6] = *node_list++;
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[3] = *node_list++;
            nodenr[2] = *node_list++;

            for (j = 0; j < 12; j++)
            {
                Nodeinfo *n;
                int n0, n1;
                int m;
                Boolean exists;
                int k;

                n0 = nodenr[edge[j][0]];
                n1 = nodenr[edge[j][1]];

                n = &(nodeinfo[n0]);
                if (n->marked)
                {
                    exists = FALSE;
                    for (k = 0; k < n->nedges; k++)
                    {
                        if (n->edge[k] == n1)
                        {
                            exists = TRUE;
                            break;
                        }
                    }
                    if (!exists)
                    {
                        m = ++n->nedges;
                        n->edge = (int *)realloc(n->edge, m * sizeof(int));
                        n->edge[m - 1] = n1;
                    }
                }
                n = &(nodeinfo[n1]);
                if (n->marked)
                {
                    exists = FALSE;
                    for (k = 0; k < n->nedges; k++)
                    {
                        if (n->edge[k] == n0)
                        {
                            exists = TRUE;
                            break;
                        }
                    }
                    if (!exists)
                    {
                        m = ++n->nedges;
                        n->edge = (int *)realloc(n->edge, m * sizeof(int));
                        n->edge[m - 1] = n0;
                    }
                }
            }
        }
        else if (type == UCD_TETRAHEDRON)
        {
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[2] = *node_list++;
            nodenr[3] = *node_list++;

            for (j = 0; j < 6; j++)
            {
                Nodeinfo *n;
                int n0, n1;
                int m;
                Boolean exists;
                int k;

                n0 = nodenr[tedge[j][0]];
                n1 = nodenr[tedge[j][1]];

                n = &(nodeinfo[n0]);
                if (n->marked)
                {
                    exists = FALSE;
                    for (k = 0; k < n->nedges; k++)
                    {
                        if (n->edge[k] == n1)
                        {
                            exists = TRUE;
                            break;
                        }
                    }
                    if (!exists)
                    {
                        m = ++n->nedges;
                        n->edge = (int *)realloc(n->edge, m * sizeof(int));
                        n->edge[m - 1] = n1;
                    }
                }
                n = &(nodeinfo[n1]);
                if (n->marked)
                {
                    exists = FALSE;
                    for (k = 0; k < n->nedges; k++)
                    {
                        if (n->edge[k] == n0)
                        {
                            exists = TRUE;
                            break;
                        }
                    }
                    if (!exists)
                    {
                        m = ++n->nedges;
                        n->edge = (int *)realloc(n->edge, m * sizeof(int));
                        n->edge[m - 1] = n0;
                    }
                }
            }
        }
        else
        {
            node_list += nvertices[type];
        }
    }

    /* Compute and store gradient for marked nodes */
    // TODO:
    //AVSmodify_parameter("status", AVS_VALUE, "Computing gradients ...", 0, 0);
    us->moduleStatus("computing gradients", 60);
    for (i = 0; i < ucd->nnodes; i++)
    {
        Nodeinfo *n;
        float xx, yy, zz, xy, xz, yz, sx, sy, sz;
        float s0;
        fvec3 x0;
        int k;
        float d;

        n = &(nodeinfo[i]);
        if (!n->marked)
            continue;

        s0 = surface_data[i];
        get_coord(ucd, i, x0);

        xx = yy = zz = xy = xz = yz = sx = sy = sz = 0;

        for (j = 0; j < n->nedges; j++)
        {
            int k;
            float s;
            fvec3 x;
            float dx, dy, dz, ds;

            k = n->edge[j];
            s = surface_data[k];
            get_coord(ucd, k, x);
            ds = s - s0;
            dx = x[0] - x0[0];
            dy = x[1] - x0[1];
            dz = x[2] - x0[2];

            xx += dx * dx;
            yy += dy * dy;
            zz += dz * dz;
            xy += dx * dy;
            xz += dx * dz;
            yz += dy * dz;
            sx += ds * dx;
            sy += ds * dy;
            sz += ds * dz;
        }
        d = det3(xx, xy, xz, xy, yy, yz, xz, yz, zz);
        if (d == 0)
            d = 1e-19;
        n->grad[0] = det3(sx, sy, sz, xy, yy, yz, xz, yz, zz) / d;
        n->grad[1] = det3(xx, xy, xz, sx, sy, sz, xz, yz, zz) / d;
        n->grad[2] = det3(xx, xy, xz, xy, yy, yz, sx, sy, sz) / d;
    }

#endif

    /* Create vertices and normals */
    createVerticesAndNormals(us, // ucd,
                             unst,
                             *temp, compGradient, compHess, compEigenvectExtr, useBisection, extremumNr, level, nodeinfo, clip_lower_data, clip_higher_data,
#if SUPPORT_COLORMAP
                             cmap, RGB, lower, upper, color_data,
#endif
                             //obj,
                             ugeom,
                             last_vertex_nr, &clip);

    /* Generate triangles */
    std::vector<int> triangleConn;
    int falsePositiveIntersections = 0;
    int falseNegativeIntersections = 0;
    if (generateTriangles(us,
                          //ucd,
                          *excludeNodes,
                          unst,
                          -1,
                          modeNr,
                          extremumNr,
                          filter_by_cell,
                          compScalar,
                          *temp,
                          compGradient,
                          compHess,
                          compEigenvals,
                          compEigenvectExtr,
                          level,
                          scalar_min,
                          NULL, // scalar_min is same for all cells
                          scalar_max,
                          compClipScalar,
                          clip_scalar_min,
                          clip_scalar_max,
                          Hess_extr_eigenval_min,
                          combine_exceptions,
                          max_exceptions,
                          clip_min_x,
                          clip_max_x,
                          clip_min_y,
                          clip_max_y,
                          clip_min_z,
                          clip_max_z,
                          //node_list,
                          nodeinfo,
                          clip,
                          count_hit,
                          count_tria,
                          falsePositiveIntersections,
                          falseNegativeIntersections,
                          //obj,
                          ugeom,
                          &triangleConn) < 0)
    {
        return false;
    }

    // post process
    printf("%d false positive and %d false negative intersections\n",
           falsePositiveIntersections, falseNegativeIntersections);

    {
        char stat[512];

        //for (int i = 0; i < ucd->nnodes; i++) {
        for (int i = 0; i < unst->nNodes; i++)
        {
            if (nodeinfo[i].marked)
                count_marked++;
        }

        sprintf(stat,
                "%d hex, %d tet, %d other (ignored)\n%d triangles generated\n",
                count_hex, count_tet, count_other, count_tria);
        // TODO:
        //AVSmodify_parameter("status", AVS_VALUE, stat, 0, 0);
        us->info(stat);
    }

    //for (int i = 0; i < ucd->nnodes; i++) {
    for (int i = 0; i < unst->nNodes; i++)
    {
        if (nodeinfo[i].niedges > 0)
            free(nodeinfo[i].iedge);
        if (nodeinfo[i].nedges > 0)
            free(nodeinfo[i].edge);
    }
    free(nodeinfo);

#if COMPUTE_RIDGE
    //AVSmodify_parameter("status", AVS_VALUE, "Post-processing mesh ...", 0, 0);
    // make triangle winding consistent
    std::vector<int> triangleComponents;
    std::vector<int> triangleComponentSizes;
    makeTrianglesConsistent(&triangleConn, count_tria,
                            &triangleComponents, &triangleComponentSizes);

    // generate object triangles
    for (int t = 0; t < count_tria; t++)
    {
#if 0 // all components
    int indices[3] = { triangleConn[t*3+0], triangleConn[t*3+1], triangleConn[t*3+2] };
    GEOMadd_polygon(obj, 3, indices, 0, GEOM_COPY_DATA);
#else // HACK for component filtering ##### TODO (still using all vertices)
        if (triangleComponentSizes[triangleComponents[t]] >= min_size)
        {
            // UniGeom uses 0-based indices
            int indices[3] = { triangleConn[t * 3 + 0] - 1,
                               triangleConn[t * 3 + 1] - 1,
                               triangleConn[t * 3 + 2] - 1 };
            //GEOMadd_polygon(obj, 3, indices, 0, GEOM_COPY_DATA);
            ugeom->addPolygon(3, indices);

            // #### TODO: maybe use GEOMadd_disjoint_polygon with GEOM_NOT_SHARED (see AVS developer doc)
            // would also need to define normals
        }
#endif
    }

    // count components after filtering
    int finalCompCnt = 0;
    for (int c = 0; c < (int)triangleComponentSizes.size(); c++)
    {
        if (triangleComponentSizes[c] >= min_size)
        {
            finalCompCnt++;
        }
    }
    printf("%d components after filtering\n", finalCompCnt);

#endif

#if !NORMALS_FROM_GRAD
    //if (count_tria > 0) GEOMgen_normals(obj, 0);
    if ((count_tria > 0) && generate_normals)
        ugeom->generateNormals();
#endif

    // DELETEME
    //*ridge = GEOMinit_edit_list(*ridge);
    //GEOMedit_geometry(*ridge, "ridge", obj);
    //GEOMdestroy_obj(obj);
    ugeom->assignObj("ridge");

#if !EXTERNAL_DATA
//delete temp;
#if 0 // #######
        //if (ucd_temp) UCDstructure_free(ucd_temp);
#else
//DELETEME *debug = ucd_temp;
#endif
#endif

    clock_t t_end = clock();
    double seconds = (t_end - t_start) / ((double)CLOCKS_PER_SEC);
    printf("computation took %g seconds\n", seconds);

    return true;
}
