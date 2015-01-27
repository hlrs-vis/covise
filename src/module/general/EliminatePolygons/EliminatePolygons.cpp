/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Eliminates polygons of poly_away out of poly1               **
 **                                                                          **
 ** Name:        EliminatePolygons                                           **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: Sven Kufer		                                            **
 **                                                                          **
 ** History:  								    **
 ** April-00     					       		    **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#define FOR_ALL 1
#define PER_TIMESTEP 2
#define PER_BLOCK_AND_P_TIMESTEP 3

#include "EliminatePolygons.h"
#include <util/coviseCompat.h>
#include <float.h>
#include <do/coDoSet.h>

EliminatePolygons::EliminatePolygons(int argc, char *argv[])
    : coModule(argc, argv, "Eliminate a subset of a Polygon")
{
    outPort_polySet = addOutputPort("GridOut0", "Polygons", "polygons or set of polygons");
    p_grid1 = addInputPort("GridIn0", "Polygons", "Polygons");
    p_grid2 = addInputPort("GridIn1", "Polygons", "Polygons to throw away");
    //
    // this is not a module on its own any more

    fields_are_set = 0;
    disc_num = 100;
}

void EliminatePolygons::Destruct()
{
    if (!fields_are_set)
        return;
    fields_are_set = 0;
    poly_now++;
    for (int t = 0; t < disc_num; t++)
    {
        delete[] sorted_polygons[t];
    }
    delete[] sorted_polygons;
    delete[] num_sorted;
    delete[] ypoly_min;
    delete[] xpoly_min;
    delete[] xpoly_max;
}

//
// put polygons to throw away into PolygonList
//

const coDistributedObject *EliminatePolygons::handle_poly(const coDistributedObject *obj_in)
{
    const coDistributedObject *const *objs;
    const coDistributedObject *ret;
    int numsets, i;

    if (const coDoSet *in_set = dynamic_cast<const coDoSet *>(obj_in))
    {
        if (elimMode == PER_TIMESTEP)
            elimMode = PER_BLOCK_AND_P_TIMESTEP;
        else if (in_set->getAttribute("TIMESTEP") != NULL)
            elimMode = PER_TIMESTEP;

        objs = in_set->getAllElements(&numsets);
        for (i = 0; i < numsets; i++)
        {
            ret = handle_poly(objs[i]);
            if (ret != NULL)
                polygonList[num_in_polygons++] = (coDoPolygons *)ret;
        }
    }

    if (dynamic_cast<const coDoPolygons *>(obj_in))
    {
        if (elimMode == FOR_ALL)
            polygonList[num_in_polygons++] = (const coDoPolygons *)obj_in;
        else
        {
            return obj_in;
        }
    }
    return NULL;
}

int EliminatePolygons::compute(const char *)
{
    outPort_polySet->setCurrentObject(eliminate(p_grid1->getCurrentObject(), p_grid2->getCurrentObject(), outPort_polySet->getObjName()));
    return 0;
}

//
// core compute function
//

coDoPolygons *EliminatePolygons::eliminate(const coDistributedObject *poly1, const coDistributedObject *poly_away,
                                           const char *outName)
{
    coDistributedObject *ret;

    elimMode = FOR_ALL;
    poly_now = 0;
    if (poly_away != NULL)
    {
        num_in_polygons = 0;
        handle_poly(poly_away); // put polygons to throw away into PolygonList
        if (poly1 != NULL)
        {
            ret = handle_port1(poly1, outName); // compute elimination recursively
            Destruct();
            return (coDoPolygons *)ret;
        }
    }
    return NULL;
}

//
// splits up sets if necessary and
// uses function eliminate_polygons for every polygon of obj_in and the actual polygon
//  PolygonList[poly_now]
//

coDistributedObject *EliminatePolygons::handle_port1(const coDistributedObject *obj_in, const char *obj_name)
{
    const coDistributedObject *const *objs;
    coDistributedObject **poly_out, *poly3;
    coDoSet *out_set;
    const coDoPolygons *poly1;
    int numsets, num_attrib;
    const char **attr_names, **attr_values;
    char my_obj_name[100];

    if (const coDoSet *in_set = dynamic_cast<const coDoSet *>(obj_in))
    {
        objs = in_set->getAllElements(&numsets);
        poly_out = new coDistributedObject *[numsets + 1];
        for (int i = 0; i < numsets; i++)
        {
            if (in_set->getAttribute("TIMESTEP") != NULL && elimMode != FOR_ALL && fields_are_set)
                Destruct();
            sprintf(my_obj_name, "%s_%d", obj_name, i + 1);
            poly_out[i] = (coDistributedObject *)handle_port1(objs[i], my_obj_name);
        }
        poly_out[numsets] = NULL;
        out_set = new coDoSet(obj_name, poly_out);
        num_attrib = in_set->getAllAttributes(&attr_names, &attr_values);
        out_set->addAttributes(num_attrib, attr_names, attr_values);

        return out_set;
        delete poly_out;
    }

    else if (dynamic_cast<const coDoPolygons *>(obj_in))
    {
        if (elimMode == PER_BLOCK_AND_P_TIMESTEP && fields_are_set)
            Destruct();
        if (poly_now < num_in_polygons)
        {
            poly1 = (const coDoPolygons *)obj_in;
            poly3 = (coDistributedObject *)eliminate_poly(poly1, polygonList[poly_now], obj_name);

            // sl: For stationary data the attributes may also be necessary!!!
            num_attrib = obj_in->getAllAttributes(&attr_names, &attr_values);
            poly3->addAttributes(num_attrib, attr_names, attr_values);
            return poly3;
        }
    }
    return NULL;
}

//
// calculates poly1 - poly2 ( checks only for exact matches)
// and returns a polygon with the object name obj_name
//

coDoPolygons *EliminatePolygons::eliminate_poly(const coDoPolygons *poly1, const coDoPolygons *poly2, char const *obj_name)
{
    int *pl1, *cl1;
    float *x_coords1, *y_coords1, *z_coords1;
    int num_points1, num_corners1, num_polygons1;

    int *pl2, *cl2;
    float *x_coords2, *y_coords2, *z_coords2;
    int num_corners2, num_polygons2;

    coDoPolygons *poly_out;
    int *pl_out, *cl_out;
    // float *x_coords_out, *y_coords_out, *z_coords_out;
    int num_corners_out = 0, num_polygons_out = 0;

    num_points1 = poly1->getNumPoints();
    num_corners1 = poly1->getNumVertices();
    num_polygons1 = poly1->getNumPolygons();
    poly1->getAddresses(&x_coords1, &y_coords1, &z_coords1, &cl1, &pl1);

    num_corners2 = poly2->getNumVertices();
    num_polygons2 = poly2->getNumPolygons();
    poly2->getAddresses(&x_coords2, &y_coords2, &z_coords2, &cl2, &pl2);

    pl_out = new int[num_polygons1];
    cl_out = new int[num_corners1];

    int t, s;
    int next_pol;
    int sort_nr;

    if (fields_are_set == 0)
    {
        //
        // sort polygons by x-coordinates
        //

        xmin = FLT_MAX;
        xmax = -FLT_MAX;

        xpoly_min = new float[num_polygons2];
        xpoly_max = new float[num_polygons2];
        ypoly_min = new float[num_polygons2];

        if (xpoly_min == NULL || xpoly_max == NULL || ypoly_min == NULL)
        {
            cerr << "memory fault " << endl;
            return 0;
        }

        for (t = 0; t < num_polygons2; t++)
        {
            next_pol = (t == num_polygons2 - 1) ? num_corners2 : pl2[t + 1];
            xpoly_min[t] = FLT_MAX;
            xpoly_max[t] = -FLT_MAX;
            ypoly_min[t] = FLT_MAX;

            for (s = pl2[t]; s < next_pol; s++)
            {
                if (x_coords2[cl2[s]] < xpoly_min[t])
                    xpoly_min[t] = x_coords2[cl2[s]];
                if (x_coords2[cl2[s]] > xpoly_max[t])
                    xpoly_max[t] = x_coords2[cl2[s]];
                if (y_coords2[cl2[s]] < ypoly_min[t])
                    ypoly_min[t] = y_coords2[cl2[s]];
            }
            if (xpoly_min[t] < xmin)
                xmin = xpoly_min[t];
            if (xpoly_max[t] > xmax)
                xmax = xpoly_max[t];
        }

        xstep = (xmax - xmin) / (disc_num - 1);
        sorted_polygons = new int *[disc_num];
        num_sorted = new int[disc_num];

        if (sorted_polygons == NULL || num_sorted == NULL)
        {
            cerr << "memory fault";
            return 0;
        }
        for (t = 0; t < disc_num; t++)
        {
            sorted_polygons[t] = new int[num_polygons2];
            if (sorted_polygons[t] == NULL)
            {
                cerr << "memory fault2" << endl;
                return 0;
            }
            num_sorted[t] = 0;
        }

        // sort now

        for (t = 0; t < num_polygons2; t++)
        {
            sort_nr = (int)((xpoly_min[t] - xmin) / xstep);
            sorted_polygons[sort_nr][num_sorted[sort_nr]++] = t;
            if (num_sorted[sort_nr] >= num_polygons2)
                cerr << "error here" << endl;
        }

        fields_are_set = 1;
        //for( t=0; t<disc_num; t++ )
        // {
        // cerr << xmin+t* xstep <<" : ";
        // for( s=0; s<num_sorted[t]; s++ )
        //  cerr << sorted_polygons[t][s] << " ";
        // cerr << endl;
        // }
    }

    //
    // now do the computation
    //
    float eps = FLT_EPSILON;
    int i, j, next_poly1, r, next_poly2;

    bool is_in, corner_is_in;
    float xthispoly_min, xthispoly_max, ythispoly_min;

    for (i = 0; i < num_polygons1; i++)
    {
        is_in = false;
        //
        // find xmin of polygon i
        //
        next_pol = (i == num_polygons1 - 1) ? num_corners1 : pl1[i + 1];
        xthispoly_min = FLT_MAX;
        xthispoly_max = -FLT_MAX;
        ythispoly_min = FLT_MAX;
        for (s = pl1[i]; s < next_pol; s++)
        {
            if (x_coords1[cl1[s]] < xthispoly_min)
                xthispoly_min = x_coords1[cl1[s]];
            if (x_coords1[cl1[s]] > xthispoly_max)
                xthispoly_max = x_coords1[cl1[s]];
            if (y_coords1[cl1[s]] < ythispoly_min)
                ythispoly_min = y_coords1[cl1[s]];
        }

        sort_nr = (int)((xthispoly_min - xmin) / xstep);
        if (sort_nr >= 0 && sort_nr < disc_num)
        {
            for (t = 0; t < num_sorted[sort_nr] && !is_in; t++)
            {
                if (t >= num_polygons2)
                    cerr << "error" << endl;
                r = sorted_polygons[sort_nr][t];
                if (ythispoly_min == ypoly_min[r] && xthispoly_min == xpoly_min[r] && xthispoly_max == xpoly_max[r])
                {
                    next_poly2 = (r == num_polygons2 - 1) ? num_corners2 : pl2[r + 1];
                    corner_is_in = true;
                    for (s = pl2[r]; s < next_poly2 && corner_is_in; s++)
                    {
                        next_poly1 = (i == num_polygons1 - 1) ? num_corners1 : pl1[i + 1];
                        corner_is_in = false;
                        for (j = pl1[i]; j < next_poly1; j++)
                        {
                            if (fabs(x_coords1[cl1[j]] - x_coords2[cl2[s]]) <= eps && fabs(y_coords1[cl1[j]] - y_coords2[cl2[s]]) <= eps && fabs(z_coords1[cl1[j]] - z_coords2[cl2[s]]) <= eps)
                                corner_is_in = true;
                        }
                    }
                    is_in = corner_is_in;
                }
            }
        }

        if (!is_in)
        { // take polygon
            pl_out[num_polygons_out++] = num_corners_out;
            next_poly1 = (i == num_polygons1 - 1) ? num_corners1 : pl1[i + 1];
            for (j = pl1[i]; j < next_poly1; j++)
            {
                cl_out[num_corners_out++] = cl1[j];
            }
        }
    }

    poly_out = new coDoPolygons(obj_name, num_points1, x_coords1, y_coords1, z_coords1,
                                num_corners_out, cl_out, num_polygons_out, pl_out);
    if (!poly_out->objectOk())
        Covise::sendError("Error creating object");
    delete[] pl_out;
    delete[] cl_out;

    return (poly_out);
}

MODULE_MAIN(Filter, EliminatePolygons)
