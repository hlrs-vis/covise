/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "clip_interval.h"
#include <appl/ApplInterface.h>
#include <util/coviseCompat.h>
#include <do/coDoPolygons.h>

coDoPolygons *clip_interval::replicatePolygon(const char *out_poly_name)
{
    coDoPolygons *ret;
    int *rp_l, *rv_l;
    float *rx, *ry, *rz;

    ret = new coDoPolygons(out_poly_name, in_no_points, in_no_v, in_no_pol);
    ret->getAddresses(&rx, &ry, &rz, &rv_l, &rp_l);
    memcpy(rx, in_x, in_no_points * sizeof(float));
    memcpy(ry, in_y, in_no_points * sizeof(float));
    memcpy(rz, in_z, in_no_points * sizeof(float));
    memcpy(rv_l, in_vl, in_no_v * sizeof(int));
    memcpy(rp_l, in_pol_l, in_no_pol * sizeof(int));
    return ret;
}

coDoPoints *clip_interval::replicatePoint(const char *out_poly_name)
{
    coDoPoints *ret;
    float *rx, *ry, *rz;

    ret = new coDoPoints(out_poly_name, in_no_points);
    ret->getAddresses(&rx, &ry, &rz);
    memcpy(rx, in_x, in_no_points * sizeof(float));
    memcpy(ry, in_y, in_no_points * sizeof(float));
    memcpy(rz, in_z, in_no_points * sizeof(float));
    return ret;
}

clip_interval::clip_interval(const coDoPolygons *poly, const coDoFloat *data,
                             const coDistributedObject **data_map, int no_ports, int dummy, float min, float max)
    : no_ports_(no_ports)
    , data_map_(data_map)
    , upon_dummy_(dummy)
{
    poly->getAddresses(&in_x, &in_y, &in_z, &in_vl, &in_pol_l);
    in_no_v = poly->getNumVertices();
    //	cerr << "no vertices: " << in_no_v << endl;
    in_no_pol = poly->getNumPolygons();
    //	cerr << "no polygon: " << in_no_pol << endl;
    in_no_points = poly->getNumPoints();
    data->getAddress(&in_data);
    no_data_points = data->getNumPoints();
    per_polygon = 0;
    if (no_data_points != in_no_points)
        if (no_data_points == in_no_pol)
            per_polygon = 1;
    /*********************************************************
      else
      {
         cerr << "no_points differ ! data: " << no_data_points << " points:" << in_no_points << endl;
         cerr << "polys: " << in_no_pol << endl;
      }
   *********************************************************/
    min_value = min;
    max_value = max;
}

int clip_interval::do_clip(coDoPolygons **out_poly, const char *out_poly_name,
                           coDoFloat **out_data, const char *out_data_name,
                           coDistributedObject **out_map_data, const char **out_map_data_name)
{

    o_no_pol = 0;
    o_no_v = 0;
    o_no_points = 0;

    int *sel_polygons = new int[in_no_pol]; // used for mapped data
    memset(sel_polygons, '\0', in_no_pol * sizeof(int));

    int i, j;
    // dummy output if we have dummy data
    if (no_data_points == 0)
    {
        if (upon_dummy_)
            *out_poly = new coDoPolygons(out_poly_name, 0, 0, 0);
        else
            *out_poly = replicatePolygon(out_poly_name);

        *out_data = new coDoFloat(out_data_name, 0);
        int ret = do_mapped_data(sel_polygons, out_map_data, out_map_data_name);
        delete[] sel_polygons;
        return ret;
    }
    if (!per_polygon && no_data_points < in_no_points)
    {
        Covise::sendWarning("Less data points than polygon points");
        delete[] sel_polygons;
        return 1; // not enough data - bail out !
    }
    int *t_p_l = new int[in_no_pol];
    int *t_vl = new int[in_no_v];
    float *t_x = new float[in_no_points];
    float *t_y = new float[in_no_points];
    float *t_z = new float[in_no_points];
    float *t_data = new float[no_data_points];

    tags = new int[no_data_points];
    new_index = new int[in_no_points];
    // checks all data if it is inside the chosen interval and sets tags to 1 if yes else to 0
    float data_min = FLT_MAX, data_max = -FLT_MAX;
    for (i = 0; i < no_data_points; i++)
    {
        if (in_data[i] >= min_value && in_data[i] <= max_value)
            tags[i] = 1;
        else
            tags[i] = 0;
        if (in_data[i] <= data_min)
            data_min = in_data[i];
        if (in_data[i] >= data_max)
            data_max = in_data[i];
    }
    Covise::sendInfo("data_min: %f data_max: %f\n", data_min, data_max);
    for (i = 0; i < in_no_points; i++)
        new_index[i] = -1;
    //traverse all polygons
    for (i = 0; i < in_no_pol; i++)
    {
        int selected = 0;
        int start_pol = in_pol_l[i];
        int end_pol;
        if (i < in_no_pol - 1)
            end_pol = in_pol_l[i + 1];
        else
            end_pol = in_no_v;
        // cerr << "start: " << start_pol	<<" end: "<< end_pol << endl;
        if (!per_polygon)
            for (j = start_pol; j < end_pol; j++)
            {
                // if one vertex has a value in the interval, stop processing, set flag
                if (tags[in_vl[j]])
                {
                    selected = 1;
                    break;
                }
            }
        else
            selected = tags[i];

        // now, we need to copy the polygon into the temporary polygon list
        if (selected)
            sel_polygons[i] = 1;

        if (selected)
        {
            if (per_polygon)
            {
                t_p_l[o_no_pol] = o_no_v;
                t_data[o_no_pol] = in_data[i];
                for (j = start_pol; j < end_pol; j++)
                {
                    int vertex = in_vl[j];
                    if (new_index[vertex] < 0) //new_index < 0 means, Point wasn't processed until now
                    {
                        // That means, we need to copy all data, and increment the indices
                        t_vl[o_no_v] = o_no_points;
                        t_x[o_no_points] = in_x[vertex];
                        t_y[o_no_points] = in_y[vertex];
                        t_z[o_no_points] = in_z[vertex];
                        new_index[vertex] = o_no_points;
                        o_no_points++;
                        o_no_v++;
                    }
                    else // Othervise, new_index contains the new vertex-index
                    {
                        // now we just need to set the vertex-list
                        t_vl[o_no_v] = new_index[vertex];
                        o_no_v++;
                    }
                }
                o_no_pol++; //another polygon copied
            }
            else // per point
            {
                t_p_l[o_no_pol] = o_no_v;
                for (j = start_pol; j < end_pol; j++)
                {
                    int vertex = in_vl[j];
                    if (new_index[vertex] < 0) //new_index < 0 means, Point wasn't processed until now
                    {
                        // That means, we need to copy all data, and increment the indices
                        t_vl[o_no_v] = o_no_points;
                        t_x[o_no_points] = in_x[vertex];
                        t_y[o_no_points] = in_y[vertex];
                        t_z[o_no_points] = in_z[vertex];
                        t_data[o_no_points] = in_data[vertex];
                        new_index[vertex] = o_no_points;
                        o_no_points++;
                        o_no_v++;
                    }
                    else // Othervise, new_index contains the new vertex-index
                    {
                        // now we just need to set the vertex-list
                        t_vl[o_no_v] = new_index[vertex];
                        o_no_v++;
                    }
                }
                o_no_pol++; //another polygon copied
            }
        }
    }
    //	cerr << "polys: " << o_no_pol << " verts: " << o_no_v << " points: " << o_no_points << endl;
    // Now we copy the temporary list (wich is too large) into tight fitting lists
    float *o_x = new float[o_no_points];
    float *o_y = new float[o_no_points];
    float *o_z = new float[o_no_points];
    float *o_scal;
    if (per_polygon)
        o_scal = new float[o_no_pol];
    else
        o_scal = new float[o_no_points];
    int *o_vl = new int[o_no_v];
    int *o_pol_l = new int[o_no_pol];
    memcpy(o_x, t_x, o_no_points * sizeof(float));
    memcpy(o_y, t_y, o_no_points * sizeof(float));
    memcpy(o_z, t_z, o_no_points * sizeof(float));
    if (per_polygon)
        memcpy(o_scal, t_data, o_no_pol * sizeof(float));
    else
        memcpy(o_scal, t_data, o_no_points * sizeof(float));
    memcpy(o_vl, t_vl, o_no_v * sizeof(int));
    memcpy(o_pol_l, t_p_l, o_no_pol * sizeof(int));
    // and put those lists into shared mem
    *out_poly = new coDoPolygons(out_poly_name, o_no_points, o_x, o_y, o_z, o_no_v, o_vl, o_no_pol, o_pol_l);
    if (per_polygon)
    {
        *out_data = new coDoFloat(out_data_name, o_no_pol, o_scal);
    }
    else
    {
        *out_data = new coDoFloat(out_data_name, o_no_points, o_scal);
    }
    // now process mapped data
    int ret = do_mapped_data(sel_polygons, out_map_data, out_map_data_name);

    // clean up - we're set
    delete[] o_x;
    delete[] o_y;
    delete[] o_z;
    delete[] o_scal;
    delete[] o_vl;
    delete[] o_pol_l;
    delete[] t_x;
    delete[] t_y;
    delete[] t_z;
    delete[] t_data;
    delete[] t_vl;
    delete[] t_p_l;
    delete[] sel_polygons;
    delete[] tags;
    delete[] new_index;
    return ret; //nothing can go wrong
}

int clip_interval::do_mapped_data(const int *sel_polygons,
                                  coDistributedObject **out_map_data, const char **out_map_data_name)
{
    int i, j, k, ret = 0;
    float *u_in_c[3];
    float *u_c[3];
    for (i = 0; i < no_ports_; ++i)
    {
        if (!data_map_[i])
            continue;
        if (!data_map_[i]->objectOk())
        {
            Covise::sendWarning("Object not OK at port %d", i + 1 + 2);
            ret = -1;
            continue;
        }
        if (const coDoFloat *data_scal = dynamic_cast<const coDoFloat *>(data_map_[i]))
        {
            data_scal->getAddress(&u_in_c[0]);
            if (o_no_points && data_scal->getNumPoints() == in_no_points)
            {
                // data per vertex
                out_map_data[i] = new coDoFloat(
                    out_map_data_name[i], o_no_points);
                ((coDoFloat *)(out_map_data[i]))->getAddress(&u_c[0]);
                for (j = 0; j < in_no_points; ++j)
                {
                    if (new_index[j] >= 0)
                    {
                        u_c[0][new_index[j]] = u_in_c[0][j];
                    }
                }
            }
            else if (o_no_points && data_scal->getNumPoints() == in_no_pol)
            {
                // data per polygon
                out_map_data[i] = new coDoFloat(
                    out_map_data_name[i], o_no_pol);
                ((coDoFloat *)(out_map_data[i]))->getAddress(&u_c[0]);
                for (j = 0, k = 0; j < in_no_pol; ++j)
                {
                    if (sel_polygons[j] == 1)
                    {
                        u_c[0][k] = u_in_c[0][j];
                        ++k;
                    }
                }
            }
            else // dummy
            {
                if (o_no_points && data_scal->getNumPoints() != 0)
                {
                    Covise::sendWarning("Object in port %d has wrong length: output dummy", i + 2);
                }
                out_map_data[i] = new coDoFloat(
                    out_map_data_name[i], 0);
            }
        }
        else if (const coDoVec3 *data_vect = dynamic_cast<const coDoVec3 *>(data_map_[i]))
        {
            data_vect->getAddresses(&u_in_c[0], &u_in_c[1], &u_in_c[2]);
            if (o_no_points && data_vect->getNumPoints() == in_no_points)
            {
                // data per vertex
                out_map_data[i] = new coDoVec3(
                    out_map_data_name[i], o_no_points);
                ((coDoVec3 *)(out_map_data[i]))->getAddresses(&u_c[0], &u_c[1], &u_c[2]);
                for (j = 0; j < in_no_points; ++j)
                {
                    if (new_index[j] >= 0)
                    {
                        u_c[0][new_index[j]] = u_in_c[0][j];
                        u_c[1][new_index[j]] = u_in_c[1][j];
                        u_c[2][new_index[j]] = u_in_c[2][j];
                    }
                }
            }
            else if (o_no_points && data_vect->getNumPoints() == in_no_pol)
            {
                // data per polygon
                out_map_data[i] = new coDoVec3(
                    out_map_data_name[i], o_no_pol);
                ((coDoVec3 *)(out_map_data[i]))->getAddresses(&u_c[0], &u_c[1], &u_c[2]);
                for (j = 0, k = 0; j < in_no_pol; ++j)
                {
                    if (sel_polygons[j] == 1)
                    {
                        u_c[0][k] = u_in_c[0][j];
                        u_c[1][k] = u_in_c[1][j];
                        u_c[2][k] = u_in_c[2][j];
                        ++k;
                    }
                }
            }
            else
            {
                // dummy
                if (o_no_points && data_vect->getNumPoints() != 0)
                {
                    Covise::sendWarning("Object in port %d has wrong length: output dummy", i + 2);
                }
                out_map_data[i] = new coDoVec3(
                    out_map_data_name[i], 0);
            }
        }
        else if (const coDoTensor *data_tens = dynamic_cast<const coDoTensor *>(data_map_[i]))
        {
            data_tens->getAddress(&u_in_c[0]);
            if (o_no_points && data_tens->getNumPoints() == in_no_points)
            {
                // data per vertex
                out_map_data[i] = new coDoTensor(
                    out_map_data_name[i], o_no_points, data_tens->getTensorType());
                ((coDoFloat *)(out_map_data[i]))->getAddress(&u_c[0]);
                for (j = 0; j < in_no_points; ++j)
                {
                    if (new_index[j] >= 0)
                    {
                        u_c[0][new_index[j]] = u_in_c[0][j];
                    }
                }
            }
            else if (o_no_points && data_tens->getNumPoints() == in_no_pol)
            {
                // data per polygon
                out_map_data[i] = new coDoTensor(
                    out_map_data_name[i], o_no_pol, data_tens->getTensorType());
                ((coDoTensor *)(out_map_data[i]))->getAddress(&u_c[0]);
                for (j = 0, k = 0; j < in_no_pol; ++j)
                {
                    if (sel_polygons[j] == 1)
                    {
                        u_c[0][k] = u_in_c[0][j];
                        ++k;
                    }
                }
            }
            else
            {
                // dummy
                if (o_no_points && data_tens->getNumPoints() != 0)
                {
                    Covise::sendWarning("Object in port %d has wrong length: output dummy", i + 2);
                }
                out_map_data[i] = new coDoTensor(
                    out_map_data_name[i], 0, data_tens->getTensorType());
            }
        }
        else
        {
            Covise::sendWarning("Object in port %d has a wrong type", i + 1 + 2);
            ret = -1;
        }
    }
    return ret;
}

clip_interval::clip_interval(const coDoPoints *points, const coDoFloat *data,
                             const coDistributedObject **data_map, int no_ports, int dummy, float min, float max)
    : no_ports_(no_ports)
    , data_map_(data_map)
    , upon_dummy_(dummy)
{
    points->getAddresses(&in_x, &in_y, &in_z);
    in_no_pol = 0;
    in_no_points = points->getNumPoints();

    data->getAddress(&in_data);
    no_data_points = data->getNumPoints();
    per_polygon = 0;
    /*********************************************************
     if(no_data_points!=in_no_points)
     if(no_data_points==in_no_pol)
      per_polygon=1;
   else
   {
      cerr << "no_points differ ! data: " << no_data_points << " points:" << in_no_points << endl;
      //cerr << "polys: " << in_no_pol << endl;
   //}
   **********************************************************/
    min_value = min;
    max_value = max;
}

int clip_interval::do_clip(coDoPoints **out_points, const char *out_poly_name, coDoFloat **out_data, const char *out_data_name, coDistributedObject **out_map_data, const char **out_map_data_name)
{
    int i;
    o_no_pol = 0; // used in do_mapped_data in case of dummy data!!!
    o_no_points = 0;
    int *sel_polygons = new int[in_no_points]; // used for mapped data
    memset(sel_polygons, '\0', in_no_points * sizeof(int));

    if (no_data_points == 0)
    {
        if (upon_dummy_)
            *out_points = new coDoPoints(out_poly_name, 0);
        else
            *out_points = replicatePoint(out_poly_name);
        *out_data = new coDoFloat(out_data_name, 0);
        int ret = do_mapped_data(sel_polygons, out_map_data, out_map_data_name);
        return ret;
    }
    else if (no_data_points != in_no_points)
    {
        Covise::sendWarning("Number of points and data do not match");
        return 1;
    }

    float *t_x = new float[in_no_points];
    float *t_y = new float[in_no_points];
    float *t_z = new float[in_no_points];
    float *t_data = new float[no_data_points];

    tags = sel_polygons;
    new_index = new int[in_no_points];
    // checks all data if it is inside the chosen interval and sets tags to 1 if yes else to 0
    float data_min = FLT_MAX, data_max = -FLT_MAX;

    for (i = 0; i < no_data_points; i++)
    {
        if (in_data[i] >= min_value && in_data[i] <= max_value)
            tags[i] = 1;
        else
            tags[i] = 0;
        if (in_data[i] <= data_min)
            data_min = in_data[i];
        if (in_data[i] >= data_max)
            data_max = in_data[i];
    }
    Covise::sendInfo("data_min: %f data_max: %f\n", data_min, data_max);
    for (i = 0; i < in_no_points; i++)
        new_index[i] = -1;

    int selected;

    for (i = 0; i < in_no_points; i++)
    {
        selected = tags[i];

        // now, we need to copy the polygon into the temporary polygon list
        if (selected)
        {
            if (new_index[i] < 0) //new_index < 0 means, Point wasn't processed until now
            {
                // That means, we need to copy all data, and increment the indices
                //	t_vl[o_no_v]=o_no_points;
                t_x[o_no_points] = in_x[i];
                t_y[o_no_points] = in_y[i];
                t_z[o_no_points] = in_z[i];
                t_data[o_no_points] = in_data[i];
                new_index[i] = o_no_points;
                o_no_points++;
                // o_no_v++;
            }
        }
    }

    *out_points = new coDoPoints(out_poly_name, o_no_points, t_x, t_y, t_z);
    *out_data = new coDoFloat(out_data_name, o_no_points, t_data);
    int ret = do_mapped_data(sel_polygons, out_map_data, out_map_data_name);

    delete[] t_x;
    delete[] t_y;
    delete[] t_z;
    delete[] t_data;
    delete[] sel_polygons;
    delete[] new_index;
    return ret; //nothing can go wrong,
    // sl: schade, aber manches kann noch schief gehen...
}
