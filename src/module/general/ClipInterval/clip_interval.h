/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CLIPINTERVAL_H
#define CLIPINTERVAL_H

#include <float.h>
#include <do/coDoData.h>

namespace covise
{
class coDistributedObject;
class coDoPolygons;
class coDoPoints;
}

using namespace covise;

class clip_interval
{
private:
    coDoPolygons *out_poly;
    float *in_data, *out_data;
    float min_value, max_value;
    // Polygonlists
    float *in_x, *in_y, *in_z;
    int in_no_v, *in_vl, o_no_v;
    int in_no_pol, *in_pol_l, o_no_pol;
    int in_no_points, o_no_points;
    int no_data_points;
    // Datalists
    int in_no_scal, o_no_scal;
    float *in_scal;
    // tag-list for chosen points
    int *tags;
    // list for new indices
    int *new_index;
    // flag if data is defined per polygon
    int per_polygon;
    // data for mapping
    int no_ports_;
    const coDistributedObject **data_map_;
    int do_mapped_data(const int *, coDistributedObject **, const char **);
    coDoPolygons *replicatePolygon(const char *);
    coDoPoints *replicatePoint(const char *out_poly_name);
    // dummy
    int upon_dummy_;

public:
    clip_interval(const coDoPolygons *poly, const coDoFloat *data,
                  const coDistributedObject **data_map, int no_ports, int dummy,
                  float min, float max);
    int do_clip(coDoPolygons **poly, const char *out_poly_name, coDoFloat **data, const char *out_data_name, coDistributedObject **, const char **);

    clip_interval(const coDoPoints *points, const coDoFloat *data,
                  const coDistributedObject **data_map, int no_ports, int dummy,
                  float min, float max);
    int do_clip(coDoPoints **points, const char *out_poly_name, coDoFloat **data, const char *out_data_name, coDistributedObject **, const char **);
};
#endif
