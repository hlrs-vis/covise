/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELIMINATE_POLYGONS_H
#define _ELIMINATE_POLYGONS_H

#include <api/coModule.h>
using namespace covise;

class EliminatePolygons : public coModule
{
    //friend class SDomainsurface;
private:
    int disc_num;
    coDoPolygons *poly2;
    int **sorted_polygons;
    int *num_sorted;
    float *xpoly_min;
    float *ypoly_min;
    float *xpoly_max;
    double xmin, xmax, xstep;
    int fields_are_set;
    // for handling polygons to eliminate
    char elimMode;
    int num_in_polygons, poly_now;
    const coDoPolygons *polygonList[1000];

    // compute callback
    virtual int compute(const char *port);
    void Destruct();
    const coDistributedObject *handle_poly(const coDistributedObject *poly_in);
    coOutputPort *outPort_polySet;
    coInputPort *p_grid1, *p_grid2;

    coDoPolygons *eliminate_poly(const coDoPolygons *, const coDoPolygons *, char const *);
    coDistributedObject *handle_port1(const coDistributedObject *obj_in, const char *obj_name);

public:
    coDoPolygons *eliminate(const coDistributedObject *poly1,
                            const coDistributedObject *poly_away, const char *outName);
    EliminatePolygons(int argc, char *argv[]);
};
#endif
