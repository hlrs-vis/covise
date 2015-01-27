/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Probe3D
//
//  Probing module
//
//  Initial version: 1.8.2004 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity GmbH
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _PROBE_3D_H_
#define _PROBE_3D_H_

#include "api/coModule.h"
#include "BBoxAdmin.h"
#include "util/coviseCompat.h"

using namespace covise;

class Probe3D : public coModule
{
public:
    Probe3D(int argc, char *argv[]);
    virtual ~Probe3D();
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);

protected:
private:
    bool firstComputation() const;
    bool getDiagnose() const;
    void loadNames();
    void eraseNames();
    void assignOctTrees(const BBoxAdmin *, vector<vector<const coDistributedObject *> > &);
    // per_cell=false : calculate points to evaluate
    // pre_cell=true  : calculate polygon points, every center of a polygon is an evaluation point
    void loadPoints(vector<float> &x,
                    vector<float> &y,
                    vector<float> &z, bool per_cell = false) const;
    void loadSquarePoints(int num_sidepoints, vector<float> point,
                          vector<float> normal, float side_length,
                          vector<float> &x, vector<float> &y, vector<float> &z, bool per_cell) const;

    void interpolateForAGrid(const float *coordinates, float *result,
                             const coDistributedObject *grid,
                             const coDistributedObject *field);
    void gInterpolate(const vector<float> &x,
                      const vector<float> &y,
                      const vector<float> &z,
                      const vector<vector<const coDistributedObject *> > &grids,
                      const vector<vector<const coDistributedObject *> > &field,
                      vector<vector<float> > &gresults);
    coDistributedObject *gOutput(const vector<vector<float> > &gresults, float min, float max, float avg);
    coDistributedObject *pOutput(const vector<vector<float> > &presults,
                                 const char *attribName, float min, float max);
    coDoPolygons *makePolygon(const char *name) const;
    coDoFloat *makeField(const char *name, const vector<float> &field) const;
    bool gridIsTimeDependent_;
    bool polyIsTimeDependent_;
    // 3d grid input
    coInputPort *p_grid_;
    coInputPort *p_gdata_;
    coInputPort *p_goctree_;

    // polygon input
    coInputPort *p_poly_;
    coInputPort *p_pdata_;
    coInputPort *p_poctree_;

    coInputPort *p_colorMapIn_;
    void transientMinMaxCalculation(float &min, float &max, float &avg,
                                    const vector<vector<float> > &gresults) const;
    void staticMinMaxCalculation(float &min, float &max, float &avg,
                                 const vector<float> &gresults) const;

    coOutputPort *p_gout_; // 3d grid output
    //coOutputPort *p_pout_; // poly output

    coChoiceParam *p_dimension_; // 3d, poly, both
    coChoiceParam *p_icon_type_; // point, square (only point in case of poly)

    float point_[3], normal_[3], side_, dir_[3];
    coFloatVectorParam *p_start1_, *p_start2_, *p_direction_;

    coIntSliderParam *p_numsidepoints_; // only for square

    // octree administration
    BBoxAdmin gBBoxAdmin_;
    BBoxAdmin pBBoxAdmin_;

    static void expandSets(coInputPort *port, vector<vector<const coDistributedObject *> > &tsteps,
                           bool *timeDependent = NULL);
    vector<vector<const coDistributedObject *> > grid_tsteps_;
    vector<vector<const coDistributedObject *> > gdata_tsteps_;
    vector<vector<const coDistributedObject *> > prid_tsteps_;
    vector<vector<const coDistributedObject *> > pdata_tsteps_;

    /*
   bool GoodOctTrees();
   bool GoodOctTrees(coDistributedObject *grid,coDistributedObject *otree);
*/

    // object names
    string gridName_;
    string gfieldName_;
    string gtreeName_;
    string pridName_;
    string pfieldName_;
    string ptreeName_;
};

#endif
