/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE SimplifySurfaceNT
//
//  New SimplifySurface
//
//  Initial version: end of year 2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: 19.05.2004 SL: better preservation of border lines.

#ifndef _SIMPLIFY_SURFACE_H_
#define _SIMPLIFY_SURFACE_H_

#include <api/coSimpleModule.h>
using namespace covise;

#define EDGECOLLAPSE 0
#define QUADRICCLUSTERING 1
#define DECIMATEPRO 2
#define QUADRICDECIMATION 3

class SimplifySurface : public coSimpleModule
{
public:
    SimplifySurface(int argc, char *argv[]);
    virtual ~SimplifySurface();

protected:
    virtual int compute(const char *port);

private:
    // MaxAngleVertex is used by Triangulate,
    // it returns the local node of a triangle (a number
    // >= 0 and < num_conn), where num_conn is the number of polygon vertices
    // x, y, z are coordinates accessed through start_conn
    int MaxAngleVertex(int num_conn,
                       const int *start_conn,
                       const float *x,
                       const float *y,
                       const float *z);
    // fan-like triangulation of polygons
    // the center of the fan is the vertex with the largest angle
    void Triangulate(vector<int> &tri_conn_list,
                     int num_poly, int num_conn,
                     const int *poly_list,
                     const int *conn_list,
                     const float *x, const float *y, const float *z);
    enum
    {
        NUM_DATA = 1
    };
    static const float PERCENT_DEFAULT;
    coInputPort *p_meshIn;
    coInputPort *p_dataIn[NUM_DATA];
    coInputPort *p_normalsIn;

    coOutputPort *p_meshOut;
    coOutputPort *p_dataOut[NUM_DATA];
    coOutputPort *p_normalsOut;

    coChoiceParam *p_method;

    coFloatParam *param_percent;
    coFloatParam *param_normaldeviation;
    coFloatSliderParam *param_domaindeviation;
    coFloatParam *param_datarelativeweight;
    coBooleanParam *param_ignoredata;
#ifdef HAVE_VTK
    // used for QuadricClustering
    coFloatVectorParam *param_divisions;
    coBooleanParam *param_smoothSurface;
    coBooleanParam *param_divisions_absolute;

    // used for DecimatePro
    coBooleanParam *param_preserveTopology;
    coBooleanParam *param_meshSplitting;
    coBooleanParam *param_boundaryVertexDeletion;
    coFloatParam *param_maximumError;
    coFloatParam *param_featureAngle; // not implemented yet
    coFloatParam *param_splitAngle; // not implemented yet

    virtual void postInst();
    virtual void param(const char *paramname, bool inMapLoading);
    void initializeNeighbourhood(int *elems_contain_point, int *elems_at_point_list, vector<int> &tri_conn_list, int n_vert);
    int getNumNeighbours(int *elems_contain_point, int vertex);
    void getNeighbourElemsOfVertex(int n_elems, int *elems_contain_point, int *elems_at_point_list, int *neighbourElems, int n_elems2);
    void getLocalCoords(double coord[][3], double origpos[3], double *k, double *l);
#endif
    /*   These parameters are now configure options with useful defaults
            coFloatParam *param_boundaryfactor;
            coIntScalarParam *param_valence;
            coChoiceParam *param_algo;         */
    float cf_BoundaryFactor;
    int cf_MaxValence;
    int cf_Algorithm;
};
#endif
