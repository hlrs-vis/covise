/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CuttingSurface_H
#define _CuttingSurface_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE CuttingSurfaceUsg CuttingSurface module                **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  23.02.95  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
#include <do/coDoGeometry.h>
#ifdef _COMPLEX_MODULE_
#include <alg/coColors.h>
#endif

using namespace covise;

class CuttingSurfaceModule : public covise::coSimpleModule
{

private:
    const short int shiftOut;
    int maxPolyPerVertex; //maximal number of polygons dor one vertex

    // params and ports
    int DataType; // 1 scalar, 0 vector
    bool Polyhedra; // use polyhedra support or not

    coInputPort *p_MeshIn, *p_DataIn, *p_IBlankIn;
    coOutputPort *p_MeshOut, *p_DataOut, *p_NormalsOut;
#ifdef _COMPLEX_MODULE_
    void getMinMax(const coDistributedObject *obj, float &min, float &max);
    coInputPort *p_ColorMapIn;

    coInputPort *p_SampleGeom_;
    coInputPort *p_SampleData_;
    coDoGeometry *SampleToGeometry(const coDistributedObject *grid, const coDistributedObject *data);

    coOutputPort *p_GeometryOut;
    coBooleanParam *p_color_or_texture;
    coChoiceParam *p_vector;
    coFloatParam *p_arrow_head_factor;
    coFloatParam *p_arrow_head_angle;
    coBooleanParam *p_project_lines;
    coFloatVectorParam *p_minmax;
    coBooleanParam *p_autoScale;
#endif
    bool DoPostHandle;

    coBooleanParam *p_gennormals, *p_genstrips, *p_genDummyS;
    coFloatParam *p_scalar;
    coFloatVectorParam *p_vertex, *p_point;
    coBooleanParam *p_skew;
    coChoiceParam *p_option;

    coFloatParam *p_vertexratio;

    // private member functions
    void UpdateScalar(int);
    void UpdatePoint(int);
    //
    int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    void ini_borders();
    void comp_borders(int nb_elem, float *x, float *y, float *z);

    float x_minb, x_maxb, y_minb, y_maxb, z_minb, z_maxb;

    virtual void preHandleObjects(coInputPort **);
    virtual void postHandleObjects(coOutputPort **);

    coDistributedObject *dummy_polygons(string name, int noSteps,
                                        float **dummyX, float **dummyY, float **dummyZ);
    coDistributedObject *dummy_tr_strips(string name, int noSteps,
                                         float **dummyX, float **dummyY, float **dummyZ);

    coDistributedObject *dummy_data(string name, int noSteps);
    coDistributedObject *dummy_normals(string name, int noSteps,
                                       float *dummyX, float *dummyY, float *dummyZ);
    virtual void copyAttributesToOutObj(coInputPort **input_ports,
                                        coOutputPort **output_ports, int i);
    void addFeedbackParams(coDistributedObject *obj);

#ifdef _COMPLEX_MODULE_
    void StaticParts(coDistributedObject **geopart,
                     coDistributedObject **normpart,
                     coDistributedObject **colorpart,
                     const coDistributedObject *geo,
                     const coDistributedObject *data,
                     string geometryOutName,
                     bool ColorMapAttrib = true,
                     const ScalarContainer *SCont = NULL);
    coFloatSliderParam *p_scale;
    coChoiceParam *p_length;
    coIntScalarParam *p_num_sectors;
#endif
    bool isDummy_;

    // which feedback to send, maybe both?
    enum FeedbackStyle
    {
        FEED_NONE,
        FEED_OLD,
        FEED_NEW,
        FEED_BOTH
    };
    FeedbackStyle fbStyle_;

    // automatically create module title? Disabled if user sets Title
    bool autoTitle;

    // config variable for autoTitle set?
    bool autoTitleConfigured;

public:
    // parameters for immediate mode
    float param_vertex[3];
    float param_point[3];
    float param_scalar;
    int param_option;
    bool pointMode; // true if point given last, false if scalar given

    float vertexAllocRatio; // allocate x% of numVert for output vertices
    CuttingSurfaceModule(int argc, char *argv[]);
    virtual ~CuttingSurfaceModule()
    {
    }
};
#endif // _CuttingSurface_H
