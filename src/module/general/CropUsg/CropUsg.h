/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CROPUSG_H
#define _CROPUSG_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ReduceUsg application module                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoPoints.h>
#include <api/coSimpleModule.h>
using namespace covise;

class CropUsg : public coSimpleModule
{
public:
    CropUsg(int argc, char *argv[]);

private:
    const char **sNames;
    // sets objects to ports
    // (at least those that are common to polygons and unsgrd's):
    // vector and scalar objects
    void common_assign(const coDoVec3 *vdata,
                       coDoVec3 *vDataOut,
                       const coDoFloat **sDataInArr,
                       coDoFloat **sDataOut);

    /// compute call-back
    virtual int compute(const char *port);
    virtual void preHandleObjects(coInputPort **);
    virtual void postInst();
    virtual void param(const char *paramname, bool /*in_map_loading*/);

    /// does the computation and returns a newly created object for output

    void reduce(const coDoUnstructuredGrid *grid_in,
                const coDoVec3 *vdata,
                const coDoFloat **sdata,
                coDoUnstructuredGrid **grid_out,
                coDoVec3 **vodata,
                coDoFloat ***sodata,
                const char *Gname, const char *Vname, const char **Snames);

    void reduce_poly(const coDoPolygons *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoPolygons **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames);

    void reduce_poly(const coDoLines *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoLines **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames);

    void reduce_poly(const coDoPoints *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoPoints **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames);

    /// method to find if a point is inside the croped region
    int outside(float x, float y, float z);
    int outside(float x);
    int dist_positive(float x, float y, float z, float d);

    int CheckDimensions(const coDistributedObject *grid, const coDoVec3 *vobj, const coDoFloat **sobjs);
    /// ports
    coInputPort *p_GridInPort_;
    coOutputPort *p_GridOutPort_;

    coInputPort *p_vDataInPort_;
    coOutputPort *p_vDataOutPort_;

    coInputPort *p_sData1InPort_;
    coOutputPort *p_sData1OutPort_;

    coInputPort *p_sData2InPort_;
    coOutputPort *p_sData2OutPort_;

    coInputPort *p_sData3InPort_;
    coOutputPort *p_sData3OutPort_;

    coInputPort *p_sData4InPort_;
    coOutputPort *p_sData4OutPort_;

    coInputPort *p_sData5InPort_;
    coOutputPort *p_sData5OutPort_;

    coInputPort *p_paramInPort_;
    coOutputPort *p_paramOutPort_;

    /// parameters
    coChoiceParam *p_method;
    coBooleanParam *p_type_;
    coFloatParam *p_Xmin_;
    coFloatParam *p_Xmax_;
    coFloatParam *p_Ymin_;
    coFloatParam *p_Ymax_;
    coFloatParam *p_Zmin_;
    coFloatParam *p_Zmax_;

    coFloatVectorParam *p_normal_;
    coFloatVectorParam *p_point_;

    coBooleanParam *p_invert_crop_;
    coBooleanParam *p_strict_removal;

    coFloatParam *p_data_min;
    coFloatParam *p_data_max;
    bool *sDataPerElem;
    /// float parameter values
    float Xmin_;
    float Xmax_;
    float Ymin_;
    float Ymax_;
    float Zmin_;
    float Zmax_;

    float nn[3]; // normal of plane
    float pp[3]; // base point of plane

    // int parameters
    int invert_crop;

    // number of used scalar data ports
    int numSclData_;
};

// we ensure an inline definition of the following methods
// by implementing it in the header

inline int
CropUsg::outside(float x, float y, float z)
{
    return (int)((x > Xmax_) || (x < Xmin_)
                 || (y > Ymax_) || (y < Ymin_)
                 || (z > Zmax_) || (z < Zmin_));
}

inline int
CropUsg::outside(float x)
{
    return (int)((x > Xmax_) || (x < Xmin_));
}

#endif // _CROPUSG_H
