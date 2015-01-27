/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Transform
//
//  Transform geometry and data
//
//  Initial version:   26.09.97 Lars Frenzel
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: 31.10.00 Sven Kufer (new API)
//          18.05.02 Sergio Leseduarte (module is rewritten
//                   from scratch for better meintenance and enhancement,
//                   this has been motivated by the tiling option)
//           7.06.02 Sergio Leseduarte. Now the module may also get the
//                   parameter values through an attribute: TRANSFORM.

#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include <api/coSimpleModule.h>
using namespace covise;
#include <api/coHideParam.h>
#include <util/coviseCompat.h>

#include "BBoxes.h"
#include "Matrix.h"
#include "Geometry.h"

class Transform : public coSimpleModule
{
private:
    /// compute callback
    virtual int compute(const char *port);
    virtual void postInst();
    /// returns which data port has displacements for tiling
    int lookForDisplacements();
    /// preHandleObjects is required for tiling, because we could not calculate the real bounding box in compute if the model has several spacial blocks
    virtual void preHandleObjects(coInputPort **);
    virtual void setIterator(coInputPort **, int);
    virtual void postHandleObjects(coOutputPort **);

    static const int TYPE_MIRROR = 1;
    static const int TYPE_TRANSLATE = 2;
    static const int TYPE_ROTATE = 3;
    static const int TYPE_SCALE = 4;
    static const int TYPE_MULTI_ROTATE = 5;
    static const int TYPE_TILE = 6;
    static const int TIME_DEPENDENT = 7;

    // Diagnose
    int preHandleFailed_;
    int lagrange_;

    coBooleanParam *p_create_set_;
    coFloatParam *p_mirror_dist_, *p_multirot_scalar_,
        *p_rotate_scalar_, *p_scale_scalar_;
    coBooleanParam *p_mirror_and_original_;
    coFloatVectorParam *p_trans_vertex_, *p_scale_vertex_,
        *p_rotate_vertex_, *p_multirot_vertex_;
    coFloatVectorParam *p_mirror_normal_, *p_rotate_normal_,
        *p_multirot_normal_;
    coIntScalarParam *p_number_;
    coChoiceParam *p_type_; // *p_oldtype_;
    coChoiceParam *p_scale_type_;

    // tiling
    coChoiceParam *p_tiling_plane_;
    coIntVectorParam *p_tiling_min_;
    coIntVectorParam *p_tiling_max_;
    coBooleanParam *p_flipTile_;

    // Time-dependent
    coFileBrowserParam *p_time_matrix_;

    void useTransformAttribute(const coDistributedObject *geomIn);

    coHideParam *h_mirror_dist_;
    coHideParam *h_multirot_scalar_;
    coHideParam *h_rotate_scalar_;
    coHideParam *h_scale_scalar_;
    coHideParam *h_mirror_and_original_;
    coHideParam *h_trans_vertex_;
    coHideParam *h_scale_vertex_;
    coHideParam *h_rotate_vertex_;
    coHideParam *h_multirot_vertex_;
    coHideParam *h_mirror_normal_;
    coHideParam *h_rotate_normal_;
    coHideParam *h_multirot_normal_;
    coHideParam *h_number_;
    coHideParam *h_type_;
    coHideParam *h_tiling_plane_;
    coHideParam *h_tiling_min_;
    coHideParam *h_tiling_max_;
    coHideParam *h_flipTile_;
    std::vector<coHideParam *> hparams_;

    // time-dependent transformation
    vector<Matrix> dynamicRefSystem_;

    // ports
    static const int NUM_DATA_IN_PORTS = 4;
    coInputPort *p_geo_in_, *p_data_in_[NUM_DATA_IN_PORTS];
    coOutputPort *p_geo_out_, *p_data_out_[NUM_DATA_IN_PORTS], *p_angle_out_;

    // parameter controlling data behaviour
    coChoiceParam *p_dataType_[NUM_DATA_IN_PORTS];
    coHideParam *h_dataType_[NUM_DATA_IN_PORTS];

    BBoxes BBoxes_;
    int lookUp_;

    /** work out all required transformations
       * @param numTransformations pointer to the variable where the number of transformations is written by this function
       * @return Matrix array with the transformations
       */
    Matrix *fillTransformations(int *numTransformations);
    /** write multirotations
       * @param matrix array of transformations modified by this function
       */
    void MultiRotateMatrix(Matrix *matrix);
    /** write tiling transformations
       * @param matrix array of transformations modified by this function
       */
    void TileMatrix(Matrix *retMatrix);
    /** output transformed geometry
       * @param name output object name
       * @param geom container for input and output coordinates filled by the function
       * @param matrix the transformation
       * @param setFlag if we are bunching the output in a set
       */
    const coDistributedObject *OutputGeometry(const char *name,
                                              Geometry &geom,
                                              const coDistributedObject *,
                                              const Matrix &matrix,
                                              int setFlag);
    /** output transformed data
       * @param name output object name
       * @param geom container for input and output coordinates
       * @param dataType real vector or scalar or pseudo- vector or scalar, or displacements
       * @param matrix the eulerian transformation
       * @param matrix the lagrnMat transformation (only when tiling)
       * @param setFlag if we are bunching the output in a set
       */
    const coDistributedObject *OutputData(const char *name,
                                          const coDistributedObject *,
                                          const Geometry &,
                                          int dataType,
                                          const Matrix &matrix,
                                          const Matrix *lagrnMat,
                                          int setFlag);

    /// reusing obj, copy to new if necessary
    const coDistributedObject *retObject(const coDistributedObject *obj);

    coDistributedObject *AssembleObjects(const coDistributedObject *in, const char *name, const coDistributedObject *const *);
    static bool isUnstructured(const coDistributedObject *);

    // RedressOrientation corrects incorrect orientation
    void RedressOrientation(int numTransformations, Matrix *transformations);
    // RedressOrientation corrects incorrect orientation
    void RedressOrientation(coDistributedObject *grid,
                            coDistributedObject *data,
                            int port,
                            Matrix *transformation);

    bool transformationAsAttribute;

protected:
    /// copy all attributes rotating some attributes used in COVER
    void copyRotateAttributes(coDistributedObject *out, const coDistributedObject *in,
                              const Matrix &) const;

public:
    /// constructor
    Transform(int argc, char *argv[]);
};
#endif // _TRANSFORM_H_
