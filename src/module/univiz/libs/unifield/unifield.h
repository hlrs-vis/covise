/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// Structured Field
//
// CGL ETH Zuerich
// Filip Sadlo 2006 - 2007

// Usage: define AVS or COVISE or VTK but not more than one
//        define also COVISE5 for Covise 5.x

/* TODO
   - VTK support: actually only a quick hack, find out if there is a array
     type inside VTK that is appropriate
 */

#ifndef _UNIFIELD_H_
#define _UNIFIELD_H_

#define UNIFIELD_VERSION "0.01"

#include <vector>

#include "linalg.h"

#ifdef AVS
// AVS
#include <avs/avs.h>
#include <avs/field.h>
#endif

#ifdef COVISE
// Covise
#ifdef COVISE5
#include <coModule.h>
#else
#include <api/coModule.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>
using namespace covise;
#endif
#endif

#ifdef VTK
#include "vtkStructuredGrid.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#endif

using namespace std;

#ifdef COVISE
using namespace covise;
#endif

class UniField
{

public:
    typedef enum
    {
        DT_FLOAT
    } dataTypeEnum;

private:
#ifdef AVS
    AVSfield *avsFld;
#endif

#ifdef COVISE
#ifdef COVISE5
    DO_StructuredGrid *covGrid;
    DO_Structured_S3D_Data *covScalarData;
    DO_Structured_V2D_Data *covVector2Data;
    DO_Structured_V3D_Data *covVector3Data;
    std::vector<coOutPort *> outPorts;
#else
    coDoStructuredGrid *covGrid;
    coDoFloat *covScalarData;
    coDoVec2 *covVector2Data;
    coDoVec3 *covVector3Data;
    std::vector<coOutputPort *> outPorts;
#endif
#endif

#ifdef VTK
private:
#if 0
  float *vtkFld;
  float *vtkPositions;
  int nDims;
  int dims[256];
  int vecLen;
  int nSpace;
  bool regular;
#else
    vtkStructuredGrid *vtkFld;
    int selectedComp;
#endif
#endif

public:
    //UniField(int nDims, int *dims, int vecLen, int nSpace, bool regular, int dataType, const char *name=NULL);
    UniField(int nDims, int *dims, int nSpace, bool regular, int nComp, int *compVecLen, const char **compNames, int dataType);

#ifdef AVS
    UniField(AVSfield *fld);
#endif

#ifdef COVISE
#ifdef COVISE5
    UniField(std::vector<coOutPort *> outPortVec);
#else
    UniField(std::vector<coOutputPort *> outPortVec);
#endif

#ifdef COVISE5
    UniField(DO_StructuredGrid *cGrid,
             DO_Structured_S3D_Data *cScalarData,
             DO_Structured_V2D_Data *cVector2Data);
#else
    UniField(coDoStructuredGrid *cGrid,
             coDoFloat *cScalarData,
             coDoVec2 *cVector2Data);
#endif

#ifdef COVISE5
    UniField(DO_StructuredGrid *cGrid,
             DO_Structured_S3D_Data *cScalarData,
             DO_Structured_V3D_Data *cVector3Data);
#else
    UniField(coDoStructuredGrid *cGrid,
             coDoFloat *cScalarData,
             coDoVec3 *cVector3Data);
#endif
#endif

#ifdef VTK
#if 0
  UniField(float *data, float *positions);
#else
    UniField(vtkStructuredGrid *vtkSG);
#endif
#endif

    ~UniField();

#ifdef AVS
    AVSfield *getField(void);
#endif

#ifdef COVISE
#ifdef COVISE5
    void getField(DO_StructuredGrid **covGrid,
                  DO_Structured_S3D_Data **covScalarData,
                  DO_Structured_V2D_Data **covVector2Data);
    void getField(DO_StructuredGrid **covGrid,
                  DO_Structured_S3D_Data **covScalarData,
                  DO_Structured_V3D_Data **covVector3Data);
#else
    void getField(coDoStructuredGrid **covGrid,
                  coDoFloat **covScalarData,
                  coDoVec2 **covVector2Data);
    void getField(coDoStructuredGrid **covGrid,
                  coDoFloat **covScalarData,
                  coDoVec3 **covVector3Data);
#endif
#endif

#ifdef VTK
#if 0
  void getField(float **data,
		float **positions);
#else
    vtkStructuredGrid *getField(void);
#endif
#endif

    //bool allocField(int nDims, int *dims, int vecLen, int nSpace, bool regular, int dataType, const char *name);
    bool allocField(int nDims, int *dims, int nSpace, bool regular, int nComp, int *compVecLen, const char **compNames, int dataType);
    void freeField();

    int getDimNb(void);
    int getDim(int axis);
    int getNodeNb(void);

    void setCoord(int i, vec3 pos);
    void setCoord(int i, int j, vec3 pos);
    void getCoord(int i, vec3 pos);
    void getCoord(int i, int j, vec3 pos);

    int getCompNb(void);
    int getCompVecLen(int comp);
    const char *getCompName(int comp);
    //float *getCompPtr(int comp);
    void selectComp(int comp);

    void setScalar(int i, double value);
    void setVectorComp(int i, int j, int vcomp, double value);
    double getVectorComp(int i, int vcomp);
    double getVectorComp(int i, int j, int vcomp);
};

// ------ inlined accessors ----------------------------------------------------

inline int UniField::getDimNb(void)
{
#ifdef AVS
    return avsFld->ndim;
#endif

#ifdef COVISE
    return 3; // ### TODO: OK? (seems that Covise fields are always 3D)
#endif

#ifdef VTK
    return 3; // ### TODO: OK? (seems that VTK fields are always 3D)
#endif
}

inline int UniField::getDim(int axis)
{
    assert(axis >= 0);
    assert(axis < 3);
#ifdef AVS
    switch (axis)
    {
    case 0:
        return MAXX(avsFld);
        break;
    case 1:
        return MAXY(avsFld);
        break;
    case 2:
        return MAXZ(avsFld);
        break;
    }
#endif

#ifdef COVISE
    int nx, ny, nz;
#ifdef COVISE5
    covGrid->get_grid_size(&nx, &ny, &nz);
#else
    covGrid->getGridSize(&nx, &ny, &nz);
#endif
    switch (axis)
    {
    case 0:
        return nx;
    case 1:
        return ny;
    case 2:
        return nz;
    }
#endif

#ifdef VTK
#if 0
  return dims[axis];
#else
    return vtkFld->GetDimensions()[axis];
#endif
#endif

    return -1;
}

inline int UniField::getNodeNb(void)
{
    int nnodes = 1;
    for (int i = 0; i < getDimNb(); i++)
    {
        nnodes *= getDim(i);
    }
    return nnodes;
}

inline void UniField::setCoord(int i, vec3 pos)
{
#ifdef AVS
    int dim = avsFld->dimensions[0];
    avsFld->points[i + 0 * dim] = pos[0];
    avsFld->points[i + 1 * dim] = pos[1];
    avsFld->points[i + 2 * dim] = pos[2];
#endif

#ifdef COVISE
    float *x, *y, *z;
#ifdef COVISE5
    covGrid->get_adresses(&x, &y, &z);
#else
    covGrid->getAddresses(&x, &y, &z);
#endif
    x[i] = pos[0];
    y[i] = pos[1];
    z[i] = pos[2];
#endif

#ifdef VTK
#if 0
  vtkPositions[i * nSpace + 0] = pos[0];
  vtkPositions[i * nSpace + 1] = pos[1];
  vtkPositions[i * nSpace + 2] = pos[2];
#else
    // #### HACK: assuming float coordinates
    float *ptr = ((vtkFloatArray *)vtkFld->GetPoints()->GetData())->GetPointer(0);
    int nSpace = 3; // only 3d points in VTK
    //double *ptr = vtkFld->GetPoint(i);
    ptr[i * nSpace + 0] = pos[0];
    ptr[i * nSpace + 1] = pos[1];
    ptr[i * nSpace + 2] = pos[2];
#endif
#endif
}

inline void UniField::setCoord(int i, int j, vec3 pos)
{
#ifdef AVS
    int dimI = avsFld->dimensions[0];
    int dimJ = avsFld->dimensions[1];
    avsFld->points[i + j * dimI + 0 * dimI * dimJ] = pos[0];
    avsFld->points[i + j * dimI + 1 * dimI * dimJ] = pos[1];
    avsFld->points[i + j * dimI + 2 * dimI * dimJ] = pos[2];
#endif

#ifdef COVISE
    float *x, *y, *z;
    int nx, ny, nz;
#ifdef COVISE5
    covGrid->get_adresses(&x, &y, &z);
    covGrid->get_grid_size(&nx, &ny, &nz);
#else
    covGrid->getAddresses(&x, &y, &z);
    covGrid->getGridSize(&nx, &ny, &nz);
#endif
    x[i + j * nx] = pos[0];
    y[i + j * nx] = pos[1];
    z[i + j * nx] = pos[2];
#endif

#ifdef VTK
#if 0
  vtkPositions[i * nSpace + j * dims[0] * nSpace + 0] = pos[0];
  vtkPositions[i * nSpace + j * dims[0] * nSpace + 1] = pos[1];
  vtkPositions[i * nSpace + j * dims[0] * nSpace + 2] = pos[2];
#else
    // #### HACK: assuming float coordinates
    float *ptr = ((vtkFloatArray *)vtkFld->GetPoints()->GetData())->GetPointer(0);
    int nSpace = 3; // only 3d points in VTK
    //double *ptr = vtkFld->GetPoint(i);
    int dim0 = getDim(0);
    ptr[i * nSpace + j * dim0 * nSpace + 0] = pos[0];
    ptr[i * nSpace + j * dim0 * nSpace + 1] = pos[1];
    ptr[i * nSpace + j * dim0 * nSpace + 2] = pos[2];
#endif
#endif
}

inline void UniField::getCoord(int i, vec3 pos)
{
#ifdef AVS
    int dim = avsFld->dimensions[0];
    pos[0] = avsFld->points[i + 0 * dim];
    pos[1] = avsFld->points[i + 1 * dim];
    pos[2] = avsFld->points[i + 2 * dim];
#endif

#ifdef COVISE
    float *x, *y, *z;
#ifdef COVISE5
    covGrid->get_adresses(&x, &y, &z);
#else
    covGrid->getAddresses(&x, &y, &z);
#endif
    pos[0] = x[i];
    pos[1] = y[i];
    pos[2] = z[i];
#endif

#ifdef VTK
#if 0
  pos[0] = vtkPositions[i * nSpace + 0];
  pos[1] = vtkPositions[i * nSpace + 1];
  pos[2] = vtkPositions[i * nSpace + 2];
#else
    // #### HACK: assuming float coordinates
    float *ptr = ((vtkFloatArray *)vtkFld->GetPoints()->GetData())->GetPointer(0);
    int nSpace = 3; // only 3d points in VTK
    //double *ptr = vtkFld->GetPoint(i);
    pos[0] = ptr[i * nSpace + 0];
    pos[1] = ptr[i * nSpace + 1];
    pos[2] = ptr[i * nSpace + 2];
#endif
#endif
}

inline void UniField::getCoord(int i, int j, vec3 pos)
{
#ifdef AVS
    int dimI = avsFld->dimensions[0];
    int dimJ = avsFld->dimensions[1];
    pos[0] = avsFld->points[i + j * dimI + 0 * dimI * dimJ];
    pos[1] = avsFld->points[i + j * dimI + 1 * dimI * dimJ];
    pos[2] = avsFld->points[i + j * dimI + 2 * dimI * dimJ];
#endif

#ifdef COVISE
    float *x, *y, *z;
    int nx, ny, nz;
#ifdef COVISE5
    covGrid->get_adresses(&x, &y, &z);
    covGrid->get_grid_size(&nx, &ny, &nz);
#else
    covGrid->getAddresses(&x, &y, &z);
    covGrid->getGridSize(&nx, &ny, &nz);
#endif
    pos[0] = x[i + j * nx];
    pos[1] = y[i + j * nx];
    pos[2] = z[i + j * nx];
#endif

#ifdef VTK
#if 0
  pos[0] = vtkPositions[i * nSpace + j * dims[0] * nSpace + 0];
  pos[1] = vtkPositions[i * nSpace + j * dims[0] * nSpace + 1];
  pos[2] = vtkPositions[i * nSpace + j * dims[0] * nSpace + 2];
#else
    // #### HACK: assuming float coordinates
    float *ptr = ((vtkFloatArray *)vtkFld->GetPoints()->GetData())->GetPointer(0);
    int nSpace = 3; // only 3d points in VTK
    //double *ptr = vtkFld->GetPoint(i);
    int dim0 = getDim(0);
    pos[0] = ptr[i * nSpace + j * dim0 * nSpace + 0];
    pos[1] = ptr[i * nSpace + j * dim0 * nSpace + 1];
    pos[2] = ptr[i * nSpace + j * dim0 * nSpace + 2];
#endif
#endif
}

inline int UniField::getCompNb(void)
{
#ifdef AVS
    return 1;
#endif
#ifdef COVISE
    return 1;
#endif
#ifdef VTK
    return vtkFld->GetPointData()->GetNumberOfArrays();
#endif
}

inline int UniField::getCompVecLen(int comp)
{
#ifdef AVS
    return avsFld->veclen;
#endif
#ifdef COVISE
    if (covVector2Data)
        return 2;
    else
        return 3;
#endif
#ifdef VTK
    return vtkFld->GetPointData()->GetArray(comp)->GetNumberOfComponents();
#endif
}

inline const char *UniField::getCompName(int comp)
{
#ifdef AVS
    return NULL;
#endif
#ifdef COVISE
    return NULL;
#endif
#ifdef VTK
    return vtkFld->GetPointData()->GetArrayName(comp);
#endif
}

#if 0
inline float *UniField::getCompPtr(int comp)
{
#ifdef AVS
  printf("UniField::getCompPtr: COVISE version not yet implemented! ###\n");
#endif
#ifdef COVISE
  printf("UniField::getCompPtr: COVISE version not yet implemented! ###\n");
#endif
#ifdef VTK
  // #### HACK: assuming float data
  return ((vtkFloatArray *) vtkFld->GetPointData()->GetArray(comp))->GetPointer(0);
#endif
}
#endif

inline void UniField::selectComp(int comp)
{
#ifdef VTK
    selectedComp = comp;
#endif
}

inline void UniField::setScalar(int i, double val)
{
#ifdef AVS
    I1DV((AVSfield_float *)avsFld, i)[0] = val; // HACK #### fixed to float...
#endif

#ifdef COVISE
    float *dat;
#ifdef COVISE5
    covScalarData->get_adress(&dat);
#else
    covScalarData->getAddress(&dat);
#endif
    dat[i] = val;
#endif

#ifdef VTK
#if 0
  vtkFld[i] = val;
#else
    // #### HACK: assuming float data
    float *ptr = ((vtkFloatArray *)vtkFld->GetPointData()->GetArray(selectedComp))->GetPointer(0);
    ptr[i] = val;
#endif
#endif
}

inline void UniField::setVectorComp(int i, int j, int vcomp, double value)
{
#ifdef AVS
    I2DV((AVSfield_float *)avsFld, i, j)[vcomp] = value; // HACK #### fixed to float...
#endif

#ifdef COVISE
    float *dat[2];
    int nx, ny, nz;
#ifdef COVISE5
    covVector2Data->get_adresses(&dat[0], &dat[1]);
    covGrid->get_grid_size(&nx, &ny, &nz);
#else
    covVector2Data->getAddresses(&dat[0], &dat[1]);
    covGrid->getGridSize(&nx, &ny, &nz);
#endif
    dat[vcomp][i + j * nx] = value;
#endif

#ifdef VTK
#if 0
  vtkFld[i * 2 + j * dims[0] * 2 + vcomp] = value;
#else
    // #### HACK: assuming float data
    float *ptr = ((vtkFloatArray *)vtkFld->GetPointData()->GetArray(selectedComp))->GetPointer(0);
    int dim0 = getDim(0);
    int vecLen = getCompVecLen(selectedComp);
    ptr[i * vecLen + j * dim0 * vecLen + vcomp] = value;
#endif
#endif
}

inline double UniField::getVectorComp(int i, int vcomp)
{
#ifdef AVS
    return I1DV((AVSfield_float *)avsFld, i)[vcomp]; // HACK #### fixed to float...
#endif

#ifdef COVISE
    printf("UniField::getVectorComp: COVISE version not yet implemented! ###\n");
    return 0.0;
#endif

#ifdef VTK
    printf("UniField::getVectorComp: VTK version not yet implemented! ###\n");
    return 0.0;
#endif
}

inline double UniField::getVectorComp(int i, int j, int vcomp)
{
#ifdef AVS
    return I2DV((AVSfield_float *)avsFld, i, j)[vcomp]; // HACK #### fixed to float...
#endif

#ifdef COVISE
    float *dat[2];
    int nx, ny, nz;
#ifdef COVISE5
    covVector2Data->get_adresses(&dat[0], &dat[1]);
    covGrid->get_grid_size(&nx, &ny, &nz);
#else
    covVector2Data->getAddresses(&dat[0], &dat[1]);
    covGrid->getGridSize(&nx, &ny, &nz);
#endif
    return dat[vcomp][i + j * nx];
#endif

#ifdef VTK
#if 0
  return vtkFld[i * 2 + j * dims[0] * 2 + vcomp];
#else
    // #### HACK: assuming float data
    float *ptr = ((vtkFloatArray *)vtkFld->GetPointData()->GetArray(selectedComp))->GetPointer(0);
    int dim0 = getDim(0);
    int vecLen = getCompVecLen(selectedComp);
    return ptr[i * vecLen + j * dim0 * vecLen + vcomp];
#endif
#endif
}

#endif // _UNIFIELD_H_
