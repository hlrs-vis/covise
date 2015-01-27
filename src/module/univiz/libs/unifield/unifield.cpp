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

#include "unifield.h"

#ifdef VTK
#include "vtkPointData.h"
#endif

UniField::UniField(int nDims, int *dims, int nSpace, bool regular, int nComp, int *compVecLen, const char **compNames, int dataType)
{
    //allocField(nDims, dims, vecLen, nSpace, regular, dataType, name);
    allocField(nDims, dims, nSpace, regular, nComp, compVecLen, compNames, dataType);
}

#ifdef AVS
UniField::UniField(AVSfield *fld)
{
    avsFld = fld;
}
#endif

#ifdef COVISE
#ifdef COVISE5
UniField::UniField(std::vector<coOutPort *> opvec)
#else
UniField::UniField(std::vector<coOutputPort *> opvec)
#endif
{ // does not allocate nor assign data
    covGrid = NULL;
    covScalarData = NULL;
    covVector2Data = NULL;
    covVector3Data = NULL;

    outPorts = opvec;
}

#ifdef COVISE5
UniField::UniField(DO_StructuredGrid *cGrid,
                   DO_Structured_S3D_Data *cScalarData,
                   DO_Structured_V2D_Data *cVector2Data)
#else
UniField::UniField(coDoStructuredGrid *cGrid,
                   coDoFloat *cScalarData,
                   coDoVec2 *cVector2Data)
#endif
{
    covGrid = cGrid;
    covScalarData = cScalarData;
    covVector2Data = cVector2Data;
    covVector3Data = NULL;
    // outPorts: kept empy
}

#ifdef COVISE5
UniField::UniField(DO_StructuredGrid *cGrid,
                   DO_Structured_S3D_Data *cScalarData,
                   DO_Structured_V3D_Data *cVector3Data)
#else
UniField::UniField(coDoStructuredGrid *cGrid,
                   coDoFloat *cScalarData,
                   coDoVec3 *cVector3Data)
#endif
{
    covGrid = cGrid;
    covScalarData = cScalarData;
    covVector3Data = NULL;
    covVector3Data = cVector3Data;
    // outPorts: kept empy
}
#endif

#ifdef VTK
#if 0
UniField::UniField(float *data, float *positions)
{ // does not allocate nor assign data
  vtkFld = data;
  vtkPositions = positions;
}
#else
UniField::UniField(vtkStructuredGrid *vtkSG)
{
    vtkFld = vtkSG;
    selectedComp = 0;
}
#endif
#endif

UniField::~UniField()
{
#ifdef AVS
#endif

#ifdef COVISE
#endif

#ifdef VTK
#endif
}

void UniField::freeField()
{
#ifdef AVS
    if (avsFld)
    {
        AVSfield_free(avsFld);
        avsFld = NULL;
    }
#endif

#ifdef COVISE
// HACK ### actually assuming that field got assigned to a Covise output port
// -> nothing to be done
#endif

#ifdef VTK
#if 0
  if (vtkFld) {
    free(vtkFld);
    vtkFld = NULL;
    free(vtkPositions);
    vtkPositions = NULL;
  }
#else
    if (vtkFld)
        vtkFld->Delete();
#endif
#endif
}

#ifdef AVS
bool UniField::allocField(int nDims, int *dims, int nSpace, bool regular, int nComp, int *compVecLen, const char **, int dataType)
#else
#ifdef COVISE
//bool UniField::allocField(t vecLen, int, bool, int, const char)
bool UniField::allocField(int nDims, int *dims, int, bool, int nComp, int *compVecLen, const char **, int)
#else // VTK
//bool UniField::allocField(int nDims, int *dims, int vecLen, int nSpace, bool, int, const char *name)
bool UniField::allocField(int nDims, int *dims, int nSpace, bool, int nComp, int *compVecLen, const char **compNames, int)
#endif
#endif
{ // compNames: may be NULL and may have NULL entries
#ifdef AVS
    char regularity[256];
    sprintf(regularity, (regular ? "regular" : "irregular"));

    char type[256];
    switch (dataType)
    {
    case DT_FLOAT:
    {
        sprintf(type, "float");
    }
    break;
    default:
        printf("UniField: ERROR: unsupported data type\n");
    }

    if (nComp > 1)
        printf("UniField: ERROR: compVecLen>1 not supported\n");

    char desc[256];
    sprintf(desc, "field %dD %d-vector %s %d-space %s",
            nDims, compVecLen[0], regularity, nSpace, type);

    AVSfield_float *wf;
    wf = (AVSfield_float *)AVSdata_alloc(desc, dims);
    avsFld = (AVSfield *)wf;
    return (avsFld);
#endif

#ifdef COVISE
    if (outPorts.size() < 1)
    {
        fprintf(stderr, "UniField:allocField: no output port information, exit\n");
        exit(1); // ###
    }
#ifdef COVISE5
    covGrid = new DO_StructuredGrid
#else
    covGrid = new coDoStructuredGrid
#endif
        (outPorts[0]->getObjName(), // ### not ok
         dims[0],
         (nDims >= 2 ? dims[1] : 1),
         (nDims >= 3 ? dims[2] : 1));
    if (compVecLen[0] <= 1)
    {
#ifdef COVISE5
        covScalarData = new DO_Structured_S3D_Data(outPorts[1]->getObjName(), // ### not ok
                                                   dims[0],
                                                   (nDims >= 2 ? dims[1] : 1),
                                                   (nDims >= 3 ? dims[2] : 1));
#else
        covScalarData = new coDoFloat(outPorts[1]->getObjName(), // ### not ok
                                      dims[0] * (nDims >= 2 ? dims[1] : 1) * (nDims >= 3 ? dims[2] : 1));
#endif
    }
    else if (compVecLen[0] == 2)
    {
#ifdef COVISE5
        covVector2Data = new DO_Structured_V2D_Data(outPorts[1]->getObjName(), // ### not ok
                                                    dims[0],
                                                    (nDims >= 2 ? dims[1] : 1),
                                                    (nDims >= 3 ? dims[2] : 1));
#else
        covVector2Data = new coDoVec2(outPorts[1]->getObjName(), // ### not ok
                                      dims[0] * (nDims >= 2 ? dims[1] : 1) * (nDims >= 3 ? dims[2] : 1));
#endif
    }

    else if (compVecLen[0] == 3)
    {
#ifdef COVISE5
        covVector3Data = new DO_Structured_V3D_Data(outPorts[1]->getObjName(), // ### not ok
                                                    dims[0],
                                                    (nDims >= 2 ? dims[1] : 1),
                                                    (nDims >= 3 ? dims[2] : 1));
#else
        covVector3Data = new coDoVec3(outPorts[1]->getObjName(), // ### not ok
                                      dims[0] * (nDims >= 2 ? dims[1] : 1) * (nDims >= 3 ? dims[2] : 1));
#endif
    }
    else
    {
        fprintf(stderr, "UniField: error: unsupported vector length\n");
    }
    return (covGrid && (covScalarData || covVector2Data || covVector3Data));
#endif

#ifdef VTK
#if 0
  this->nDims = nDims;

  int vSize = 1;
  for (int i=0; i<nDims; i++) {
    this->dims[i] = dims[i];
    vSize *= dims[i];
  }

  this->vecLen = vecLen;

  this->nSpace = nSpace;
  
  this->regular = regular;

  vtkFld = (float *) malloc(vSize * vecLen * sizeof(float));

  vtkPositions = (float *) malloc(vSize * nSpace * sizeof(float));
#else

    if (!vtkFld)
        vtkFld = vtkStructuredGrid::New();

    selectedComp = 0;

    // get dims
    // ##### TODO: seems that VTK only supports 3-dims
    int dimensions[3] = { 1, 1, 1 };
    if (nDims > 3)
    {
        printf("UniField: error: VTK supports only 3-dim\n");
        nDims = 3;
    }
    int nnodes = 1;
    for (int i = 0; i < nDims; i++)
    {
        dimensions[i] = dims[i];
        nnodes *= dims[i];
    }

    // grid
    vtkFld->SetDimensions(dimensions);

    // coords
    vtkFloatArray *coords = vtkFloatArray::New();
    coords->SetNumberOfComponents(nSpace);
    coords->SetNumberOfTuples(nnodes);
    vtkPoints *points = vtkPoints::New();
    points->SetData(coords);
    vtkFld->SetPoints(points);
    coords->Delete();
    points->Delete();

    // data
    for (int c = 0; c < nComp; c++)
    {
        vtkFloatArray *dat = vtkFloatArray::New();
        dat->SetNumberOfComponents(compVecLen[c]);
        dat->SetNumberOfTuples(nnodes);
        if (compNames && compNames[c])
            dat->SetName(compNames[c]);
        else
        {
            char buf[256];
            sprintf(buf, "%p", dat);
            dat->SetName(buf);
        }
        vtkFld->GetPointData()->AddArray(dat);
        dat->Delete();
    }
#endif
    return true; // ###
#endif
}

#ifdef AVS
AVSfield *UniField::getField(void)
{
    return avsFld;
}
#endif

#ifdef COVISE
#ifdef COVISE5
void UniField::getField(DO_StructuredGrid **covGridP,
                        DO_Structured_S3D_Data **covScalarDataP,
                        DO_Structured_V2D_Data **covVector2DataP)
#else
void UniField::getField(coDoStructuredGrid **covGridP,
                        coDoFloat **covScalarDataP,
                        coDoVec2 **covVector2DataP)
#endif
{
    if (covGridP)
        *covGridP = covGrid;
    if (covScalarDataP)
        *covScalarDataP = covScalarData;
    if (covVector2DataP)
        *covVector2DataP = covVector2Data;
}
#endif

#ifdef COVISE
#ifdef COVISE5
void UniField::getField(DO_StructuredGrid **covGridP,
                        DO_Structured_S3D_Data **covScalarDataP,
                        DO_Structured_V3D_Data **covVector3DataP)
#else
void UniField::getField(coDoStructuredGrid **covGridP,
                        coDoFloat **covScalarDataP,
                        coDoVec3 **covVector3DataP)
#endif
{
    if (covGridP)
        *covGridP = covGrid;
    if (covScalarDataP)
        *covScalarDataP = covScalarData;
    if (covVector3DataP)
        *covVector3DataP = covVector3Data;
}
#endif

#ifdef VTK
#if 0
void UniField::getField(float **data,
			float **positions)
{
  *data = vtkFld;
  *positions = vtkPositions;
}
#else
vtkStructuredGrid *UniField::getField(void)
{
    return vtkFld;
}
#endif
#endif
