/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VTK2COVISE_H
#define VTK2COVISE_H

class vtkDataSet;
class vtkDataSetAttributes;
class vtkFieldData;
class vtkDataObject;
class vtkDataArray;
class vtkInformation;
class vtkImageData;

#include <util/coExport.h>
#include <do/coDistributedObject.h>
#include <string>
#include <vector>

namespace covise
{

class coDoAbstractData;
class coDoAbstractStructuredGrid;
class coDoGeometry;
class coDoPixelImage;
class coDoGrid;

class ALGEXPORT coVtk
{
public:
    enum Attributes
    {
        Scalars = 0,
        Vectors = 1,
        Normals = 2,
        TexCoords = 3,
        Tensors = 4,
        Any = 5
    };

    enum Flags
    {
        None = 0,
        Normalize = 1,
        RequireDouble = 2
    };

    static coDoGeometry *vtk2Covise(const coObjInfo &info, vtkDataSet *vtk);
    static coDoGrid *vtkGrid2Covise(const coObjInfo &info, vtkDataSet *vtk);
    static coDoAbstractData *vtkData2Covise(const coObjInfo &info, vtkDataArray *varr, const coDoAbstractStructuredGrid *sgrid = NULL);
    static coDoAbstractData *vtkData2Covise(const coObjInfo &info, vtkDataSetAttributes *vtk, int attribute, const char *name = NULL, const coDoAbstractStructuredGrid *sgrid = NULL);
    static coDoAbstractData *vtkData2Covise(const coObjInfo &info, vtkDataSet *vtk, int attribute, const char *name = NULL, const coDoAbstractStructuredGrid *sgrid = NULL);
    static coDoPixelImage *vtkImage2Covise(const coObjInfo &info, vtkImageData *vtk);

    static vtkDataObject *covise2Vtk(const coDistributedObject *geo,
                                     const std::vector<const coDistributedObject *> &fields,
                                     const coDistributedObject *normals);
    static vtkDataSet *covise2Vtk(const coDistributedObject *obj);
    static vtkDataSet *coviseGeometry2Vtk(const coDoGeometry *geo);
    static vtkDataSet *coviseGrid2Vtk(const coDoGrid *grid);
    static vtkDataArray *coviseData2Vtk(const coDoGrid *grid, const coDoAbstractData *data, Flags flags = None);

    static bool isPortRequired(vtkInformation *info);
    static std::string inPortTypeList(vtkInformation *info);
    static std::string outPortTypeList(vtkInformation *info);
};
}

#endif
