/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// helpers for Paraview
// filip sadlo
// cgl eth 2007

#include <vector>

#include "paraview_ext.h"

#include "unifield.h"
#include "unstructured.h"

void passLineFieldVertexData(vtkStructuredGrid *input,
                             std::vector<int> *usedNodes,
                             std::vector<int> *usedNodesVertCnt,
                             vtkPointSet *output)
{
    // input wrapper
    UniField *unif = new UniField(input);

    int vertCntTot = 0;
    for (int i = 0; i < (int)usedNodes->size(); i++)
    {
        vertCntTot += (*usedNodesVertCnt)[i];
    }

    // pass data
    for (int c = 0; c < unif->getCompNb(); c++)
    {

        // setup data
        vtkFloatArray *dat = vtkFloatArray::New();
        dat->SetNumberOfComponents(unif->getCompVecLen(c));
        dat->SetNumberOfTuples(vertCntTot);
        if (unif->getCompName(c))
            dat->SetName(unif->getCompName(c));
        else
        {
            char buf[256];
            sprintf(buf, "%p", dat);
            dat->SetName(buf);
        }
        output->GetPointData()->AddArray(dat);

        float *arrPtr = dat->GetPointer(0);

        unif->selectComp(c);

        // set data
        for (int lidx = 0; lidx < (int)usedNodes->size(); lidx++)
        {

            int line = (*usedNodes)[lidx];

            int lineVertCnt = (*usedNodesVertCnt)[lidx];

            for (int v = 0; v < lineVertCnt; v++)
            {
                for (int vc = 0; vc < unif->getCompVecLen(c); vc++)
                {
                    *(arrPtr++) = unif->getVectorComp(v, line, vc);
                }
            }
        }

        // reference counted, we can delete
        dat->Delete();
    }

    delete unif;
}

void passInterpolatePointData(vtkUnstructuredGrid *input,
                              vtkPointSet *output)
{ // pass data (interpolate from input)

    vtkPoints *outputPoints = output->GetPoints();

    // generate wrapper for all input
    Unstructured *unst_all;
    {
        std::vector<vtkFloatArray *> svec;
        std::vector<vtkFloatArray *> vvec;

        for (int a = 0; a < input->GetPointData()->GetNumberOfArrays(); a++)
        {
            vtkDataArray *inArray = input->GetPointData()->GetArray(a);
            if (inArray->GetNumberOfComponents() == 1)
                svec.push_back((vtkFloatArray *)inArray); // ### FLOAT HACK
            else
                vvec.push_back((vtkFloatArray *)inArray); // ### FLOAT HACK
        }

        unst_all = new Unstructured(input,
                                    (svec.size() ? &svec : NULL),
                                    (vvec.size() ? &vvec : NULL));
    }

    for (int c = 0; c < unst_all->getNodeCompNb(); c++)
    {

        // alloc array
        vtkFloatArray *arr = vtkFloatArray::New();
        vtkIdType numPts = outputPoints->GetNumberOfPoints();
        int veclen = unst_all->getNodeCompVecLen(c);
        arr->SetNumberOfComponents(veclen);
        arr->SetNumberOfTuples(numPts);

        float *arrPtr = arr->GetPointer(0);

        if (veclen == 1)
            unst_all->selectScalarNodeData(c);
        else
            unst_all->selectVectorNodeData(c);

        //printf("c=%d veclen=%d\n", c, veclen);
        for (int i = 0; i < outputPoints->GetNumberOfPoints(); i++)
        {

            vec3 pos;
            outputPoints->GetPoint(i, pos);
            unst_all->findCell(pos);

            if (veclen == 1)
            {
                double sc = unst_all->interpolateScalar();
                //printf("i=%d pos=[%g %g %g] scal=%g\n",
                //     i, pos[0], pos[1], pos[2],
                //     sc);
                arrPtr[i] = sc;
            }
            else if (veclen == 3)
            {
                vec3 v;
                unst_all->interpolateVector3(v);
                //printf("i=%d pos=[%g %g %g] v=[%g %g %g]\n",
                //     i, pos[0], pos[1], pos[2],
                //     v[0], v[1], v[2]);
                arrPtr[i * 3 + 0] = v[0];
                arrPtr[i * 3 + 1] = v[1];
                arrPtr[i * 3 + 2] = v[2];
            }
            else
            {
                printf("vtkFlowTopo: ERROR: unsupported input vector length\n");
            }
        }

        //printf("adding %s\n", unst_all->getNodeCompLabel(c));
        arr->SetName(unst_all->getNodeCompLabel(c));
        output->GetPointData()->AddArray(arr);

        // reference is counted, we can delete
        arr->Delete();
    }

    delete unst_all;
}

vtkUnstructuredGrid *generateUniformUSG(const char *name,
                                        float originX, float originY, float originZ,
                                        int cellsX, int cellsY, int cellsZ,
                                        float cellSize)
{ // generates uniform hexahedral USG

    bool success = true;

    int ncells = cellsX * cellsY * cellsZ;
    int nnodes = (cellsX + 1) * (cellsY + 1) * (cellsZ + 1);

    vtkUnstructuredGrid *grid = vtkUnstructuredGrid::New();

    vtkFloatArray *coords = vtkFloatArray::New();
    coords->SetNumberOfComponents(3);
    coords->SetNumberOfTuples(nnodes);

    vtkIdTypeArray *listcells = vtkIdTypeArray::New();
    // this array contains a list of NumberOfCells tuples
    // each tuple is 1 integer, i.e. the number of indices following it (N)
    // followed by these N integers
    listcells->SetNumberOfValues(ncells + 8 * ncells);

    vtkCellArray *cells = vtkCellArray::New();
    vtkPoints *points = vtkPoints::New();

    int *types = new int[ncells];
    if (!types)
    {
        printf("allocation failed\n");
        success = false;
        goto FreeOut;
    }

    // set coordinates and connectivity
    {
        vtkIdType *clist = listcells->GetPointer(0);
        float *ptr = coords->GetPointer(0);
        int n = 0;
        int c = 0;
        for (int z = 0; z < cellsZ + 1; z++)
        {
            for (int y = 0; y < cellsY + 1; y++)
            {
                for (int x = 0; x < cellsX + 1; x++)
                {
                    ptr[n * 3 + 0] = originX + x * cellSize;
                    ptr[n * 3 + 1] = originY + y * cellSize;
                    ptr[n * 3 + 2] = originZ + z * cellSize;

                    if ((x < cellsX) && (y < cellsY) && (z < cellsZ))
                    {
                        int nX = cellsX + 1;
                        int nXY = nX * (cellsY + 1);
                        int nXYPX = nXY + nX;
                        vtkIdType list[8] = {
                            n, n + 1, n + nX + 1, n + nX,
                            n + nXY, n + nXY + 1, n + nXYPX + 1, n + nXYPX
                        };

                        clist[c * 9] = 8;
                        memcpy(clist + c * 9 + 1, list, 8 * sizeof(vtkIdType));
                        types[c] = VTK_HEXAHEDRON;

                        c++;
                    }
                    n++;
                }
            }
        }
    }

    cells->SetCells(ncells, listcells);
    grid->SetCells(types, cells);
    points->SetData(coords);
    grid->SetPoints(points);

FreeOut:
    delete[] types;
    listcells->Delete();
    cells->Delete();
    coords->Delete();
    points->Delete();

    if (success)
        return grid;
    else
        return NULL;
}

char *getLabelName(const char *widgetName)
{
    static char buf[256];
    // REVERSE-ENGINEERING HACK:
    // THIS MUST BE CONSISTENT WITH pqNamedWidgets.cxx: createPanelLabel()
    sprintf(buf, "%s_label", widgetName);
    return buf;
}

// work arounds for ParaView bugs ---------------------------------------------

bool arrayExists(vtkFloatArray *optionalScalar, vtkFloatArray *requiredVector)
{
    float *sp = optionalScalar->GetPointer(0);
    float *vp = requiredVector->GetPointer(0);

    // ###### VERY UGLY HACK !!!
    if (sp[0] == vp[0] && sp[1] == vp[1] && sp[2] == vp[2] && sp[3] == vp[3])
    {
        // arrays are the same -> optional array does not exist
        return false;
    }
    else
    {
        return true;
    }
}
