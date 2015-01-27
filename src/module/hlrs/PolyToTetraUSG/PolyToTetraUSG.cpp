/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PolyToTetraUSG.h"

#include <sys/time.h>
#include <util/coVector.h>

#include <vtkPolyhedron.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include "vtkDataSetMapper.h"
#include "vtkSmartPointer.h"
#include "vtkExtractEdges.h"
#include "vtkProperty.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPolyhedron.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkPoints.h"
#include "vtkDataArray.h"
#include "vtkPointLocator.h"
#include "vtkDoubleArray.h"

/*! \brief constructor
 *
 * create In/Output Ports and module parameters
 */
PolyToTetraUSG::PolyToTetraUSG(int argc, char **argv)
    : coModule(argc, argv, "Convert Polyhedra to Tetrahedra")
{

    p_mesh = addInputPort("mesh", "UnstructuredGrid", "mesh");
    p_meshOut = addOutputPort("meshOut", "UnstructuredGrid", "meshOut");
}

PolyToTetraUSG::~PolyToTetraUSG()
{
}

/*! \brief param callback
 *
 * called when a parameter in a module is changed.
 */
void PolyToTetraUSG::param(const char * /* name */, bool /* inMapLoading */)
{
}

coDistributedObject *PolyToTetraUSG::convert(coDistributedObject *mesh,
                                             int child, int level)
{

    coDistributedObject *out = NULL;
    coDoUnstructuredGrid *grid = NULL;
    coDoSet *set = NULL;

    char meshName[128];

    if ((set = dynamic_cast<coDoSet *>(mesh)))
    {

        int numChildren = set->getNumElements();
        coDistributedObject **meshList = new coDistributedObject *[numChildren + 1];

        for (int index = 0; index < numChildren; index++)
            meshList[index] = convert(set->getElement(index), index, level + 1);

        meshList[numChildren] = NULL;

        if (level > 0)
            snprintf(meshName, 128, "%s_%d_%d", p_meshOut->getObjName(),
                     child, level);
        else
            snprintf(meshName, 128, "%s", p_meshOut->getObjName());

        out = new coDoSet(strdup(meshName), meshList);
    }
    else if ((grid = dynamic_cast<coDoUnstructuredGrid *>(mesh)))
    {

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->Initialize();

        vtkSmartPointer<vtkCellArray> faces = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkIdList> pointsIds = vtkSmartPointer<vtkIdList>::New();
        pointsIds->Initialize();

        vtkSmartPointer<vtkIdList> facePointsIds = vtkSmartPointer<vtkIdList>::New();
        facePointsIds->Initialize();

        vtkSmartPointer<vtkIdList> ptids = vtkIdList::New();
        vtkSmartPointer<vtkPoints> pts = vtkPoints::New();

        vtkSmartPointer<vtkUnstructuredGrid> ugrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

        int numElem, numConn, numCoord;
        int *elemList = NULL, *connList = NULL, *typeList = NULL;
        float *x = NULL, *y = NULL, *z = NULL;

        grid->getGridSize(&numElem, &numConn, &numCoord);
        grid->getAddresses(&elemList, &connList, &x, &y, &z);
        grid->getTypeList(&typeList);

        std::vector<int> newElemList, newTypeList, newConnList;

        printf("%d elements\n", numElem);

        for (int elemIndex = 0; elemIndex < numElem; elemIndex++)
        {

            if ((elemIndex % 1000) == 0 && elemIndex != 0)
            {
                printf("  %d/%d elements\n", elemIndex, numElem);
            }

            switch (typeList[elemIndex])
            {

            case TYPE_POLYHEDRON:
            {

                int start = elemList[elemIndex];
                int end;
                if (elemIndex == numElem - 1)
                    end = numConn - 1;
                else
                    end = elemList[elemIndex + 1] - 1;
                /*
                printf("polyhedron %d corners\n   ", end - start + 1);
                for (int index = start; index <= end; index ++)
                   printf("%d ", connList[index]);
                printf("\n");
                */
                points->Reset();
                faces->Reset();
                pointsIds->Reset();
                facePointsIds->Reset();

                std::map<int, int> indexLookup;
                std::map<int, int> reverseLookup;

                int startNode = -1;

                for (int index = start; index <= end; index++)
                {

                    if (connList[index] == startNode)
                    {

                        faces->InsertNextCell(facePointsIds);

                        if (index != end)
                        {
                            facePointsIds->Reset();

                            startNode = -1;
                            continue;
                        }
                    }
                    else
                    {
                        int localIndex = -1;
                        std::map<int, int>::iterator i = indexLookup.find(index);
                        if (i == indexLookup.end())
                        {
                            localIndex = (int)points->GetNumberOfPoints();
                            indexLookup[index] = localIndex;
                            reverseLookup[localIndex] = index;

                            points->InsertNextPoint(x[connList[index]],
                                                    y[connList[index]],
                                                    z[connList[index]]);
                        }
                        else
                            localIndex = i->second;

                        pointsIds->InsertNextId(localIndex);
                        facePointsIds->InsertNextId(localIndex);
                    }

                    if (startNode == -1)
                    {
                        startNode = connList[index];
                    }
                }
                ugrid->Reset();
                ugrid->SetPoints(points);
                ugrid->InsertNextCell(VTK_POLYHEDRON,
                                      (int)points->GetNumberOfPoints(),
                                      pointsIds->GetPointer(0),
                                      (int)faces->GetNumberOfCells(),
                                      faces->GetPointer());

                vtkPolyhedron *polyhedron = static_cast<vtkPolyhedron *>(ugrid->GetCell(0));
                if (polyhedron)
                {
                    numPolyhedra++;
                    ptids->Reset();
                    pts->Reset();

                    int d = polyhedron->Triangulate(0, ptids, pts);
                    /*
                   printf("%d original size: %d, tetrahedra: %d\n",
                          d, end - start + 1, (int) ptids->GetNumberOfIds());
                   */

                    for (int tetra = 0; tetra < (int)ptids->GetNumberOfIds() / 4; tetra++)
                    {
                        newElemList.push_back(newConnList.size());
                        newTypeList.push_back(TYPE_TETRAHEDER);
                        for (int index = 0; index < 4; index++)
                            newConnList.push_back(connList[reverseLookup[ptids->GetId(tetra * 4 + index)]]);
                    }
                }
                else
                    printf("+++NO POLYHEDRON\n");

                break;
            }

            default:
                newElemList.push_back(newConnList.size());
                newTypeList.push_back(typeList[elemIndex]);

                int start = elemList[elemIndex];
                int end;
                if (elemIndex == numElem - 1)
                    end = numConn - 1;
                else
                    end = elemList[elemIndex + 1] - 1;
                for (int index = start; index <= end; index++)
                    newConnList.push_back(connList[index]);

                break;
            }
        }

        if (level > 0)
            snprintf(meshName, 128, "%s_%d_%d", p_meshOut->getObjName(),
                     child, level);
        else
            snprintf(meshName, 128, "%s", p_meshOut->getObjName());

        printf("elements: %zu, conn: %zu, coord: %d\n", newElemList.size(),
               newConnList.size(), numCoord);
        out = new coDoUnstructuredGrid(strdup(meshName), newElemList.size(),
                                       newConnList.size(), numCoord,
                                       &(newElemList[0]), &(newConnList[0]),
                                       x, y, z, &(newTypeList[0]));
    }

    return out;
}

int PolyToTetraUSG::compute(const char * /* port */)
{

    numPolyhedra = 0;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    coDistributedObject *grid = p_mesh->getCurrentObject();

    if (grid)
    {

        coDistributedObject *out = convert(grid);
        if (out)
            p_meshOut->setCurrentObject(out);
    }
    else
        return STOP_PIPELINE;

    gettimeofday(&t1, NULL);

    double t = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / 1000000;

    printf("%d polyhedra in %f sec: %f per polyhedron\n",
           numPolyhedra, t, numPolyhedra / t);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, PolyToTetraUSG)
