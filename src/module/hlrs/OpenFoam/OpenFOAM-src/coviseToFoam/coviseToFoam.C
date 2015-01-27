/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 1991-2010 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------

Application
    coviseToFoam

Description
    Converts a Covise mesh to FOAM format

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "polyMesh.H"
#include "wallPolyPatch.H"
#include "genericPolyPatch.H"
#include "cyclicPolyPatch.H"
#include "cyclicGgiPolyPatch.H"
#include "preservePatchTypes.H"
#include "cellModeller.H"
#include "volFields.H"

#include "coSimClient.H"

#include <vector>
#include <map>
#include <fstream>

typedef struct
{
    int faceNr;
    int otherCell;
} cf;

int compare(const void *cf1, const void *cf2)
{
    if ((((cf *)cf1)->otherCell == -1) || (((cf *)cf2)->otherCell == -1))
    {
        if ((((cf *)cf1)->otherCell == -1) && (((cf *)cf2)->otherCell == -1))
        {
            if (((cf *)cf1)->faceNr > ((cf *)cf2)->faceNr)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            if (((cf *)cf1)->otherCell == -1)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    if (((cf *)cf1)->otherCell > ((cf *)cf2)->otherCell)
    {
        return true;
    }
    else
    {
        return false;
    }
}

struct CoviseToFoamCellTypeConverter : public std::map<int, std::string>
{
    static const int numCoviseCellTypes = 10;
    enum
    {
        TYPE_HEXAGON = 7,
        TYPE_HEXAEDER = 7,
        TYPE_PRISM = 6,
        TYPE_PYRAMID = 5,
        TYPE_TETRAHEDER = 4,
        TYPE_QUAD = 3,
        TYPE_TRIANGLE = 2,
        TYPE_BAR = 1,
        TYPE_NONE = 0,
        TYPE_POINT = 10
    };

    CoviseToFoamCellTypeConverter()
        : std::map<int, std::string>()
    {
        insert(std::pair<int, std::string>(TYPE_HEXAEDER, "hex"));
        insert(std::pair<int, std::string>(TYPE_PRISM, "prism"));
        insert(std::pair<int, std::string>(TYPE_PYRAMID, "pyr"));
        insert(std::pair<int, std::string>(TYPE_TETRAHEDER, "tet"));
    }
};

struct FenflossBocoType
{
    static const int inlet = 100;
    static const int outlet = 110;
    static const int periodicOne = 120;
    static const int periodicTwo = 130;
};

using namespace Foam;

// Determine whether cell is inside-out by checking for any wrong-oriented
// face.
bool correctOrientation(const pointField &points, const cellShape &shape)
{
    // Get centre of shape.
    point cc(shape.centre(points));

    // Get outwards pointing faces.
    faceList faces(shape.faces());

    forAll(faces, i)
    {
        const face &f = faces[i];

        vector n(f.normal(points));

        // Check if vector from any point on face to cc points outwards
        if (((points[f[0]] - cc) & n) < 0)
        {
            // Incorrectly oriented
            return false;
        }
    }

    return true;
}

//int main(int argc, char *argv[])
int coviseToFoam(Foam::Time &runTime)
{
    //   argList args(argc, argv);

    //#   include "createTime.H"

    Info << "\nConnecting to covise...\n";
    if (coInitConnect() != 0)
    {
        Info << " not connected, exit!" << endl;
        return -1;
    };
    Info << " connected!" << endl;

    // --- Receive grid ---
    Info << "\nReceiving grid...\n";
    //int numCoord;  USGNums[0]
    //int numElem;   USGNums[1]
    //int numConn;   USGNums[2]
    int USGNums[8];
    const char *machinetype;
    recvData(USGNums, 6 * sizeof(int));

    fprintf(stderr, "USGNums[0] = %d\n", USGNums[0]);
    fprintf(stderr, "USGNums[1] = %d\n", USGNums[1]);
    fprintf(stderr, "USGNums[2] = %d\n", USGNums[2]);
    fprintf(stderr, "USGNums[3] = %d\n", USGNums[3]);
    fprintf(stderr, "USGNums[4] = %d\n", USGNums[4]);
    fprintf(stderr, "USGNums[5] = %d\n", USGNums[5]);

    if (USGNums[5] == 0)
    {
        machinetype = "gate";
    }
    else
    {
        machinetype = "francis";
        // receive remaining 4 size information integers (Francis needs more size info integers)
        int sizeinfo[2];
        recvData(sizeinfo, 2 * sizeof(int));
        USGNums[6] = sizeinfo[0];
        USGNums[7] = sizeinfo[1];
    }

    float *x = new float[USGNums[0]]; // x-coordinates
    float *y = new float[USGNums[0]]; // y-coordinates
    float *z = new float[USGNums[0]]; // z-coordinates
    int *el; // element list
    int *tl; // type list
    int *conn; // connectivity list

    if (USGNums[5] == 0) // gate: use shape mesh constructor for polyMesh
    {
        std::cerr << "number of points:           " << USGNums[0] << endl;
        std::cerr << "number of elements:         " << USGNums[1] << endl;
        std::cerr << "size connectivity list:     " << USGNums[2] << endl;
        std::cerr << "number blades:              " << USGNums[3] << endl;
        std::cerr << "number of parallel regions: " << USGNums[4] << endl;
        std::cerr << "machine type:               " << machinetype << endl;

        recvData(x, USGNums[0] * sizeof(float));
        Info << "\tReceived array x..." << endl;
        recvData(y, USGNums[0] * sizeof(float));
        Info << "\tReceived array y..." << endl;
        recvData(z, USGNums[0] * sizeof(float));
        Info << "\tReceived array z..." << endl;

        int nMeshPoints = USGNums[0]; //int numCoord
        pointField points(nMeshPoints);
        for (int iMeshPoint = 0; iMeshPoint < nMeshPoints; ++iMeshPoint)
        {
            points[iMeshPoint].x() = x[iMeshPoint];
            points[iMeshPoint].y() = y[iMeshPoint];
            points[iMeshPoint].z() = z[iMeshPoint];
        }

        el = new int[USGNums[1]];
        tl = new int[USGNums[1]];
        conn = new int[USGNums[2]];

        recvData(el, USGNums[1] * sizeof(int));
        Info << "\tReceived array el..." << endl;
        recvData(tl, USGNums[1] * sizeof(int));
        Info << "\tReceived array tl..." << endl;

        recvData(conn, USGNums[2] * sizeof(int));
        Info << "\tReceived array conn..." << endl;

        Info << " done!" << endl;

        CoviseToFoamCellTypeConverter cellConv;
        //const cellModel& hex = *(cellModeller::lookup("hex"));
        //int nMeshCells = 2;
        int nMeshCells = USGNums[1]; //int numElem
        int sConnList = USGNums[2]; //int numConn
        cellShapeList cellShapes(nMeshCells);
        bool *cellOrientation = new bool[nMeshCells];
        //std::fill(cellOrientation, cellOrientation+nMeshCells, true);
        memset(cellOrientation, true, nMeshCells * sizeof(bool));
        /*
	FILE *fp;
	char fn[200];
	sprintf(fn,"shape_errors.txt");
	if( ( fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"couldn't open file '%s'!\n",fn);
	}
*/
        for (int iMeshCell = 0; iMeshCell < nMeshCells; ++iMeshCell)
        {
            std::map<int, std::string>::iterator cellConvIt = cellConv.find(tl[iMeshCell]);
            if (cellConvIt == cellConv.end())
            {
                Info << " Covise cell type " << tl[iMeshCell] << " not known. Continuing with next cell!" << endl;
                continue;
            }
            const cellModel &cmodel = *(cellModeller::lookup(cellConvIt->second));

            labelList cellPoints(((iMeshCell < nMeshCells - 1) ? el[iMeshCell + 1] : sConnList) - el[iMeshCell]);
            //labelList cellPoints(sConnList-el[iMeshCell]);

            if (cellConvIt->first == CoviseToFoamCellTypeConverter::TYPE_HEXAEDER)
            {

                cellPoints[0] = conn[(el[iMeshCell] + 0)];
                cellPoints[1] = conn[(el[iMeshCell] + 1)];
                cellPoints[2] = conn[(el[iMeshCell] + 2)];
                cellPoints[3] = conn[(el[iMeshCell] + 3)];
                cellPoints[4] = conn[(el[iMeshCell] + 4)];
                cellPoints[5] = conn[(el[iMeshCell] + 5)];
                cellPoints[6] = conn[(el[iMeshCell] + 6)];
                cellPoints[7] = conn[(el[iMeshCell] + 7)];
                /*
		 cellPoints[0] = conn[(el[iMeshCell]+4)];
		 cellPoints[1] = conn[(el[iMeshCell]+5)];
		 cellPoints[2] = conn[(el[iMeshCell]+6)];
		 cellPoints[3] = conn[(el[iMeshCell]+7)];
		 cellPoints[4] = conn[(el[iMeshCell]+0)];
		 cellPoints[5] = conn[(el[iMeshCell]+1)];
		 cellPoints[6] = conn[(el[iMeshCell]+2)];
		 cellPoints[7] = conn[(el[iMeshCell]+3)];
	*/
            }
            else
            {
                for (int iCellPoint = 0; iCellPoint < cellPoints.size(); ++iCellPoint)
                {
                    cellPoints[iCellPoint] = conn[(el[iMeshCell] + iCellPoint)];
                }
            }

            cellShapes[iMeshCell] = cellShape(cmodel, cellPoints);

            const cellShape &cell = cellShapes[iMeshCell];

            if (!correctOrientation(points, cell))
            {
                Info << "Inverting hex " << iMeshCell << endl;
                /*
	fprintf(fp,"Inverting hex %d: %d %d %d %d %d %d %d %d\n", iMeshCell, conn[(el[iMeshCell]+0)], 
									     conn[(el[iMeshCell]+1)], 
									     conn[(el[iMeshCell]+2)], 
									     conn[(el[iMeshCell]+3)], 
									     conn[(el[iMeshCell]+4)], 
									     conn[(el[iMeshCell]+5)], 
									     conn[(el[iMeshCell]+6)], 
									     conn[(el[iMeshCell]+7)]);
*/

                // Reorder hex.
                cellPoints[0] = conn[(el[iMeshCell] + 4)];
                cellPoints[1] = conn[(el[iMeshCell] + 5)];
                cellPoints[2] = conn[(el[iMeshCell] + 6)];
                cellPoints[3] = conn[(el[iMeshCell] + 7)];
                cellPoints[4] = conn[(el[iMeshCell] + 0)];
                cellPoints[5] = conn[(el[iMeshCell] + 1)];
                cellPoints[6] = conn[(el[iMeshCell] + 2)];
                cellPoints[7] = conn[(el[iMeshCell] + 3)];

                cellShapes[iMeshCell] = cellShape(cmodel, cellPoints);
                //cells[cellI] = cellShape(hex, hexPoints);
                cellOrientation[iMeshCell] = false;
            }

            /*if(cellShapes[iMeshCell].mag(points) < 0.0)
	      {
		 std::cout << "cell " << iMeshCell << " has negative magnitude" << std::endl;
		 faceList cellFaces = cellShapes[iMeshCell].faces();
		 std::cout << "Faces: " << std::endl;
		 for(int i=0; i<cellFaces.size(); ++i) {
		    pointField cellPoints = cellFaces[i].points(points);
		    for(int j=0; j<cellPoints.size(); ++j) {
		       point p = cellPoints[j];
		       std::cout << "(" << p[0] << ", " << p[1] << ", " << p[2] << ") " << std::endl;
		    }
		 }

		 cellPoints[0] = conn[(el[iMeshCell]+0)];
		 cellPoints[1] = conn[(el[iMeshCell]+1)];
		 cellPoints[2] = conn[(el[iMeshCell]+2)];
		 cellPoints[3] = conn[(el[iMeshCell]+3)];
		 cellPoints[4] = conn[(el[iMeshCell]+4)];
		 cellPoints[5] = conn[(el[iMeshCell]+5)];
		 cellPoints[6] = conn[(el[iMeshCell]+6)];
		 cellPoints[7] = conn[(el[iMeshCell]+7)];
		 cellShapes[iMeshCell] = cellShape(cmodel, cellPoints);
	      }*/
            /*if(cmodel.name() == "hex") {
		 // Faces in right direction?
		 // Get centre of shape.
		 point cc(cellShapes[iMeshCell].centre(points));

		 // Get outwards pointing faces.
		 faceList faces(cellShapes[iMeshCell].faces());

		 bool rightOrientation = true;
		 forAll(faces, i)
		 {
		    const face& f = faces[i];

		    vector n(f.normal(points));

		    // Check if vector from any point on face to cc points outwards
		    if (((points[f[0]] - cc) & n) < 0)
		    {
		       // Incorrectly oriented
		       rightOrientation = false;
		       break;
		    }
		 }
		 if(!rightOrientation) {
		    cellPoints[0] = cellPoints[4];
		    cellPoints[1] = cellPoints[5];
		    cellPoints[2] = cellPoints[6];
		    cellPoints[3] = cellPoints[7];
		    cellPoints[4] = cellPoints[0];
		    cellPoints[5] = cellPoints[1];
		    cellPoints[6] = cellPoints[2];
		    cellPoints[7] = cellPoints[3];
		    cellShapes[iMeshCell] = cellShape(cmodel, cellPoints);
		 }
	      }*/
        }

        //fclose(fp);

        // --- Receive boundary conditions ---
        Info << "\nReceiving boundary conditions...\n";

        int bocoNums[4];
        recvData(bocoNums, 4 * sizeof(int));
        for (int g = 0; g < 4; g++)
        {
            fprintf(stderr, "****** bocoNums[%d] = %d\n", g, bocoNums[g]);
        }

        std::cerr << "\tSize wall column: " << bocoNums[0] << ", number wall columns: " << bocoNums[1] << std::endl;
        std::cerr << "\tSize balance column: " << bocoNums[2] << ", number balance columns: " << bocoNums[3] << std::endl;

        int *wall = new int[(bocoNums[0] * bocoNums[1])];
        recvData(wall, bocoNums[0] * bocoNums[1] * sizeof(int));

        int *balance = new int[(bocoNums[2] * bocoNums[3])];
        recvData(balance, bocoNums[2] * bocoNums[3] * sizeof(int));

        const int npatch = 5; //wall, inlet, outlet, periodic
        faceListList boundary(npatch);
        wordList patchNames(npatch);
        wordList patchTypes(npatch);
        word defaultFacesName = "defaultFaces";
        word defaultFacesType = wallPolyPatch::typeName;
        wordList patchPhysicalTypes(npatch);

        //build wall boundary conditions
        faceList &wallFaceList = boundary[0];
        patchNames[0] = "wall";
        patchTypes[0] = wallPolyPatch::typeName;
        patchPhysicalTypes[0] = "wallFunctions";

        wallFaceList.setSize(bocoNums[1]);

        for (int iWallBoundary = 0; iWallBoundary < bocoNums[1]; ++iWallBoundary)
        {
            face &wallFace = wallFaceList[iWallBoundary];
            int *wallIt = wall + bocoNums[0] * iWallBoundary;
            wallFace.setSize(4);

            /*if(cellOrientation[wallIt[4]-1]==true) {
		 for(int iLabel = 0; iLabel<4; ++iLabel) {
		    wallFace[iLabel] = wallIt[iLabel]-1;
		 }
	      }
	      else {
		 for(int iLabel = 0; iLabel<4; ++iLabel) {
		    wallFace[iLabel] = wallIt[3-iLabel]-1;
		 }
	      }*/

            const cellShape &cell = cellShapes[wallIt[4] - 1];
            point cc(cell.centre(points));

            for (int iLabel = 0; iLabel < 4; ++iLabel)
            {
                wallFace[iLabel] = wallIt[iLabel] - 1;
            }
            vector n(wallFace.normal(points));

            if (((points[wallFace[0]] - cc) & n) < 0)
            {
                wallFace = wallFace.reverseFace();
            }
        }

        /*binary search useless because balance list is salad
	   //search for starts and ends of boundary condition types in balance list
	   std::vector<std::pair<int*, int> > startPointerList;
	   if(bocoNums[3]>1)
	   {
	      startPointerList.push_back(std::make_pair(balance, *(balance + 5)));
	      int* balanceLastPointer = balance+(bocoNums[2]*(bocoNums[3]-1));

	      //while( *(startPointerList.back().first+5) != *(balanceLastPointer+5) ) {
	      while( true ) {
		 int* currentStartPointer = startPointerList.back().first;
		 int currentBocoType = *(currentStartPointer + 5);
		 int* currentEndPointer = balanceLastPointer;

		 int diffColumns;
		 do {
		    diffColumns = (currentEndPointer - currentStartPointer)/bocoNums[2];
		    //std::cout << "diffColumns: " << diffColumns << std::endl;
		    int* currentSearchPointer = currentStartPointer + ((diffColumns+1)/2)*bocoNums[2];
		    if(currentBocoType == *(currentSearchPointer+5)) {
		       currentStartPointer = currentSearchPointer;
		    }
		    else {
		       currentEndPointer = currentSearchPointer;
		    }
		 } while(diffColumns>1);

		 if(currentStartPointer == currentEndPointer) break;

		 startPointerList.push_back(std::make_pair(currentEndPointer, *(currentEndPointer+5)));
		 std::cout << "Found start pointer at: " << (startPointerList.back().first-balance)/bocoNums[2] << ", type: " << startPointerList.back().second << std::endl;
	      }
	   }
	   else if(bocoNums[3]==1)
	   {
	      startPointerList.push_back(std::make_pair(balance, *(balance + 5)));
	   }
	   */

        //build balance boundary conditions
        faceList &inletFaceList = boundary[1];
        inletFaceList.setSize(bocoNums[3]);
        int inletFaceListIt = 0;
        faceList &outletFaceList = boundary[2];
        outletFaceList.setSize(bocoNums[3]);
        int outletFaceListIt = 0;
        faceList &cyclicOneFaceList = boundary[3];
        cyclicOneFaceList.setSize(bocoNums[3]);
        int cyclicOneFaceListIt = 0;
        //faceList cyclicTwoFaceList;
        faceList &cyclicTwoFaceList = boundary[4];
        cyclicTwoFaceList.setSize(bocoNums[3]);
        int cyclicTwoFaceListIt = 0;

        patchNames[1] = "inlet";
        //patchTypes[1] = genericPolyPatch::typeName;
        //patchPhysicalTypes[1] = "inlet";
        patchTypes[1] = polyPatch::typeName;
        patchNames[2] = "outlet";
        //patchTypes[2] = genericPolyPatch::typeName;
        //patchPhysicalTypes[2] = "outlet";
        patchTypes[2] = polyPatch::typeName;
        patchNames[3] = "periodic1";
        //patchTypes[3] = cyclicPolyPatch::typeName;
        patchTypes[3] = genericPolyPatch::typeName;
        patchNames[4] = "periodic2";
        //patchTypes[3] = cyclicPolyPatch::typeName;
        patchTypes[4] = genericPolyPatch::typeName;

        for (int iBalanceBoundary = 0; iBalanceBoundary < bocoNums[3]; ++iBalanceBoundary)
        {
            int *balanceIt = balance + bocoNums[2] * iBalanceBoundary;
            face *balanceFace = NULL;

            if (balanceIt[5] == FenflossBocoType::inlet)
            {
                balanceFace = &(inletFaceList[inletFaceListIt++]);
            }
            else if (balanceIt[5] == FenflossBocoType::outlet)
            {
                balanceFace = &(outletFaceList[outletFaceListIt++]);
            }
            else if (balanceIt[5] == FenflossBocoType::periodicOne)
            {
                balanceFace = &(cyclicOneFaceList[cyclicOneFaceListIt++]);
            }
            else if (balanceIt[5] == FenflossBocoType::periodicTwo)
            {
                balanceFace = &(cyclicTwoFaceList[cyclicTwoFaceListIt++]);
            }
            else
            {
                continue;
            }

            balanceFace->setSize(4);

            const cellShape &cell = cellShapes[balanceIt[4] - 1];
            point cc(cell.centre(points));

            for (int iLabel = 0; iLabel < 4; ++iLabel)
            {
                (*balanceFace)[iLabel] = balanceIt[iLabel] - 1;
            }
            vector n(balanceFace->normal(points));

            if (((points[(*balanceFace)[0]] - cc) & n) < 0)
            {
                (*balanceFace) = balanceFace->reverseFace();
            }

            /*if(cellOrientation[balanceIt[4]-1]==true) {
		 for(int iLabel = 0; iLabel<4; ++iLabel) {
		    (*balanceFace)[iLabel] = balanceIt[iLabel]-1;
		 }
	      }
	      else {
		 for(int iLabel = 0; iLabel<4; ++iLabel) {
		    (*balanceFace)[iLabel] = balanceIt[3-iLabel]-1;
		 }
	      }*/

            //for(int iLabel = 0; iLabel<4; ++iLabel) {
            //   (*balanceFace)[iLabel] = balanceIt[iLabel]-1;
            //}
        }
        inletFaceList.setSize(inletFaceListIt);
        outletFaceList.setSize(outletFaceListIt);
        cyclicOneFaceList.setSize(cyclicOneFaceListIt);
        cyclicTwoFaceList.setSize(cyclicTwoFaceListIt);

        //Works only for v1.7.1
        //cyclicOneFaceList.append(cyclicTwoFaceList);
        //cyclicOneFaceList.setSize(cyclicOneFaceListIt+cyclicTwoFaceListIt);
        //std::copy(cyclicTwoFaceList.begin(), cyclicTwoFaceList.end(), cyclicOneFaceList.begin()+cyclicOneFaceListIt);

        std::cerr << "Boco num faces: inlet: " << inletFaceListIt << ", outlet: " << outletFaceListIt << ", cyclic one: " << cyclicOneFaceListIt << ", cyclic two: " << cyclicTwoFaceListIt << std::endl;

        /*preservePatchTypes
	    (
		runTime,
		runTime.constant(),
		polyMesh::defaultRegion,
		patchNames,
		patchTypes,
		defaultFacesName,
		defaultFacesType,
		patchPhysicalTypes
	    );*/

        //xferMove works only for v1.6
        //polyMesh pShapeMesh
        //patched version:
        fvMesh pShapeMesh(
            IOobject(
                polyMesh::defaultRegion,
                runTime.constant(),
                runTime),
            xferMove(points),
            cellShapes,
            boundary,
            patchNames,
            patchTypes,
            defaultFacesName,
            defaultFacesType,
            patchPhysicalTypes);

        /*polyMesh pShapeMesh
	    (
		IOobject
		(
		    polyMesh::defaultRegion,
		    runTime.constant(),
		    runTime
		),
		points,
		cellShapes,
		boundary,
		patchNames,
		patchTypes,
		defaultFacesName,
		defaultFacesType,
		patchPhysicalTypes
	    );*/

        //Cyclic GGI boundary conditions
        Foam::List<Foam::polyPatch *> pPatches(pShapeMesh.boundaryMesh().size());
        //only non-periodic boundaries (2: num periodic boundaries in Gate)
        for (int iPatch = 0; iPatch < pShapeMesh.boundaryMesh().size() - 2; ++iPatch)
        {
            pPatches[iPatch] = Foam::polyPatch::New(
                                   patchTypes[iPatch],
                                   patchNames[iPatch],
                                   pShapeMesh.boundaryMesh()[iPatch].faceCells().size(),
                                   pShapeMesh.boundaryMesh()[iPatch].start(),
                                   iPatch,
                                   pShapeMesh.boundaryMesh()).ptr();
        }

        label ggiStart1 = pShapeMesh.boundaryMesh()[3].start();
        label ggiStart2 = pShapeMesh.boundaryMesh()[4].start();

        label ggiSize1 = pShapeMesh.boundaryMesh()[3].faceCells().size();
        label ggiSize2 = pShapeMesh.boundaryMesh()[4].faceCells().size();

        Foam::cyclicGgiPolyPatch *cyclicGgiSide1 = new Foam::cyclicGgiPolyPatch(
            patchNames[3],
            ggiSize1,
            ggiStart1,
            3,
            pShapeMesh.boundaryMesh(),
            patchNames[4],
            patchNames[3] + word("_zone"),
            false,
            Foam::vector(0, 0, 0),
            Foam::vector(0, 0, 1),
            Foam::scalar(-360.0 / USGNums[3]));

        Foam::cyclicGgiPolyPatch *cyclicGgiSide2 = new Foam::cyclicGgiPolyPatch(
            patchNames[4],
            ggiSize2,
            ggiStart2,
            4,
            pShapeMesh.boundaryMesh(),
            patchNames[3],
            patchNames[4] + word("_zone"),
            false,
            Foam::vector(0, 0, 0),
            Foam::vector(0, 0, 1),
            Foam::scalar(360.0 / USGNums[3]));

        pPatches[3] = cyclicGgiSide1;
        pPatches[4] = cyclicGgiSide2;

        pShapeMesh.removeBoundary();

        pShapeMesh.addFvPatches(pPatches);

        /*const pointField& meshPoints = pShapeMesh.points();
	    faceList faces = pShapeMesh.faces();
	    const labelList& owners = pShapeMesh.faceOwner();
	    const labelList& neighbours = pShapeMesh.faceNeighbour();
	    for(int iFace = 0; iFace<faces.size(); ++iFace) {
	      if(neighbours[iFace]==-1) continue;

	      point centreRelVec;
	      if(owners[iFace] > neighbours[iFace]) {
		 centreRelVec = cellShapes[owners[iFace]].centre(meshPoints) - cellShapes[neighbours[iFace]].centre(meshPoints);
	      }
	      else {
		 centreRelVec = cellShapes[neighbours[iFace]].centre(meshPoints) - cellShapes[owners[iFace]].centre(meshPoints);
	      }

	      if((centreRelVec & faces[iFace].normal(meshPoints)) < 0) {
		 faces[iFace] = faces[iFace].reverseFace();
	      }  
	    }

	    polyMesh pShapeMesh2
	    (
		IOobject
		(
		    polyMesh::defaultRegion,
		    runTime.constant(),
		    runTime
		),
		xferCopy(meshPoints),
		xferMove(faces),
		xferCopy(owners),
		xferCopy(neighbours)
	    );*/

        // Set the precision of the points data to 10
        IOstream::defaultPrecision(10);

        Info << "Writing polyMesh" << endl;
        pShapeMesh.write();
        //pShapeMesh2.write();

        delete[] el;
        delete[] tl;
        delete[] conn;

        Info << "Creating initial state with boundary conditions" << endl;

        Info << "\nReceiving boundary conditions...\n";

        int initValNums[2];
        recvData(initValNums, 2 * sizeof(int));
        std::cerr << "\tNumber diriclet nodes column: " << initValNums[0] << ", size diriclet values array: " << initValNums[1] << std::endl;

        int *dirNodes = new int[(initValNums[0] * initValNums[1])];
        recvData(dirNodes, initValNums[0] * initValNums[1] * sizeof(int));

        float *dirValues = new float[(initValNums[1])];
        recvData(dirValues, initValNums[1] * sizeof(float));

        std::map<int, std::vector<float> > diricletMap;
        //size of one diriclet value record: 5 values
        const int nRecordValues = 5;
        int nDirValues = initValNums[1] / nRecordValues;
        for (int iDirValue = 0; iDirValue < initValNums[1]; iDirValue += nRecordValues)
        {
            std::vector<float> values(nRecordValues);
            for (int iRecordValue = 0; iRecordValue < nRecordValues; ++iRecordValue)
            {
                values[iRecordValue] = dirValues[iDirValue + iRecordValue];
            }
            diricletMap[dirNodes[2 * iDirValue] - 1] = values;
        }

        wordList patchBoundaryPTypes(npatch);
        patchBoundaryPTypes[0] = "zeroGradient";
        patchBoundaryPTypes[1] = "zeroGradient";
        patchBoundaryPTypes[2] = "fixedMeanValue";
        patchBoundaryPTypes[3] = "cyclicGgi";
        patchBoundaryPTypes[4] = "cyclicGgi";

        volScalarField p(
            IOobject(
                "p",
                runTime.timeName(),
                pShapeMesh,
                //IOobject::READ_IF_PRESENT,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pShapeMesh,
            //fvMesh(pShapeMesh),
            dimensionedScalar("p", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
            patchBoundaryPTypes
            //dimensionedScalar(0.0)
            );

        //forAll(p.internalField(), i)
        //{
        //  p.internalField()[i] = 0.0;
        //}

        wordList patchBoundaryUTypes(npatch);
        patchBoundaryUTypes[0] = "fixedValue";
        patchBoundaryUTypes[1] = "fixedValue";
        patchBoundaryUTypes[2] = "inletOutlet";
        patchBoundaryUTypes[3] = "cyclicGgi";
        patchBoundaryUTypes[4] = "cyclicGgi";

        volVectorField U(
            IOobject(
                "U",
                runTime.timeName(),
                pShapeMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pShapeMesh,
            dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector(0.0, 0.0, 0.0)),
            patchBoundaryUTypes);
        U.internalField() = vector(-10.0, 0.0, -10.0);
        U.boundaryField() = vector(-10.0, 0.0, -10.0);

        U.boundaryField()[1].resize(inletFaceList.size());
        for (int iFace = 0; iFace < inletFaceList.size(); ++iFace)
        {
            const face &inletFace = inletFaceList[iFace];

            int nNodes = 0;
            float u = 0.0;
            float v = 0.0;
            float w = 0.0;
            for (int iPoint = 0; iPoint < inletFace.size(); ++iPoint)
            {
                std::map<int, std::vector<float> >::iterator dirIt = diricletMap.find(inletFace[iPoint]);
                if (dirIt != diricletMap.end())
                {
                    ++nNodes;
                    u += (dirIt->second)[0];
                    v += (dirIt->second)[1];
                    w += (dirIt->second)[2];
                }
            }
            if (nNodes == 4)
            {
                u *= 0.25;
                v *= 0.25;
                w *= 0.25;

                //inletFile << "\t( " << u << "\t" << v << "\t" << w << " )" << std::endl;
                U.boundaryField()[1][iFace] = vector(u, v, w);
            }
        }

        wordList patchBoundaryKTypes(npatch);
        patchBoundaryKTypes[0] = "zeroGradient";
        patchBoundaryKTypes[1] = "turbulentIntensityKineticEnergyInlet";
        patchBoundaryKTypes[2] = "zeroGradient";
        patchBoundaryKTypes[3] = "cyclicGgi";
        patchBoundaryKTypes[4] = "cyclicGgi";

        volScalarField k(
            IOobject(
                "k",
                runTime.timeName(),
                pShapeMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pShapeMesh,
            dimensionedScalar("k", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.375),
            patchBoundaryKTypes);

        wordList patchBoundaryEpsTypes(npatch);
        patchBoundaryEpsTypes[0] = "zeroGradient";
        patchBoundaryEpsTypes[1] = "turbulentMixingLengthDissipationRateInlet";
        patchBoundaryEpsTypes[2] = "zeroGradient";
        patchBoundaryEpsTypes[3] = "cyclicGgi";
        patchBoundaryEpsTypes[4] = "cyclicGgi";

        volScalarField epsilon(
            IOobject(
                "epsilon",
                runTime.timeName(),
                pShapeMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pShapeMesh,
            dimensionedScalar("epsilon", dimensionSet(0, 2, -3, 0, 0, 0, 0), 40.0),
            patchBoundaryEpsTypes);

        Info << "Writing initial state" << endl;

        p.write();
        U.write();
        k.write();
        epsilon.write();

        Info << "Parallel Decomposition Dictionary\n" << endl;
        label nDomains = USGNums[4];
        {
            IOdictionary decompDict(
                IOobject(
                    "decomposeParDict",
                    runTime.time().system(),
                    word::null, // use region if non-standard
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE,
                    false));
            //decompDict.lookup("numberOfSubdomains") >> nDomains;
            primitiveEntry decompDictEntry("numberOfSubdomains", nDomains);
            decompDict.set(decompDictEntry);
            decompDict.regIOobject::write();
        }
    }
    else // USGNums[5]==1 -> Francis
    {
        // use direct polyMesh constructor

        std::cerr << "number of nodes:            " << USGNums[0] << std::endl;
        std::cerr << "number of faces:            " << USGNums[1] << std::endl;
        std::cerr << "size connectivity list:     " << USGNums[2] << std::endl;
        std::cerr << "number blades:              " << USGNums[3] << std::endl;
        std::cerr << "number of parallel regions: " << USGNums[4] << std::endl;
        std::cerr << "number of hex cells (3D):   " << USGNums[6] << std::endl;
        std::cerr << "number of boundary faces:   " << USGNums[7] << std::endl;
        std::cerr << "machine type:               " << machinetype << std::endl;

        int numFaces = USGNums[1]; // = faces->getNumPolygons();	// num_internal_faces + num_boundary_faces
        int numFaceVertices = USGNums[2]; // = faces->getNumVertices();
        int numPoints = USGNums[0]; // = faces->getNumPoints();

        int numCells = USGNums[6]; // number of hex 3D cells
        int numBoundaryFaces = USGNums[7]; // num_boundary_faces
        //int numFaces = numInternalFaces + numBoundaryFaces;
        int numInternalFaces = numFaces - numBoundaryFaces;
        fprintf(stderr, "numPoints=%d\n", numPoints);
        fprintf(stderr, "numFaces=%d\n", numFaces);
        fprintf(stderr, "numInternalFaces=%d\n", numInternalFaces);
        fprintf(stderr, "numBoundaryFaces=%d\n", numBoundaryFaces);

        // receive face list
        int *faces = new int[numFaceVertices];
        fprintf(stderr, "coviseToFoam: receiving %d face vertices\n", numFaceVertices);
        recvData(faces, numFaceVertices * sizeof(int));
        Info << "\tReceived faces array ..." << endl;

        // receive coordinate array
        float *coords = new float[3 * numPoints];
        fprintf(stderr, "coviseToFoam: receiving %d coords\n", 3 * numPoints);
        recvData(coords, 3 * numPoints * sizeof(float));
        Info << "\tReceived coords array ..." << endl;
        for (int i = 0; i < numPoints; i++)
        {
            x[i] = coords[3 * i + 0];
            y[i] = coords[3 * i + 1];
            z[i] = coords[3 * i + 2];
        }
        delete[] coords;

        // receive owner list
        int *owner = new int[numFaces]; // USG_Nums[7] = num_internal_faces + num_boundary_faces
        fprintf(stderr, "coviseToFoam: receiving %d owners\n", numFaces);
        recvData(owner, numFaces * sizeof(int));
        Info << "\tReceived owner array ..." << endl;
        for (int i = 0; i < numFaces; i++)
        {
            if ((owner[i] < 0) || (owner[i] > (numCells - 1)))
            {
                fprintf(stderr, "error in owner list! owner cell of face %d is %d (numCells=%d)\n", i, owner[i], numCells);
            }
        }

        // receive neighbour list
        int *neighbour = new int[numFaces]; // -1 for boundary cells
        memset(neighbour, -1, numFaces * sizeof(int));

        recvData(neighbour, numInternalFaces * sizeof(int));
        Info << "\tReceived neighbour array ..." << endl;
        fprintf(stderr, "coviseToFoam: neighbour[0] = %d\n", neighbour[0]);
        fprintf(stderr, "coviseToFoam: neighbour[%d] = %d\n", numInternalFaces - 1, neighbour[numInternalFaces - 1]);

        // construct polyMesh (without boundaries)

        //- Points supporting the mesh
        pointField points_;

        //- Global face list for polyMesh
        faceList meshFaces_;

        //- Cells as polyhedra for polyMesh
        cellList cellPolys_;

        // points
        points_.setSize(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            points_[i] = point(x[i], y[i], z[i]);
        }
        delete[] x;
        delete[] y;
        delete[] z;

        // faces
        meshFaces_.setSize(numFaces); // boundary faces are already added here - patches later using addPatches()
        for (int i = 0; i < numFaces; i++)
        {
            labelList faceNodes(4);
            forAll(faceNodes, j)
            {
                faceNodes[j] = faces[4 * i + j];
            }
            //fprintf(stderr,"face %d has nodes %d %d %d %d\n", i, faces[4*i+0], faces[4*i+1], faces[4*i+2], faces[4*i+3]);
            face myNodes(faceNodes);
            meshFaces_[i] = myNodes;
        }
        delete[] faces;

        // cells
        // Add the faces in the increasing order of neighbours

        // for each face: we know owner and neighbour cells
        // construct a list with all the faces of each cell

        // jede Zelle hat 6 Faces (5 die Randzellen)
        // jedes Face hat eine Nachbarzelle
        // diese steht entweder bei owner oder bei neighbour drin, je nachdem ...
        // wir brauchen hier eine Liste mit Faces für jede Zelle
        //     sortiert in aufsteigender Reihenfolge der jeweils anderen Zelle

        // wir speichern alle Faces ab, die in Zusammenhang mit einer Zelle referenziert werden
        int *cellFacesOwner = new int[6 * numCells]; // each cell has 6 faces (pure hex mesh)
        memset(cellFacesOwner, -1, 6 * numCells * sizeof(int));
        int *cellNumFacesOwner = new int[numCells]; // faces already worked
        memset(cellNumFacesOwner, 0, numCells * sizeof(int));
        int *cellFacesNeighbour = new int[6 * numCells];
        memset(cellFacesNeighbour, -1, 6 * numCells * sizeof(int));
        int *cellNumFacesNeighbour = new int[numCells]; // faces already worked
        memset(cellNumFacesNeighbour, 0, numCells * sizeof(int));
        //for (int i=0; i<numInternalFaces; i++)	// loop over internal faces
        for (int i = 0; i < numFaces; i++) // loop over all faces (internal + boundary)
        {
            cellFacesOwner[6 * owner[i] + cellNumFacesOwner[owner[i]]] = i;
            cellNumFacesOwner[owner[i]]++;
            if (neighbour[i] != -1)
            {
                cellFacesNeighbour[6 * neighbour[i] + cellNumFacesNeighbour[neighbour[i]]] = i;
                cellNumFacesNeighbour[neighbour[i]]++;
            }
        }

        // now we know for each cell:
        // face number and the number of faces for which we are owner
        // face number and the number of faces for which we are neighbour
        cf cellFaces[6]; // will contain int .faceNr (faces of the cell in correct order), int .otherCell (the neighbour cell)

        cellPolys_.setSize(numCells);
        for (int i = 0; i < numCells; i++) // loop over cells
        {
            //		int debug = 0;
            //		if ( (i<10) && (debug==0) ) fprintf(stderr,"i=%d\n", i);
            int numCellFaces = cellNumFacesOwner[i] + cellNumFacesNeighbour[i]; // number of faces of our cell (5: boundary cell, 6: internal cell)
            //		if ( (i<10) && (debug==0) ) fprintf(stderr,"cellNumFacesOwner[%d]=%d\n", i, cellNumFacesOwner[i]);
            int numfacepos = 0;
            for (int j = 0; j < cellNumFacesOwner[i]; j++)
            {
                cellFaces[numfacepos].faceNr = cellFacesOwner[6 * i + j];
                cellFaces[numfacepos].otherCell = neighbour[cellFaces[numfacepos].faceNr];
                numfacepos++;
            }
            //		if ( (i<10) && (debug==0) ) fprintf(stderr,"cellNumFacesNeighbour[%d]=%d\n", i, cellNumFacesNeighbour[i]);
            for (int j = 0; j < cellNumFacesNeighbour[i]; j++)
            {
                cellFaces[numfacepos].faceNr = cellFacesNeighbour[6 * i + j];
                cellFaces[numfacepos].otherCell = owner[cellFaces[numfacepos].faceNr];
                numfacepos++;
            }
            //		if ( (i<10) && (debug==0) ) fprintf(stderr,"numfacepos=%d\n", numfacepos);
            // now sort cellFaces in ascending order of cellFacesOtherCell ...
            qsort(&cellFaces[0], numCellFaces, sizeof(cf), compare);
            labelList myFaces(numCellFaces);
            forAll(myFaces, j)
            {
                //			if ( (i<10) && (debug==0) ) fprintf(stderr,"cell %d has face %d, neighbour is %d\n", i, cellFaces[j].faceNr, cellFaces[j].otherCell);
                myFaces[j] = cellFaces[j].faceNr;
            }
            cell myCell(myFaces);
            cellPolys_[i] = myCell;
        }

        // construct polyMesh: boundary faces are already added here, patches later!
        fvMesh pMesh(
            IOobject(
                polyMesh::defaultRegion,
                runTime.constant(),
                runTime),
            xferMove(points_),
            xferMove(meshFaces_),
            xferMove(cellPolys_));

        delete[] cellFacesOwner;
        delete[] cellNumFacesOwner;
        delete[] cellFacesNeighbour;
        delete[] cellNumFacesNeighbour;

        delete[] owner;
        delete[] neighbour;

        // --- Receive boundary conditions ---
        Info << "\nReceiving boundary conditions...\n";

        int bocoNums[10];
        fprintf(stderr, "coviseToFoam: receiving %d bocoNums\n", 10);
        recvData(bocoNums, 10 * sizeof(int));
        /*
	bocoNums[0] .. num blade faces
	bocoNums[1] .. num hubrot faces
	bocoNums[2] .. num hubnonrot faces
	bocoNums[3] .. num shroudrot faces
	bocoNums[4] .. num shroudnonrot faces
	bocoNums[5] .. num inlet faces
	bocoNums[6] .. num outlet faces
	bocoNums[7] .. num per1 faces
	bocoNums[8] .. num per2 faces
	bocoNums[9] .. num 3D Elements (cells)
*/

        int num_blade_faces = bocoNums[0];
        int num_hubrot_faces = bocoNums[1];
        int num_hubnonrot_faces = bocoNums[2];
        int num_shroudrot_faces = bocoNums[3];
        int num_shroudnonrot_faces = bocoNums[4];
        int num_inlet_faces = bocoNums[5];
        int num_outlet_faces = bocoNums[6];
        int num_per1_faces = bocoNums[7];
        int num_per2_faces = bocoNums[8];
        int num_cells = bocoNums[9];

        std::cerr << "\tnum blade faces: " << num_blade_faces << std::endl;
        std::cerr << "\tnum hubrot faces: " << num_hubrot_faces << std::endl;
        std::cerr << "\tnum hubnonrot faces: " << num_hubnonrot_faces << std::endl;
        std::cerr << "\tnum shroudrot faces: " << num_shroudrot_faces << std::endl;
        std::cerr << "\tnum shroudnonrot faces: " << num_shroudnonrot_faces << std::endl;
        std::cerr << "\tnum inlet faces: " << num_inlet_faces << std::endl;
        std::cerr << "\tnum outlet faces: " << num_outlet_faces << std::endl;
        std::cerr << "\tnum cyclic1 faces: " << num_per1_faces << std::endl;
        std::cerr << "\tnum cyclic2 faces: " << num_per2_faces << std::endl;
        std::cerr << "\tnum cells: " << num_cells << std::endl;

        const int npatch = 9; // blade, hubrot, hubnonrot, shroudrot, shroudnonrot, inlet, outlet, per1, per2

        Foam::List<Foam::polyPatch *> pPatches(npatch);

        wordList patchNames(npatch);
        wordList patchTypes(npatch);
        word defaultFacesName = "defaultFaces";
        word defaultFacesType = wallPolyPatch::typeName;
        wordList patchPhysicalTypes(npatch);

        Foam::polyBoundaryMesh bm(
            //- Construct given size: const IOobject&, const polyMesh&, const label size,
            IOobject(
                polyMesh::defaultRegion,
                runTime.constant(),
                runTime),
            pMesh,
            npatch);

        wordList patchType(npatch);
        wordList patchName(npatch);
        labelList patchSize(npatch);
        labelList patchStart(npatch);
        labelList patchIndex(npatch);
        wordList patchBoundaryPTypes(npatch);
        wordList patchBoundaryUTypes(npatch);
        wordList patchBoundaryKTypes(npatch);
        wordList patchBoundaryEpsTypes(npatch);

        // build patches
        int pos = numInternalFaces;
        for (int i = 0; i < npatch; i++)
        {
            patchStart[i] = pos;
            patchIndex[i] = i;

            switch (i)
            {
            case 0:
                patchName[i] = "blade";
                patchType[i] = "wall";
                patchSize[i] = num_blade_faces;
                patchBoundaryPTypes[i] = "zeroGradient";
                patchBoundaryUTypes[i] = "fixedValue";
                patchBoundaryKTypes[i] = "zeroGradient";
                patchBoundaryEpsTypes[i] = "zeroGradient";
                break;
            case 1:
                patchName[i] = "hubrot";
                patchType[i] = "wall";
                patchSize[i] = num_hubrot_faces;
                patchBoundaryPTypes[i] = "zeroGradient";
                patchBoundaryUTypes[i] = "fixedValue";
                patchBoundaryKTypes[i] = "zeroGradient";
                patchBoundaryEpsTypes[i] = "zeroGradient";
                break;
            case 2:
                patchName[i] = "hubnonrot";
                patchType[i] = "wall";
                patchSize[i] = num_hubnonrot_faces;
                patchBoundaryPTypes[i] = "zeroGradient";
                patchBoundaryUTypes[i] = "fixedValue";
                patchBoundaryKTypes[i] = "zeroGradient";
                patchBoundaryEpsTypes[i] = "zeroGradient";
                break;
            case 3:
                patchName[i] = "shroudrot";
                patchType[i] = "wall";
                patchSize[i] = num_shroudrot_faces;
                patchBoundaryPTypes[3] = "zeroGradient";
                patchBoundaryUTypes[3] = "fixedValue";
                patchBoundaryKTypes[3] = "zeroGradient";
                patchBoundaryEpsTypes[3] = "zeroGradient";
                break;
            case 4:
                patchName[i] = "shroudnonrot";
                patchType[i] = "wall";
                patchSize[i] = num_shroudnonrot_faces;
                patchBoundaryPTypes[i] = "zeroGradient";
                patchBoundaryUTypes[i] = "zeroGradient";
                patchBoundaryKTypes[i] = "zeroGradient";
                patchBoundaryEpsTypes[i] = "zeroGradient";
                break;
            case 5:
                patchName[i] = "inlet";
                patchType[i] = "patch";
                patchSize[i] = num_inlet_faces;
                patchBoundaryPTypes[i] = "zeroGradient";
                patchBoundaryUTypes[i] = "fixedValue";
                patchBoundaryKTypes[i] = "turbulentIntensityKineticEnergyInlet";
                patchBoundaryEpsTypes[i] = "turbulentMixingLengthDissipationRateInlet";
                break;
            case 6:
                patchName[i] = "outlet";
                patchType[i] = "patch";
                patchSize[i] = num_outlet_faces;
                patchBoundaryPTypes[i] = "fixedMeanValue";
                patchBoundaryUTypes[i] = "inletOutlet";
                patchBoundaryKTypes[i] = "zeroGradient";
                patchBoundaryEpsTypes[i] = "zeroGradient";
                break;
            case 7:
                patchName[i] = "cyclic1";
                patchType[i] = "patch";
                patchSize[i] = num_per1_faces;
                patchBoundaryPTypes[i] = "cyclicGgi";
                patchBoundaryUTypes[i] = "cyclicGgi";
                patchBoundaryKTypes[i] = "cyclicGgi";
                patchBoundaryEpsTypes[i] = "cyclicGgi";
                break;
            case 8:
                patchName[i] = "cyclic2";
                patchType[i] = "patch";
                patchSize[i] = num_per2_faces;
                patchBoundaryPTypes[i] = "cyclicGgi";
                patchBoundaryUTypes[i] = "cyclicGgi";
                patchBoundaryKTypes[i] = "cyclicGgi";
                patchBoundaryEpsTypes[i] = "cyclicGgi";
                break;
            }
            if (i < 7)
            {
                pPatches[i] = Foam::polyPatch::New(
                                  patchType[i],
                                  patchName[i],
                                  patchSize[i],
                                  patchStart[i],
                                  i,
                                  pMesh.boundaryMesh()).ptr();
            }
            else
            {
                if (i == 7)
                {
                    Foam::cyclicGgiPolyPatch *cyclicGgiSide1 = new Foam::cyclicGgiPolyPatch(
                        patchName[i],
                        patchSize[i],
                        patchStart[i],
                        3,
                        pMesh.boundaryMesh(),
                        //bm,
                        "cyclic2",
                        "cyclic1_zone",
                        false,
                        Foam::vector(0, 0, 0),
                        Foam::vector(0, 0, 1),
                        Foam::scalar(360.0 / USGNums[3]));
                    pPatches[i] = cyclicGgiSide1;
                }
                if (i == 8)
                {
                    Foam::cyclicGgiPolyPatch *cyclicGgiSide2 = new Foam::cyclicGgiPolyPatch(
                        patchName[i],
                        patchSize[i],
                        patchStart[i],
                        3,
                        pMesh.boundaryMesh(),
                        //bm,
                        "cyclic1",
                        "cyclic2_zone",
                        false,
                        Foam::vector(0, 0, 0),
                        Foam::vector(0, 0, 1),
                        Foam::scalar(-360.0 / USGNums[3]));
                    pPatches[i] = cyclicGgiSide2;
                }
            }

            pos += patchSize[i];

            fprintf(stderr, "\t%d: startFace %d, numFaces %d\n", i, patchStart[i], patchSize[i]);
        }

        pMesh.addFvPatches(pPatches);

        // Set the precision of the points data to 10
        IOstream::defaultPrecision(10);

        Info << "Writing polyMesh" << endl;
        pMesh.write();

        Info << "Creating initial state with boundary conditions" << endl;

        Info << "\nReceiving boundary conditions...\n";

        std::cerr << "\tNumber of inlet faces: " << num_inlet_faces << ", number of cells: " << num_cells << std::endl;

        //   1. velocity at inlet (on faces)
        float *Uinlet_values = new float[3 * num_inlet_faces];
        //fprintf(stderr,"coviseToFoam: receiving %d Uinlet_values\n", 3*num_inlet_faces);
        recvData(Uinlet_values, 3 * num_inlet_faces * sizeof(float));

        //   2. U initialization (on cells)
        float *UVolume_values = new float[3 * num_cells];
        //fprintf(stderr,"coviseToFoam: receiving %d UVolume_values\n", 3*numCells);
        recvData(UVolume_values, 3 * numCells * sizeof(float));

        //   3. p initialization (on cells)
        float *pVolume_values = new float[3 * num_cells];
        //fprintf(stderr,"coviseToFoam: receiving %d pVolume_values\n", numCells);
        recvData(pVolume_values, 1 * numCells * sizeof(float));

        volScalarField p(
            IOobject(
                "p",
                runTime.timeName(),
                pMesh,
                //IOobject::READ_IF_PRESENT,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pMesh,
            dimensionedScalar("p", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
            patchBoundaryPTypes);

        forAll(p.internalField(), iCell)
        {
            p.internalField()[iCell] = pVolume_values[iCell];
        }
        delete[] pVolume_values;

        volVectorField Urel(
            IOobject(
                "Urel",
                runTime.timeName(),
                pMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pMesh,
            dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector(0.0, 0.0, 0.0)),
            patchBoundaryUTypes);

        forAll(Urel.internalField(), iCell)
        {
            Urel.internalField()[iCell] = vector(UVolume_values[3 * iCell + 0], UVolume_values[3 * iCell + 1], UVolume_values[3 * iCell + 2]);
        }
        Urel.boundaryField() = vector(0.0, 0.0, 0.0);

        for (int iFace = 0; iFace < num_inlet_faces; ++iFace)
        {
            Urel.boundaryField()[5][iFace] = vector(Uinlet_values[3 * iFace + 0], Uinlet_values[3 * iFace + 1], Uinlet_values[3 * iFace + 2]);
        }

        delete[] UVolume_values;
        delete[] Uinlet_values;

        volScalarField k(
            IOobject(
                "k",
                runTime.timeName(),
                pMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pMesh,
            dimensionedScalar("k", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.375),
            patchBoundaryKTypes);

        volScalarField epsilon(
            IOobject(
                "epsilon",
                runTime.timeName(),
                pMesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            pMesh,
            dimensionedScalar("epsilon", dimensionSet(0, 2, -3, 0, 0, 0, 0), 60.0),
            patchBoundaryEpsTypes);

        Info << "Writing initial state" << endl;

        p.write();
        Urel.write();
        k.write();
        epsilon.write();

        Info << "Parallel Decomposition Dictionary\n" << endl;
        label nDomains = USGNums[4];
        {
            IOdictionary decompDict(
                IOobject(
                    "decomposeParDict",
                    runTime.time().system(),
                    word::null, // use region if non-standard
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE,
                    false));
            //decompDict.lookup("numberOfSubdomains") >> nDomains;
            primitiveEntry decompDictEntry("numberOfSubdomains", nDomains);
            decompDict.set(decompDictEntry);
            decompDict.regIOobject::write();
        }
    }

    Info << "End\n" << endl;

    return 0;
}

// ************************************************************************* //
