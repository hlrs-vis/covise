/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SnappyHexmeshDict.h"
#include <algorithm>
#include <sstream>
#include <list>
#include <set>
#include <vector>
#include <osg/Vec3d>
#include <CoolEmAllClient.h>
#include <CoolEmAll.h>
#include <string.h>
#include <string>
#include <fstream>
#include "checkForPoint.h"

using namespace std;

SnappyHexmeshDict::SnappyHexmeshDict(CoolEmAll *cc)
{
    cool = cc;
    file1.open((cc->getPathPrefix() + "/system/snappyHexMeshDict").c_str());
}
SnappyHexmeshDict::~SnappyHexmeshDict()
{
    file1.close();
}
void SnappyHexmeshDict::writeHeader()
{
    file1 << "/*--------------------------------*- C++ -*----------------------------------*\\" << endl;
    file1 << "| =========                 |                                                 |" << endl;
    file1 << "| \\\\      /  F ield         | OpenFOAM Extend Project: Open source CFD        |" << endl;
    file1 << "|  \\\\    /   O peration     | Version:  1.6-ext                               |" << endl;
    file1 << "|   \\\\  /    A nd           | Web:      www.extend-project.de                 |" << endl;
    file1 << "|    \\\\/     M anipulation  |                                                 |" << endl;
    file1 << "\\*---------------------------------------------------------------------------*/" << endl;
    file1 << "FoamFile" << endl;
    file1 << "{" << endl;
    file1 << "   version     2.0;" << endl;
    file1 << "   format      ascii;" << endl;
    file1 << "   class       dictionary;" << endl;
    file1 << "   object      snappyHexMeshDict;" << endl;
    file1 << "}" << endl;
    file1 << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << endl;
    file1 << endl;
    file1 << "castellatedMesh true;" << endl;
    file1 << "snap            true;" << endl;
    file1 << "addLayers       true;" << endl;

    file1 << endl;

    file1 << "geometry" << endl;
    file1 << "{" << endl;
}

void SnappyHexmeshDict::writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName)
{
    std::string DEBBLevel;
    DEBBLevel = ProductRevisionViewReference->getUserValue("DEBBLevel");
    std::transform(DEBBLevel.begin(), DEBBLevel.end(), DEBBLevel.begin(), ::tolower);

    std::string locationinmesh_buffer = ProductInstanceReference->getUserValue("locationInMesh");
    if (!locationinmesh_buffer.empty())
    {
        bool test = checkForPoint(locationinmesh_buffer);
        if (test == true)
        {
            locationinmesh = locationinmesh_buffer;
        }
        else
        {
            std::cerr << "no valid locationInMesh specified, please submit valid point" << std::endl;
            exit(1);
        }
    }

    std::string stlFileName;

    int pos_slash = transformedSTLFileName.find_last_of("/");
    std::string patchName = transformedSTLFileName.substr(pos_slash + 1);
    int l1 = patchName.length();
    patchName = patchName.substr(0, l1 - 4);

    if (DEBBLevel == "outlet")
    {
        PatchInfo pi;
        pi.keyword = DEBBLevel;
        pi.patchName = patchName + "_patch";
        pi.databasePath = DataBase_Path;
        patchAverageList.push_back(pi);
    }
    if (DEBBLevel == "inlet")
    {
        PatchInfo pi;
        pi.keyword = DEBBLevel;
        pi.patchName = patchName + "_patch";
        pi.databasePath = DataBase_Path;
        patchAverageList.push_back(pi);
    }
    patches.push_back(patchName);

    file1 << "    " << transformedSTLFileName << endl;
    file1 << "    {" << endl;
    file1 << "        type triSurfaceMesh;" << std::endl;
    file1 << "        name " << patchName << ";" << std::endl;
    file1 << "    }" << endl;

    DEBBLevels.push_back(DEBBLevel);
}

void SnappyHexmeshDict::writeFooter()
{
    std::string absolutePathToPatchFile = cool->getPathPrefix() + "/patchname.txt";
    ofstream patchAveragefile;
    patchAveragefile.open((absolutePathToPatchFile).c_str());
    for (it = patchAverageList.begin(); it != patchAverageList.end(); it++)
    {
        patchAveragefile << (*it).keyword << " " << (*it).patchName << std::endl;
        patchAveragefile << (*it).databasePath << std::endl;
    }
    patchAveragefile.close();

    file1 << "};" << endl;
    file1 << endl;
    file1 << "castellatedMeshControls" << endl;
    file1 << "{" << endl;
    file1 << "    maxLocalCells 30000000;" << std::endl;
    file1 << std::endl;
    file1 << "    maxGlobalCells 60000000;" << std::endl;
    file1 << std::endl;
    file1 << "    minRefinementCells 0;" << std::endl;
    file1 << std::endl;
    file1 << "    nCellsBetweenLevels 1;" << std::endl;
    file1 << std::endl;
    file1 << "    features" << std::endl;
    file1 << "    (" << std::endl;
    file1 << "    );" << std::endl;
    file1 << "    refinementSurfaces" << std::endl;
    file1 << "    {" << std::endl;
    file1 << std::endl;
    std::list<std::string>::iterator si = DEBBLevels.begin();
    for (patchesIt = patches.begin(); patchesIt != patches.end(); ++patchesIt)
    {
        if (*si == "outlet")
        {
            file1 << "        " << *patchesIt << std::endl;
            file1 << "        {" << std::endl;
            file1 << "            level (0 0);" << std::endl;
            file1 << "        }" << std::endl;
        }
        if (*si == "inlet")
        {
            file1 << "        " << *patchesIt << std::endl;
            file1 << "        {" << std::endl;
            file1 << "            level (0 0);" << std::endl;
            file1 << "        }" << std::endl;
        }
        else if (*si == "heatsink")
        {
            file1 << "        " << *patchesIt << std::endl;
            file1 << "        {" << std::endl;
            file1 << "            level (2 3);" << std::endl;
            file1 << "        }" << std::endl;
        }
        else
        {
            file1 << "        " << *patchesIt << std::endl;
            file1 << "        {" << std::endl;
            file1 << "            level (1 2);" << std::endl;
            file1 << "        }" << std::endl;
        }
        si++;
    }

    file1 << "    }" << std::endl;
    file1 << "    resolveFeatureAngle 30;" << std::endl;
    file1 << "    refinementRegions" << std::endl;
    file1 << "    {" << std::endl;
    si = DEBBLevels.begin();
    for (patchesIt = patches.begin(); patchesIt != patches.end(); ++patchesIt)
    {
        if (*si == "heatsink")
        {
            file1 << "        " << *patchesIt << std::endl;
            file1 << "        {" << std::endl;
            file1 << "            mode distance;" << std::endl;
            file1 << "            levels ((0.001 3) (0.003 2));" << std::endl;
            file1 << "        }" << std::endl;
        }
        si++;
    }

    file1 << "    }" << std::endl;

    if (locationinmesh.empty())
    {
        std::cerr << "no locationInMesh submitted, please submit locationInMesh as point" << std::endl;
        exit(1);
    }
    file1 << "    locationInMesh (" << locationinmesh << ");" << std::endl;
    file1 << "}" << std::endl;
    file1 << std::endl;
    file1 << "snapControls" << std::endl;
    file1 << "{" << std::endl;
    file1 << "    nSmoothPatch 3;" << std::endl;
    file1 << std::endl;
    file1 << "    tolerance 4.0;" << std::endl;
    file1 << std::endl;
    file1 << "    nSolveIter 30;" << std::endl;
    file1 << std::endl;
    file1 << "    nRelaxIter 5;" << std::endl;
    file1 << "}" << std::endl;
    file1 << std::endl;
    file1 << "addLayersControls" << std::endl;
    file1 << "{" << std::endl;
    file1 << "    relativeSizes true;" << std::endl;
    file1 << "    layers" << std::endl;
    file1 << "    {" << std::endl;
    file1 << "    }" << std::endl;
    file1 << std::endl;
    file1 << "    expansionRatio 1.3;" << std::endl;
    file1 << std::endl;
    file1 << "    finalLayerThickness 0.7;" << std::endl;
    file1 << std::endl;
    file1 << "    minThickness 0.25;" << std::endl;
    file1 << std::endl;
    file1 << "    nGrow 0;" << std::endl;
    file1 << std::endl;
    file1 << "    featureAngle 60;" << std::endl;
    file1 << std::endl;
    file1 << "    nRelaxIter 5;" << std::endl;
    file1 << std::endl;
    file1 << "    nSmoothSurfaceNormals 1;" << std::endl;
    file1 << std::endl;
    file1 << "    nSmoothNormals 3;" << std::endl;
    file1 << std::endl;
    file1 << "    nSmoothThickness 10;" << std::endl;
    file1 << std::endl;
    file1 << "    maxFaceThicknessRatio 0.5;" << std::endl;
    file1 << std::endl;
    file1 << "    maxThicknessToMedialRatio 0.3;" << std::endl;
    file1 << std::endl;
    file1 << "    minMedianAxisAngle 90;" << std::endl;
    file1 << std::endl;
    file1 << "    nBufferCellsNoExtrude 0;" << std::endl;
    file1 << std::endl;
    file1 << "    nLayerIter 50;" << std::endl;
    file1 << "}" << std::endl;
    file1 << std::endl;
    file1 << "meshQualityControls" << std::endl;
    file1 << "{" << std::endl;
    file1 << std::endl;
    file1 << "    maxNonOrtho 65;" << std::endl;
    file1 << std::endl;
    file1 << "    maxBoundarySkewness 20;" << std::endl;
    file1 << "    maxInternalSkewness 4;" << std::endl;
    file1 << std::endl;
    file1 << "    maxConcave 80;" << std::endl;
    file1 << std::endl;
    file1 << "    minFlatness 0.5;" << std::endl;
    file1 << std::endl;
    file1 << "    minVol 1e-13;" << std::endl;
    file1 << std::endl;
    file1 << "    minTetVol 1e-20;" << std::endl;
    file1 << std::endl;
    file1 << "    minArea -1;" << std::endl;
    file1 << std::endl;
    file1 << "    minTwist 0.05;" << std::endl;
    file1 << std::endl;
    file1 << "    minDeterminant 0.001;" << std::endl;
    file1 << std::endl;
    file1 << "    minFaceWeight 0.05;" << std::endl;
    file1 << std::endl;
    file1 << "    minVolRatio 0.01;" << std::endl;
    file1 << std::endl;
    file1 << "    minTriangleTwist -1;" << std::endl;
    file1 << std::endl;
    file1 << "    nSmoothScale 4;" << std::endl;
    file1 << std::endl;
    file1 << "    errorReduction 0.75;" << std::endl;
    file1 << "}" << std::endl;
    file1 << std::endl << std::endl;
    file1 << "debug 0;" << std::endl;
    file1 << std::endl;
    file1 << "mergeTolerance 1E-6;" << std::endl;
    file1 << std::endl;

    file1 << endl;
    file1 << "// ************************************************************************* //" << endl;
}
