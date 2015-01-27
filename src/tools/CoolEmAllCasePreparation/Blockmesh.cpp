/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <Blockmesh.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <osg/Vec3d>
#include <CoolEmAllClient.h>
#include <CoolEmAll.h>

using namespace std;

Blockmesh::Blockmesh(CoolEmAll *cc)
{
    cool = cc;
    file1.open((cc->getPathPrefix() + "/constant/polyMesh/blockMeshDict").c_str());
}
Blockmesh::~Blockmesh()
{
    file1.close();
}
void Blockmesh::writeHeader()
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
    file1 << "   object      blockMeshDict;" << endl;
    file1 << "}" << endl;
    file1 << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << endl;
    file1 << endl;
    file1 << "convertToMeters     1;" << endl;
    file1 << endl;
}

void Blockmesh::writeBound(FileReference *ProductRevisionViewReference)
{
    std::string boundingBox = ProductRevisionViewReference->getBound();
    std::string MeshResolution = ProductRevisionViewReference->getUserValue("MeshResolution");
    if ((boundingBox.length() > 0) && (MeshResolution.length() > 0))
    {
        float xmin, ymin, zmin, xmax, ymax, zmax;
        int xdim, ydim, zdim;

        sscanf(boundingBox.c_str(), "%f %f %f %f %f %f", &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        sscanf(MeshResolution.c_str(), "%d %d %d", &xdim, &ydim, &zdim);
        file1 << "vertices" << endl;
        file1 << "(" << endl;
        file1 << "  (" << xmin << " " << ymin << " " << zmin << ")" << endl;
        file1 << "  (" << xmax << " " << ymin << " " << zmin << ")" << endl;
        file1 << "  (" << xmax << " " << ymax << " " << zmin << ")" << endl;
        file1 << "  (" << xmin << " " << ymax << " " << zmin << ")" << endl;
        file1 << "  (" << xmin << " " << ymin << " " << zmax << ")" << endl;
        file1 << "  (" << xmax << " " << ymin << " " << zmax << ")" << endl;
        file1 << "  (" << xmax << " " << ymax << " " << zmax << ")" << endl;
        file1 << "  (" << xmin << " " << ymax << " " << zmax << ")" << endl;
        file1 << ");" << endl;
        file1 << "blocks" << endl;
        file1 << "(" << endl;
        file1 << "  hex (0 1 2 3 4 5 6 7) (" << xdim << " " << ydim << " " << zdim << ") simpleGrading (1 1 1) " << endl;
        file1 << ");" << endl;
        file1 << "edges" << endl;
        file1 << "(" << endl;
        file1 << ");" << endl;
        file1 << "" << endl;
        file1 << "patches" << endl;
        file1 << "(" << endl;
        file1 << ");" << endl;
        file1 << "" << endl;
        file1 << "mergePatchPairs" << endl;
        file1 << "(" << endl;
        file1 << ");" << endl;
    }
}

void Blockmesh::writeFooter()
{
    file1 << endl;
    file1 << "// ************************************************************************* //" << endl;
}
