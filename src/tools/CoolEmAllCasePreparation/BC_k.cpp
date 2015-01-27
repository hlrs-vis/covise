/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <BC_k.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <osg/Vec3d>
#include <CoolEmAllClient.h>
#include <CoolEmAll.h>

using namespace std;

BC_k::BC_k(CoolEmAll *cc)
{
    cool = cc;
    file1.open((cc->getPathPrefix() + "/0/k").c_str());
}
BC_k::~BC_k()
{
    file1.close();
}
void BC_k::writeHeader()
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
    file1 << "   class       volScalarField;" << endl;
    file1 << "   location    \"0\";" << endl;
    file1 << "   object      k;" << endl;
    file1 << "}" << endl;
    file1 << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << endl;
    file1 << endl;
    file1 << "dimensions     [0 2 -2 0 0 0 0];" << endl;
    file1 << endl;
    file1 << "internalField  uniform 0.1;" << endl;
    file1 << endl;
    file1 << "boundaryField" << endl;
    file1 << "{" << endl;
    file1 << "        defaultFaces" << endl;
    file1 << "        {" << endl;
    file1 << "                type      empty;" << endl;
    file1 << "        }" << endl;
}

void BC_k::writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName)
{
    std::string stlFileName;
    std::string DEBBLevel;

    DEBBLevel = ProductRevisionViewReference->getUserValue("DEBBLevel");

    int pos_slash = transformedSTLFileName.find_last_of("/");
    std::string patchName = transformedSTLFileName.substr(pos_slash + 1);
    int l1 = patchName.length();
    patchName = patchName.substr(0, l1 - 4) + "_patch";

    std::transform(DEBBLevel.begin(), DEBBLevel.end(), DEBBLevel.begin(), ::tolower);

    if (DEBBLevel == "inlet")
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform 0.375;" << endl;
        file1 << "        }" << endl;
    }

    else if (DEBBLevel == "outlet")
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform 0.375;" << endl;
        file1 << "        }" << endl;
    }

    else
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      kqRWallFunction;" << endl;
        file1 << "                value     uniform 0.375;" << endl;
        file1 << "        }" << endl;
    }
}

void BC_k::writeFooter()
{
    file1 << "}" << endl;
    file1 << endl;
    file1 << "// ************************************************************************* //" << endl;
}
