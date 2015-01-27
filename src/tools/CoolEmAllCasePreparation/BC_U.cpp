/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <BC_U.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <osg/Vec3d>
#include <CoolEmAllClient.h>
#include <CoolEmAll.h>

using namespace std;

BC_U::BC_U(CoolEmAll *cc)
{
    cool = cc;
    file1.open((cc->getPathPrefix() + "/0/U").c_str());
}
BC_U::~BC_U()
{
    file1.close();
}
void BC_U::writeHeader()
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
    file1 << "   class       volVectorField;" << endl;
    file1 << "   object      U;" << endl;
    file1 << "}" << endl;
    file1 << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << endl;
    file1 << endl;
    file1 << "dimensions     [0 1 -1 0 0 0 0];" << endl;
    file1 << endl;

    file1 << "internalField  uniform (0 0 0);" << endl;
    file1 << endl;
    file1 << "boundaryField" << endl;
    file1 << "{" << endl;
    file1 << "        defaultFaces" << endl;
    file1 << "        {" << endl;
    file1 << "                type      empty;" << endl;
    file1 << "        }" << endl;
}

void BC_U::writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName)
{
    std::string stlFileName;
    std::string DEBBLevel;

    DEBBLevel = ProductRevisionViewReference->getUserValue("DEBBLevel");

    int pos_slash = transformedSTLFileName.find_last_of("/");
    std::string patchName = transformedSTLFileName.substr(pos_slash + 1);
    int l1 = patchName.length();
    patchName = patchName.substr(0, l1 - 4) + "_patch";

    std::transform(DEBBLevel.begin(), DEBBLevel.end(), DEBBLevel.begin(), ::tolower);

    if (DEBBLevel == "inlet") //die Inlet-Geschwindigkeit muss vorzeichenrichtig geschrieben werden
    {
        ifstream stlFile;
        std::string to_open = cool->getPathPrefix() + "/constant/triSurface/" + transformedSTLFileName;
        //std::cout << "geoeffnete .stl-Datei: " << to_open << std::endl;
        stlFile.open(to_open.c_str());

        osg::Vec3d facetNormal;

        double velocity = cool->getCoolEmAllClient()->getValue(DataBase_Path, ProductInstanceReference->getUserValue("airflow_volume-sensor"));
        //std::cerr << "Velocity: " << velocity << " path: " << DataBase_Path<< " var: " <<ProductInstanceReference->getUserValue("airflow_volume-sensor") << endl;
        if (velocity == -1)
        {
            velocity = 0.001;
        }
        velocity = velocity / cool->getFlaeche();
        if (!stlFile.is_open())
        {
            std::cerr << "Could not open .stl-file!" << to_open << std::endl;
            exit(1);
        }
        else
        {
            std::string word_buffer;
            std::vector<string> word;

            while (stlFile.good())
            {
                std::string buffer1;
                std::getline(stlFile, buffer1);

                if (!stlFile.good())
                {
                    break;
                }
                std::istringstream is;
                is.str(buffer1);
                while (!is.eof())
                {
                    is >> word_buffer;
                    word.push_back(word_buffer);
                }
                //size_t i = word.size();
                int i = word.size();
                i = i - 1;
                if (i >= 4)
                {
                    if (word.at(i - 3) == "normal")
                    {
                        if (word.at(i - 4) == "facet")
                        {
                            std::istringstream iss2(word.at(i));
                            iss2 >> facetNormal[2];
                            std::istringstream iss3(word.at((i - 1)));
                            iss3 >> facetNormal[1];
                            std::istringstream iss4(word.at((i - 2)));
                            iss4 >> facetNormal[0];
                            break;
                        }
                    }
                }
            }
        }
        facetNormal = facetNormal * velocity;

        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform (" << facetNormal[0] << " " << facetNormal[1] << " " << facetNormal[2] << ");" << endl;
        file1 << "        }" << endl;
    }

    else if (DEBBLevel == "outlet")
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      		inletOutlet;" << endl;
        file1 << "                inletValue     	uniform (0 0 0);" << endl;
        file1 << "                value      		uniform (0 0 0);" << endl;
        file1 << "        }" << endl;
    }
    else
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform (0 0 0);" << endl;
        file1 << "        }" << endl;
    }
}

void BC_U::writeFooter()
{
    file1 << "}" << endl;
    file1 << endl;
    file1 << "// ************************************************************************* //" << endl;
}
