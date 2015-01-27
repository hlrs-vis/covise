/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <BC_T.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <osg/Vec3d>
#include <CoolEmAllClient.h>
#include <CoolEmAll.h>

using namespace std;

BC_T::BC_T(CoolEmAll *cc)
{
    cool = cc;
    file1.open((cc->getPathPrefix() + "/0/T").c_str());
}
BC_T::~BC_T()
{
    file1.close();
}
void BC_T::writeHeader()
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
    file1 << "   object      T;" << endl;
    file1 << "}" << endl;
    file1 << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << endl;
    file1 << endl;
    file1 << "dimensions     [0 0 0 1 0 0 0];" << endl;
    file1 << endl;
    file1 << "internalField  uniform 310;" << endl;
    file1 << endl;
    file1 << "boundaryField" << endl;
    file1 << "{" << endl;
    file1 << "        defaultFaces" << endl;
    file1 << "        {" << endl;
    file1 << "                type      empty;" << endl;
    file1 << "        }" << endl;
}

void BC_T::writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName)
{
    std::string stlFileName;
    std::string DEBBLevel;

    //dissipated power of a fan
    double fanPower = 6.0; //nominal power of a working fan
    double fanEfficiency = 0.4; //nominal efficiency of a fan
    double dissHeat; //dissipated heat of a running fan
    dissHeat = fanPower * (1 - fanEfficiency);

    //inlet temperature increase by fan
    double volumeFlow = cool->getCoolEmAllClient()->getValue(DataBase_Path, ProductInstanceReference->getUserValue("airflow_volume-sensor"));
    double temperature_inlet = cool->getAverageInletTemperature() + 273.15;
    double coolingTemperature = temperature_inlet;
    if (volumeFlow > 0)
    {
        coolingTemperature = dissHeat / (1.168 * volumeFlow * 1004) + temperature_inlet;
    }
    fprintf(stderr, "coolingTemperature %f\n", (float)coolingTemperature);
    fprintf(stderr, "temperature_inlet %f\n", (float)temperature_inlet);
    fprintf(stderr, "dissHeat %f\n", (float)dissHeat);

    DEBBLevel = ProductRevisionViewReference->getUserValue("DEBBLevel");

    int pos_slash = transformedSTLFileName.find_last_of("/");
    std::string patchName = transformedSTLFileName.substr(pos_slash + 1);
    int l1 = patchName.length();
    patchName = patchName.substr(0, l1 - 4) + "_patch";

    std::transform(DEBBLevel.begin(), DEBBLevel.end(), DEBBLevel.begin(), ::tolower);

    if (DEBBLevel == "heatsink")
    {
        double temperature_heatsink;
        double power = cool->getCoolEmAllClient()->getValue(DataBase_Path, ProductInstanceReference->getUserValue("power-sensor"));
        //std::cerr << "DataBase_Path: " << DataBase_Path << " ";
        //std::cerr << ProductInstanceReference->getUserValue("power-sensor") << std::endl;
        //cerr << "power-sensor: " <<  power << endl;
        if (power != -1)
        {
            //double temperature_inlet = cool->getCoolEmAllClient()->getValue(DataBase_Path,ProductInstanceReference->getUserValue("temp-sensor"));
            //
            if (temperature_inlet == -1)
            {
                temperature_inlet = 293.15;
            }
            fprintf(stderr, "flaeche %f\n", (float)cool->getFlaeche());
            double a = cool->getFlaeche();
            if (a == 0)
                a = 0.01;

            temperature_heatsink = (power + (17.9 * a * coolingTemperature)) / (17.9 * a);
        }
        else
            temperature_heatsink = 20;

        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform " << temperature_heatsink << ";" << endl;
        file1 << "        }" << endl;
    }
    else if (DEBBLevel == "inlet")
    {
        double temperature = cool->getCoolEmAllClient()->getValue(DataBase_Path, ProductInstanceReference->getUserValue("temperature-sensor"));
        //std::cerr << "temperature: " << temperature << " path: " << DataBase_Path<< " var: " <<ProductInstanceReference->getUserValue("temperature-sensor") << endl;
        if (temperature == -1)
        {
            temperature = 20;
        }
        temperature = temperature + 273.15; // Umrechnung von °C in K
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      fixedValue;" << endl;
        file1 << "                value     uniform " << temperature << ";" << endl;
        file1 << "        }" << endl;
    }

    else if (DEBBLevel == "outlet")
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      zeroGradient;" << endl;
        file1 << "        }" << endl;
    }
    else
    {
        file1 << "        " << patchName << endl;
        file1 << "        {" << endl;
        file1 << "                type      zeroGradient;" << endl;
        //file1 << "                type      fixedValue;" << endl;
        //file1 << "                value     uniform 293;" << endl;
        file1 << "        }" << endl;
    }
}

void BC_T::writeFooter()
{
    file1 << "}" << endl;
    file1 << endl;
    file1 << "// ************************************************************************* //" << endl;
}
