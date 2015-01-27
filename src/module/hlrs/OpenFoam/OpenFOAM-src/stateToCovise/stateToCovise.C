/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by original author
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

Author
    Hrvoje Jasak, Wikki Ltd.  All rights reserved

\*----------------------------------------------------------------------------*/

#include "stateToCovise.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "bound.H"
#include "coSimClient.H"
#include <vector>
#include <typeinfo>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
defineTypeNameAndDebug(stateToCovise, 0);

addToRunTimeSelectionTable(
    functionObject,
    stateToCovise,
    dictionary);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::stateToCovise::stateToCovise(
    const word &name,
    const Time &t,
    const dictionary &dict)
    : functionObject("stateToCovise")
    , cellProcAddressing(NULL)
    , name_(name)
    , time_(t)
    , regionName_(polyMesh::defaultRegion)
    ,
    //fieldName_(dict.lookup("name")),
    sendCPUTimeStep_(readScalar(dict.lookup("sendCPUTimeStep")))
    , lastCPUTime_(0.0)
    , pFieldName_("p")
    , uFieldName_(dict.lookup("Uname"))
{
    if (dict.found("region"))
    {
        dict.lookup("region") >> regionName_;
    }

    //Info<< "Creating stateToCovise for field "
    //    << fieldName_ << endl;

    std::cout << "Hello stateToCovise::stateToCovise()!!!" << std::endl;

    /*fieldName_ = word(dict.lookup("name"));
    minValue_ = readScalar(dict.lookup("minValue"));
    maxValue_ = readScalar(dict.lookup("maxValue"));

    if (dict.found("region"))
    {
        dict.lookup("region") >> regionName_;
    }*/
    if (cellProcAddressing)
    {
        delete cellProcAddressing;
        cellProcAddressing = NULL;
    }

    const fvMesh &mesh = time_.lookupObject<fvMesh>(regionName_);

    //if (mesh.globalData().parallel())
    if (Pstream::parRun())
    {
        cellProcAddressing = new labelIOList(
            IOobject(
                "cellProcAddressing",
                mesh.facesInstance(),
                mesh.meshSubDir,
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE));
        Info << "Loaded cell processor addressing!" << endl;
    }
    else
    {
        cellProcAddressing = new labelList(mesh.nCells());
        for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
        {
            (*cellProcAddressing)[iCell] = iCell;
        }
        //Info << "No distributed processor patches!" << endl;
        Info << "No parallel run! Created cell processor addressing for master (global to global)..." << endl;
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::stateToCovise::exitOnCoviseConnectionLoss()
{
    FatalErrorIn("stateToCovise") << " Lost connection to covise, aborting simulation!\n" << abort(FatalError);
}

bool Foam::stateToCovise::start()
{
    std::cout << "Hello stateToCovise::start()!!!" << std::endl;

    if (Pstream::master()) //---I'm the master---
    {
        if (coNotConnected() < 0)
        {
            Info << "\nConnecting to covise...\n";
            if (coInitConnect() != 0)
            {
                Info << " not connected, exit!" << endl;
                return false;
            };
            Info << " connected!" << endl;
        }

        Info << "\nParallel initialisation...\n";

        int parErr = 0;
        parErr += coParallelInit(Pstream::nProcs(), 1);

        //pressure
        parErr += coParallelPort("pData", 1);
        //velocity
        parErr += coParallelPort("uData", 1);

        //Only OpenFoam-1.7:
        //parErr += coParallelCellMap(0, cellProcAddressing->size(), cellProcAddressing->data());
        //OpenFoam-1.5-dev:
        parErr += coParallelCellMap(0, cellProcAddressing->size(), &((*cellProcAddressing)[0]));

        for (int slave = Pstream::firstSlave(); slave <= Pstream::lastSlave(); slave++)
        {
            IPstream fromSlave(Pstream::scheduled, slave);
            int cellProcAddressingSlaveSize;
            fromSlave >> cellProcAddressingSlaveSize;

            labelList cellProcAddressingSlave(cellProcAddressingSlaveSize);
            //Only OpenFoam-1.7:
            //fromSlave.read(reinterpret_cast<char*>(cellProcAddressingSlave.data()), cellProcAddressingSlaveSize*sizeof(Foam::label));
            //OpenFoam-1.5-dev:
            fromSlave.read(reinterpret_cast<char *>(&(cellProcAddressingSlave[0])), cellProcAddressingSlaveSize * sizeof(Foam::label));

            //Only OpenFoam-1.7:
            //parErr += coParallelCellMap(slave, cellProcAddressingSlave.size(), cellProcAddressingSlave.data());
            //OpenFoam-1.5-dev:
            parErr += coParallelCellMap(slave, cellProcAddressingSlave.size(), &(cellProcAddressingSlave[0]));
        }

        if (parErr != 0)
        {
            Info << " not successful, exit!" << endl;
            return false;
        };
        Info << " success!" << endl;
    }
    else //---I'm a slave---
    {
        OPstream toMaster(Pstream::scheduled, Pstream::masterNo());
        toMaster << cellProcAddressing->size();
        //Only OpenFoam-1.7:
        //toMaster.write(reinterpret_cast<char*>(cellProcAddressing->data()), cellProcAddressing->size()*sizeof(Foam::label));
        //OpenFoam-1.5-dev:
        toMaster.write(reinterpret_cast<char *>(&((*cellProcAddressing)[0])), cellProcAddressing->size() * sizeof(Foam::label));
    }

    lastCPUTime_ = time_.elapsedCpuTime();

    return true;
}

bool Foam::stateToCovise::execute()
{
    bool sendState = false;
    if (Pstream::master()) //---I'm the master---
    {
        scalar currentCPUTime = time_.elapsedCpuTime();
        if (currentCPUTime > lastCPUTime_ + sendCPUTimeStep_)
        {
            lastCPUTime_ += sendCPUTimeStep_;
            sendState = true;
        }
    }

    Pstream::scatter(sendState);

    if (sendState == false)
    {
        return true;
    }

    std::cout << "Hello stateToCovise::execute()!!!" << std::endl;

    const fvMesh &mesh = time_.lookupObject<fvMesh>(regionName_);

    //pressure
    volScalarField &p = const_cast<volScalarField &>(
        mesh.lookupObject<volScalarField>(pFieldName_));

    //velocity
    volVectorField &U = const_cast<volVectorField &>(
        mesh.lookupObject<volVectorField>(uFieldName_));

    /*boundMinMax
      (
       f,
       dimensionedScalar("v", f.dimensions(), minValue_),
       dimensionedScalar("V", f.dimensions(), maxValue_)
      );*/

    if (Pstream::master()) //---I'm the master---
    {
        //pressure
        {
            std::vector<float> p_float(mesh.nCells());
            for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
            {
                p_float[iCell] = p[iCell];
            }
            int sendErr = 0;

            sendErr = coParallelNode(0);
            sendErr = coSend1Data("pData", p_float.size(), &(p_float[0]));
            if (sendErr != 0)
            {
                Info << "Lost connection to covise, exit!" << endl;
                exitOnCoviseConnectionLoss();
                return false;
            };
            Info << "Master: Pressure data sent to covise!" << endl;
        }

        //      //temperature
        //      if(haveTemperature)
        //      {
        //         std::vector<float> p_float(mesh.nCells());
        //         for (int iCell = 0; iCell<mesh.nCells(); ++iCell)
        //         {
        //            p_float[iCell] = p[iCell];
        //         }
        //         int sendErr = 0;
        //
        //         sendErr = coParallelNode(0);
        //         sendErr = coSend1Data("TData", p_float.size(), &(p_float[0]));
        //         if (sendErr!=0)
        //         {
        //            Info<< "Lost connection to covise, exit!" << endl;
        //            exitOnCoviseConnectionLoss();
        //            return false;
        //         };
        //         Info<< "Master: Temperature data sent to covise!" << endl;
        //      }

        //velocity
        {
            std::vector<float> u_float_1(mesh.nCells());
            std::vector<float> u_float_2(mesh.nCells());
            std::vector<float> u_float_3(mesh.nCells());
            for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
            {
                u_float_1[iCell] = U[iCell][0];
                u_float_2[iCell] = U[iCell][1];
                u_float_3[iCell] = U[iCell][2];
            }
            int sendErr = 0;
            sendErr = coParallelNode(0);
            sendErr = coSend3Data("uData", mesh.nCells(), &(u_float_1[0]), &(u_float_2[0]), &(u_float_3[0]));
            if (sendErr != 0)
            {
                Info << "Lost connection to covise, exit!" << endl;
                exitOnCoviseConnectionLoss();
                return false;
            };
            Info << "Master: Velocity data sent to covise!" << endl;
        }

        for (int slave = Pstream::firstSlave(); slave <= Pstream::lastSlave(); slave++)
        {
            //cell-based
            /*std::vector<float> p_float(numCellPerSubdomain);
           for (int iCell = 0; iCell<numCellPerSubdomain; ++iCell)
           {
           p_float[iCell] = p[numCellPerSubdomain*iSubdomain + iCell];
           }*/
            //Pstream::scheduled uses MPI_Send(), Pstream::blocking MPI_Bsend(), Pstream::nonBlocking MPI_Isend()
            IPstream fromSlave(Pstream::scheduled, slave);

            //pressure
            int p_nCells;
            fromSlave >> p_nCells;
            std::vector<float> p_float(p_nCells);
            fromSlave.read(reinterpret_cast<char *>(&(p_float[0])), p_float.size() * sizeof(float));

            int sendErr = 0;

            sendErr = coParallelNode(slave);
            sendErr = coSend1Data("pData", p_nCells, &(p_float[0]));
            p_float.resize(0);

            //velocity
            int u_nCells;
            fromSlave >> u_nCells;
            std::vector<float> u_float_1(u_nCells);
            std::vector<float> u_float_2(u_nCells);
            std::vector<float> u_float_3(u_nCells);
            fromSlave.read(reinterpret_cast<char *>(&(u_float_1[0])), u_float_1.size() * sizeof(float));
            fromSlave.read(reinterpret_cast<char *>(&(u_float_2[0])), u_float_2.size() * sizeof(float));
            fromSlave.read(reinterpret_cast<char *>(&(u_float_3[0])), u_float_3.size() * sizeof(float));

            //int sendErr = 0;

            //sendErr = coParallelNode(slave);
            sendErr = coSend3Data("uData", u_nCells, &(u_float_1[0]), &(u_float_2[0]), &(u_float_3[0]));
            u_float_1.resize(0);
            u_float_2.resize(0);
            u_float_3.resize(0);

            if (sendErr != 0)
            {
                Info << "Lost connection to covise, exit!" << endl;
                exitOnCoviseConnectionLoss();
                return false;
            };
            Info << "Slave/Subdomain " << slave << ": Data sent to covise!" << endl;
        }

        if (coFinished() != 0)
        {
            Info << "Lost connection to covise, exit!" << endl;
            exitOnCoviseConnectionLoss();
            return false;
        };
        Info << "Finished message sent to covise!" << endl;

        if (coExecModule() != 0)
        {
            Info << "Lost connection to covise, exit!" << endl;
            exitOnCoviseConnectionLoss();
            return false;
        };
        Info << "Execute module message sent to covise!" << endl;

        if (coNotConnected() != 0)
        {
            Info << "Lost connection to covise, exit!" << endl;
            exitOnCoviseConnectionLoss();
            return false;
        }
    }
    else //---I'm a slave---
    {
        OPstream toMaster(Pstream::scheduled, Pstream::masterNo());

        //pressure
        std::vector<float> p_float(mesh.nCells());
        for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
        {
            p_float[iCell] = p[iCell];
        }

        int p_float_size = p_float.size();
        toMaster << p_float_size;
        toMaster.write(reinterpret_cast<char *>(&(p_float[0])), p_float.size() * sizeof(float));
        p_float.resize(0);

        //velocity
        int u_float_size = mesh.nCells();
        toMaster << u_float_size;

        std::vector<float> u_float_1(mesh.nCells());
        for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
        {
            u_float_1[iCell] = U[iCell][0];
        }
        toMaster.write(reinterpret_cast<char *>(&(u_float_1[0])), u_float_1.size() * sizeof(float));
        u_float_1.resize(0);

        std::vector<float> u_float_2(mesh.nCells());
        for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
        {
            u_float_2[iCell] = U[iCell][1];
        }
        toMaster.write(reinterpret_cast<char *>(&(u_float_2[0])), u_float_2.size() * sizeof(float));
        u_float_2.resize(0);

        std::vector<float> u_float_3(mesh.nCells());
        for (int iCell = 0; iCell < mesh.nCells(); ++iCell)
        {
            u_float_3[iCell] = U[iCell][2];
        }
        toMaster.write(reinterpret_cast<char *>(&(u_float_3[0])), u_float_3.size() * sizeof(float));
        u_float_3.resize(0);
    }

    return true;
}

bool Foam::stateToCovise::read(const dictionary &dict)
{
    std::cout << "Hello stateToCovise::read()!!!" << std::endl;
#if 0
    /*fieldName_ = word(dict.lookup("name"));
    minValue_ = readScalar(dict.lookup("minValue"));
    maxValue_ = readScalar(dict.lookup("maxValue"));

    if (dict.found("region"))
    {
        dict.lookup("region") >> regionName_;
    }*/
   if(cellProcAddressing) {
      delete cellProcAddressing;
      cellProcAddressing = NULL;
   }
    
   const fvMesh& mesh =
   time_.lookupObject<fvMesh>(regionName_);

   //if (mesh.globalData().parallel())
   if (Pstream::parRun())
   {
      cellProcAddressing = new labelIOList 
         (
          IOobject
          (
           "cellProcAddressing",
           mesh.facesInstance(),
           mesh.meshSubDir,
           mesh,
           IOobject::MUST_READ,
           IOobject::NO_WRITE
          )
         );
      Info << "Loaded cell processor addressing!" << endl;

   }
   else
   {
      cellProcAddressing = new labelList(mesh.nCells());
      for(int iCell = 0; iCell<mesh.nCells(); ++iCell)
      {
         (*cellProcAddressing)[iCell] = iCell;
      }
      //Info << "No distributed processor patches!" << endl;
      Info << "No parallel run! Created cell processor addressing for master (global to global)..." << endl;
   }

#endif
    return true;
}

// ************************************************************************* //
