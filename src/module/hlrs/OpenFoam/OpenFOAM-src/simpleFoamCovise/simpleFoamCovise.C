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

Application
    simpleFoam

Description
    Steady-state solver for incompressible, turbulent flow

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "RASModel.H"

#include "../coviseToFoam/coviseToFoam.H"
#include "systemCalls.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    int argc_ser = 1;
    char **argv_ser = new char *[argc_ser];
    argv_ser[0] = new char[strlen(argv[0])];
    strncpy(argv_ser[0], argv[0], strlen(argv[0]));

    char *rank_string = getenv("OMPI_COMM_WORLD_RANK");

    if (rank_string == NULL) //---I'm the only one---
    {
        Foam::Info << "No parallel run: Preprocessing..." << Foam::endl;

        Foam::argList args_ser(argc_ser, argv_ser);
        if (!args_ser.checkRootCase())
        {
            Foam::FatalError.exit();
        }

        Foam::Info << "Create time\n" << Foam::endl;
        Foam::Time runTime_ser(
            Foam::Time::controlDictName,
            args_ser.rootPath(),
            args_ser.caseName());

        coviseToFoam(runTime_ser);

        callRemoveLogs();
        callSetSet();
        callSetsToZones();
    }
    else if (atoi(rank_string) == 0) //---I'm the master---
    {
        Foam::Info << "Parallel run: Master preprocessing..." << Foam::endl;

        Foam::argList args_ser(argc_ser, argv_ser);
        if (!args_ser.checkRootCase())
        {
            Foam::FatalError.exit();
        }

        Foam::Info << "Create time\n" << Foam::endl;
        Foam::Time runTime_ser(
            Foam::Time::controlDictName,
            args_ser.rootPath(),
            args_ser.caseName());

        coviseToFoam(runTime_ser);

        callRemoveLogs();
        callSetSet();
        callSetsToZones();

        callDecomposePar();
    }

//MPI_Barrier(MPI_COMM_WORLD);
//Where is MPI?
//MPI_Barrier not exposed by Pstream -> reduction operation mistreat as barrier
/*{
      label tmp = Pstream::myProcNo();
      reduce(tmp,sumOp<label>());
   }*/

#include "setRootCase.H"
#include "createTime.H"

#include "createMesh.H"
#include "createFields.H"
#include "initContinuityErrs.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;

#include "readSIMPLEControls.H"
#include "initConvergenceCheck.H"

        p.storePrevIter();

        // Pressure-velocity SIMPLE corrector
        {
#include "UEqn.H"
#include "pEqn.H"
        }

        turbulence->correct();

        runTime.write();

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;

#include "convergenceCheck.H"
    }

    Info << "End\n" << endl;

    return 0;
}

// ************************************************************************* //
