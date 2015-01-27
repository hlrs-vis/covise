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

Class
    stateToCovise

Description
    FunctionObject reports min and max of a volScalarField and bounds it

Author
    Hrvoje Jasak, Wikki Ltd.  All rights reserved

SourceFiles
    stateToCovise.C

\*---------------------------------------------------------------------------*/

#ifndef stateToCovise_H
#define stateToCovise_H

#include "functionObject.H"
#include "dictionary.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class stateToCovise Declaration
\*---------------------------------------------------------------------------*/

class stateToCovise
    : public functionObject
{
    // Private data

    labelList *cellProcAddressing;

    //- Name
    const word name_;

    //- Reference to main object registry
    const Time &time_;

    //- Region name
    word regionName_;

    //- CPU time step to send data
    scalar sendCPUTimeStep_;
    scalar lastCPUTime_;

    //- Field name
    word pFieldName_;
    word uFieldName_;

    // Private Member Functions

    //- Disallow default bitwise copy construct
    stateToCovise(
        const stateToCovise &);

    //- Disallow default bitwise assignment
    void operator=(const stateToCovise &);

    void exitOnCoviseConnectionLoss();

public:
    //- Runtime type information
    TypeName("stateToCovise");

    // Constructors

    //- Construct from components
    stateToCovise(
        const word &name,
        const Time &,
        const dictionary &);

    // Member Functions

    //- start is called at the start of the time-loop
    virtual bool start();

    //- execute is called at each ++ or += of the time-loop
    virtual bool execute();

    //- Read and set the function object if its data has changed
    virtual bool read(const dictionary &dict);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
