/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "InvComposePlane.h"

SO_ENGINE_SOURCE(InvComposePlane);

// Initializes the InvComposePlane class. This is a one-time
// thing that is done after database initialization and before
// any instance of this class is constructed.

void
InvComposePlane::initClass()
{
    // Initialize type id variables. The arguments to the macro
    // are: the name of the engine class, the class this is
    // derived from, and the name registered with the type
    // of the parent class.
    SO_ENGINE_INIT_CLASS(InvComposePlane, SoEngine, "Engine");
}

// Constructor

InvComposePlane::InvComposePlane()
{
    // Do standard constructor stuff
    SO_ENGINE_CONSTRUCTOR(InvComposePlane);

    // Define input fields and their default values
    SO_ENGINE_ADD_INPUT(normal, (1.0, 0.0, 0.0));
    SO_ENGINE_ADD_INPUT(point, (0.0, 0.0, 0.0));

    // Define the output, specifying its type
    SO_ENGINE_ADD_OUTPUT(plane, SoSFPlane);
}

// Destructor. Does nothing.

InvComposePlane::~InvComposePlane()
{
}

// This is the evaluation routine.

void
InvComposePlane::evaluate()
{
    // Compute the product of the input fields
    SbPlane p = SbPlane(normal.getValue(), point.getValue());

    // "Send" the value to the output. In effect, we are setting
    // the value in all fields to which this output is connected.
    SO_ENGINE_OUTPUT(plane, SoSFPlane, setValue(p));
}
