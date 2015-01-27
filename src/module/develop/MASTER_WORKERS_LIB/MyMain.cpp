/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE MyExample
//
// This programme tests and documents the MasterAndWorkers library
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "MyMasterAndWorkers.h"

int
main()
{
    // computation creates an object of type MyTaskGroup.
    // This object creates some objects of class MythreadTask.
    // These MythreadTask objects define tasks that are assigned
    // to a pool of threads created by computation.
    MyMasterAndWorkers computation;
    return computation.compute();
}
