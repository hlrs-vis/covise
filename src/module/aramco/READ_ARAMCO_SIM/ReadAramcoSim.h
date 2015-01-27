/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _HELLO_H
#define _HELLO_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;
#include "AramcoSimFile.h"

class ReadAramcoSim : public coModule
{
private:
    enum
    {
        NUM_PORTS = 5
    };

    ////////// Parameters and ports

    // Filename
    coFileBrowserParam *p_filename;

    // Output selectors
    coChoiceParam *p_choice[NUM_PORTS];

    // y scaling factor
    coFloatParam *p_zScale;

    // Output data
    coOutputPort *p_mesh, *p_meshTime, *p_data[NUM_PORTS], *p_strMesh;

    ////////// Variables

    AramcoSimFile *d_simFile;

    //////////  member functions

    // read the file, called from either param or compute
    int readFile();

    /// compute call-back
    virtual int compute();

    /// param callback
    virtual void param(const char *paraname);

    /// postInst callback
    virtual void postInst();

    /// read a single data field into named object
    coDoFloat *readField(const char *name, int fieldNo, int stepNo);

public:
    ReadAramcoSim();
};
#endif
