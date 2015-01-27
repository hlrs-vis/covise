/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ****************************************
 ** Read module for MEQSICO data format **
 ** Author: Tobias Bachran              **
 **************************************** */

#include <stdlib.h>
#include <stdio.h>

#include <api/coModule.h>
using namespace covise;

class coMEQSICO : public coModule
{
private:
    //member functions
    virtual int compute(const char *port);

    coOutputPort *meshOutPort;
    coOutputPort *mesh_dualOutPort;
    coOutputPort *Voltage_scalarOutPort;
    coOutputPort *EField_vectorOutPort;

    coFileBrowserParam *filenameParam;

public:
    coMEQSICO(int agrc, char *argv[]);
    virtual ~coMEQSICO();
};
