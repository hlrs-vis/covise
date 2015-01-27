/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_UIF_PORT_H_
#define _CO_UIF_PORT_H_

////////////////////////////////////////////////////////////////////////
//						(C)2001 VirCinity
//
// Base class for input and (not yet output) ports of a module
//
//
//
//                             Sven Kufer
//                      VirCinity IT-Consulting GmbH
//                            Nobelstrasse 15
//                            70569 Stuttgart
// Date: 13.08.2001
//
////////////////////////////////////////////////////////////////////////

#include <covise/covise.h>
#include "coPort.h"

namespace covise
{

class APIEXPORT coUifPort : public coPort
{
private:
    // whether port is connected
    int d_connected;

public:
    // default (de)constructors

    ~coUifPort()
    {
    }

private:
    coUifPort(const coUifPort &);

public:
    coUifPort(const char *name, const char *desc)
        : coPort(name, desc)
    {
        d_connected = 0;
    }

    int isConnected()
    {
        return d_connected;
    }

    // called by paramCB
    virtual int paramChange()
    {
        if (d_connected == 0)
            d_connected = 1;
        else
            d_connected = 0;
        return 1;
    }
};
}
#endif
