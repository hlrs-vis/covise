/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_PORT_H_
#define _CO_PORT_H_

// 20.09.99

#include <covise/covise.h>
#include "coUifElem.h"

/**
 * Base class for all Ports
 *
 */

namespace covise
{

class APIEXPORT coPort : public coUifElem
{

protected:
    // Type information: whether this is a INPORT,OUTPORT, SWITCH, PARAM
    typedef int Typeinfo;

    // Port name, description
    char *d_name, *d_desc;

    // The 'default' string
    char *d_defString;

    // whether this port has been initialized
    int d_init;

public:
    coPort(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coPort();

    /// give all necessary info to Covise -> automatically called !
    virtual void initialize() = 0;

    /// print *this to a stream
    virtual void print(ostream &str) const = 0;

    /// get the name of this object
    virtual const char *getName() const;

    /// get the description of this object
    virtual const char *getDesc() const;

    /// return my type of element: coUifElem::{INPORT,OUTPORT,SWITCH,PARAM}
    virtual Kind kind() const = 0;

    /// Return whether this port is connected -> valid only in compute()
    int isConnected() const;

    /// Set the info Popup text
    void setInfo(const char *value) const;
};

inline ostream &operator<<(ostream &str, const coPort &port)
{
    port.print(str);
    return str;
}
}
#endif
