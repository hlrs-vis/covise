/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coInputPort.h"
#include "coOutputPort.h"
#include <appl/ApplInterface.h>
#include <do/coDistributedObject.h>

/// ----- Never forget the Destructor !! -------

using namespace covise;

coInputPort::~coInputPort()
{
    delete d_inObj;
}

// create an input port
coInputPort::coInputPort(const char *name, const char *typelist, const char *desc)
    : coUifPort(name, desc)
{
    d_defString = strcpy(new char[strlen(typelist) + 1], typelist);
    d_required = 1;
    d_inObj = NULL;
    oldObjectName = NULL;
    objectChanged = true;
}

void coInputPort::initialize()
{
    Covise::add_port(INPUT_PORT, d_name, d_defString, d_desc);
    Covise::set_port_required(d_name, d_required);
}

/// do whatever is necessary before a compute() : get the object
int coInputPort::preCompute()
{
    const char *objName = Covise::get_object_name(d_name);
    d_inObj = NULL;
    if ((!oldObjectName && !objName)
        || (oldObjectName && objName && strcmp(oldObjectName, objName) == 0))
    {
        objectChanged = false;
    }
    else
    {
        delete[] oldObjectName;
        oldObjectName = NULL;
        if (objName)
        {
            oldObjectName = new char[strlen(objName) + 1];
            strcpy(oldObjectName, objName);
        }
        objectChanged = true;
    }

    if (objName)
    {
        d_inObj = coDistributedObject::createFromShm(objName);
    }
    else if (d_required)
    {
        Covise::sendError("Error on required Port '%s' : No input object name", d_name);
        return -1;
    }

    if (d_required && (!d_inObj))
    {
// error messages usually disabled
#ifndef TOLERANT
        Covise::sendError("Error on required Port '%s' : No input object", d_name);
#endif
        return -1;
    }
    else
        return 0;
}

/// do whatever is necessary before a compute() : get the object
int coInputPort::postCompute()
{
    delete d_inObj;
    d_inObj = NULL;
    return 0;
}

/// get my active object if I have one
const coDistributedObject *coInputPort::getCurrentObject() const
{
    return d_inObj;
}

void coInputPort::setCurrentObject(const coDistributedObject *o)
{
    d_inObj = o;
}

/// print to a stream
void coInputPort::print(ostream &str) const
{
    str << "Input Port '" << d_name
        << "' Typelist='" << d_defString
        << "'" << endl;
}

coUifElem::Kind coInputPort::kind() const
{
    return coUifElem::INPORT;
}

// set port required or not
void coInputPort::setRequired(int isRequired)
{
    d_required = isRequired;
}
