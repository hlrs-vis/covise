/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coOutputPort.h"
#include "coInputPort.h"
#include <appl/ApplInterface.h>
#include <do/coDistributedObject.h>

/// ----- Never forget the Destructor !! -------

using namespace covise;

coOutputPort::~coOutputPort()
{
    if (d_outObj)
        delete d_outObj;
    if (d_objName)
        delete[] d_objName;
    if (d_depPort)
        delete[] d_depPort;
}

// create an input port
coOutputPort::coOutputPort(const char *name, const char *typelist, const char *desc)
    : coPort(name, desc)
{
    d_defString = strcpy(new char[strlen(typelist) + 1], typelist);
    d_objName = NULL;
    d_outObj = NULL;
    d_depPort = NULL;
}

void coOutputPort::initialize()
{
    Covise::add_port(OUTPUT_PORT, d_name, d_defString, d_desc);

    if (d_depPort)
    {
        char *dep = new char[strlen(d_depPort) + 5];
        strcpy(dep, "dep ");
        strcat(dep, d_depPort);
        Covise::set_port_dependency(d_name, dep);
        delete[] dep; // set_port_dependency makes copy
    }
}

void coOutputPort::setDependencyPort(coInputPort *port)
{
    const char *portName = port->getName();
    delete[] d_depPort;
    d_depPort = strcpy(new char[strlen(portName) + 1], portName);
}

/// do whatever is necessary before a compute() : get the object
int coOutputPort::preCompute()
{
    char *objName = Covise::get_object_name(d_name);

    if (objName)
    {
        d_objName = strcpy(new char[strlen(objName) + 1], objName);
        return 0;
    }
    else
    {
        Covise::sendError("Output port '%s' did not receive object name", d_name);
        return -1;
    }
}

void coOutputPort::setObjName(const char *n)
{
    if (n == d_objName)
        return;
    if (d_objName)
        delete[] d_objName;
    if (NULL == n)
    {
        n = "No_Object_Name_given";
    }
    int len = (int)strlen(n);
    d_objName = new char[len + 1];
    strcpy(d_objName, n);
    return;
}

/// do whatever is necessary before a compute() : get the object
int coOutputPort::postCompute()
{
    delete[] d_objName;
    d_objName = NULL;

    delete d_outObj;
    d_outObj = NULL;

    return 0;
}

/// set my active object if I have one
void coOutputPort::setCurrentObject(coDistributedObject *obj)
{
    d_outObj = obj;
}

coDistributedObject *coOutputPort::getCurrentObject()
{
    return d_outObj;
}

/// get my active object if I have one
const char *coOutputPort::getObjName()
{
    return d_objName;
}

coObjInfo coOutputPort::getNewObjectInfo()
{
    coObjInfo info;
    info.id.id = d_objName;
    return info;
}

void coOutputPort::print(ostream &str) const
{
    str << "Output Port '" << d_name
        << "' Typelist='" << d_defString
        << "'" << endl;
}

coUifElem::Kind coOutputPort::kind() const
{
    return coUifElem::OUTPORT;
}
