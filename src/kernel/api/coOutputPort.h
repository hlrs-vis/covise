/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_OUT_PORT_H_
#define _CO_OUT_PORT_H_

// 20.09.99
#include <covise/covise.h>
#include <util/coObjID.h>
#include "coPort.h"

namespace covise
{

class coDistributedObject;
class coInputPort;

/**
 * Data Output Port
 *
 */
class APIEXPORT coOutputPort : public coPort
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coOutputPort(const coOutputPort &);

    /// Assignment operator: NOT  IMPLEMENTED
    coOutputPort &operator=(const coOutputPort &);

    /// Default constructor: NOT  IMPLEMENTED
    coOutputPort();

    /// Object at the output port
    coDistributedObject *d_outObj;

    /// Name for the output port object
    char *d_objName;

    /// Name of the port this one is depending on
    char *d_depPort;

public:
    // create an input port
    coOutputPort(const char *name, const char *typelist, const char *desc);

    // set port required or not
    void setRequired(int isRequired);

    // set port dependent on an input port
    void setDependencyPort(coInputPort *port);

    /// give all necessary info to Covise -> automatically called !
    virtual void initialize();

    /// return my type of element: coUifElem::INPORT
    virtual Kind kind() const;

    /// Destructor : virtual in case we derive objects
    virtual ~coOutputPort();

    /// do whatever is necessary after a compute()
    virtual int preCompute();

    /// do whatever is necessary after a compute() : delete the object
    virtual int postCompute();

    /// set my active object if I have one
    void setCurrentObject(coDistributedObject *obj);

    /// get my active object
    coDistributedObject *getCurrentObject();

    /// set my active object if I have one
    const char *getObjName();

    /// get my active object if I have one
    coObjInfo getNewObjectInfo();

    /// set object-name - should only be used by the api !
    void setObjName(const char *n);

    /// print to a stream
    void print(ostream &) const;
};
}
#endif
