/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IN_PORT_H_
#define _CO_IN_PORT_H_

// 20.09.99
#include <covise/covise.h>
#include "coUifPort.h"

/**
 * Class
 *
 */

namespace covise
{

class coDistributedObject;
class coOutputPort;

/// Data input Port
class APIEXPORT coInputPort : public coUifPort
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coInputPort(const coInputPort &);

    /// Assignment operator: NOT  IMPLEMENTED
    coInputPort &operator=(const coInputPort &);

    /// Default constructor: NOT  IMPLEMENTED
    coInputPort();

    /// Object at the input port
    const coDistributedObject *d_inObj;

    /// to check whether the Dataobject changed since last execute
    char *oldObjectName;

    bool objectChanged;

    // whether the port is required
    int d_required;

public:
    // create an input port
    coInputPort(const char *name, const char *typelist, const char *desc);

    // set port required or not
    void setRequired(int isRequired);

    /// give all necessary info to Covise -> automatically called !
    virtual void initialize();

    /// return my type of element: coUifElem::INPORT
    virtual Kind kind() const;

    /// Destructor : virtual in case we derive objects
    virtual ~coInputPort();

    /// do whatever is needed before compute CB : pre-set to do nothing
    virtual int preCompute();

    /// do whatever is necessary after a compute() : delete the object
    virtual int postCompute();

    /// get my active object if I have one
    const coDistributedObject *getCurrentObject() const;

    /// set my active object - should only be used by the api
    void setCurrentObject(const coDistributedObject *o);

    /// print to a stream
    void print(ostream &) const;

    /// return true whether the input data changed since last execute.
    bool inputDataChanged()
    {
        return objectChanged;
    };
};
}
#endif
