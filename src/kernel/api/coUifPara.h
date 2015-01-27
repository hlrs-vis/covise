/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_PARA_PORT_H_
#define _CO_PARA_PORT_H_

// 13.09.99

#include <covise/covise.h>
#include "coPort.h"

/**
 * Base class to manage parameter ports
 *
 */
namespace covise
{

class APIEXPORT coUifPara : public coPort
{

public:
    // Type information: which kind of port is this
    typedef int Typeinfo;

protected:
    // calculate a number from a string: similar to DO, but not same
    static Typeinfo numericType(const char *typeStr);

private:
    /// Assignment operator: NOT IMPLEMENTED
    coUifPara &operator=(const coUifPara &);

    /// Default constructor: NOT IMPLEMENTED
    coUifPara();

    /// Copy-constructor: NOT IMPLEMENTED
    coUifPara(const coUifPara &old);

protected:
    // whether this port is active: inactive ports do not show up on show()
    int d_active;

public:
    /** Create a parameter with a certain type
       *  @param    name  Port name
       *  @param    desc  Port description
       */
    coUifPara(const char *name, const char *desc);

    /// Destructor
    virtual ~coUifPara();

    /// activate / de_activate the parameter
    virtual void setActive(int isActive);

    /// Hide this port
    virtual void hide();

    /// Show this port
    virtual void show();

    /// handle parameter changes: called by paramCB
    virtual int paramChange() = 0;

    /// do whatever needed before compute CB : enforce that all parameters do sth.
    virtual int preCompute()
    {
        return 0;
    }

    /// give all necessary info to Covise -> automatically called !
    virtual void initialize() = 0;

    /// print this to a stream: overloaded, but called by derived classes
    virtual void print(ostream &str) const;

    /// whether this port can be switched: default is true, overload for false
    virtual int switchable() const;

    /// check whether this parameter port has a certain type
    virtual int isOfType(coUifPara::Typeinfo type) = 0;

    /// enable
    virtual void enable();

    /// disable
    virtual void disable();

    /// return my type of element: returns coUifElem::PARAM
    virtual Kind kind() const;

    /// check whether this port is active
    int isActive() const;

    /// send an error message strarting with "Parameter '<name>' "
    void para_error(const char *what);

    /// get the type string of this parameter
    virtual const char *getTypeString() const = 0;

    /// get the value of this parameter as a string
    virtual const char *getValString() const = 0;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str) = 0;
};
}
#endif
