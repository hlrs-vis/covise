/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_FLOAT_SCALAR_PARAM_H_
#define _CO_FLOAT_SCALAR_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFloatParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coUifPara.h"

namespace covise
{

/// single float parameter
class APIEXPORT coFloatParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coFloatParam(const coFloatParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coFloatParam &operator=(const coFloatParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coFloatParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    float d_value;

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coFloatParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coFloatParam();

    /// Check the type
    virtual int isOfType(coUifPara::Typeinfo type);

    /// get my type
    static coUifPara::Typeinfo getType();

    /// handle parameter changes: called by paramCB
    virtual int paramChange();

    /// give dafault values to Covise -> automatically called !
    virtual void initialize();

    /// print this to a stream
    virtual void print(ostream &str) const;

    /// set/update the value: return 0 on error
    int setValue(float val);

    /// get the value
    float getValue() const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
