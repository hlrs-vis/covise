/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COLORNEW_PARAM_H
#define CO_COLORNEW_PARAM_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColorParam Parameter handling class                                  +
// +                                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coUifPara.h"

namespace covise
{

/// Multiple float parameters
class APIEXPORT coColorParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coColorParam(const coColorParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coColorParam &operator=(const coColorParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coColorParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    float d_value[4];

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coColorParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coColorParam();

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

    /// set for  data
    int setValue(float r, float g, float b, float a);

    /// get the value of a single color component
    float getValue(int c) const;

    /// get the value : return 0 on errror, 1 on success
    int getValue(float &r, float &g, float &b, float &a) const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
