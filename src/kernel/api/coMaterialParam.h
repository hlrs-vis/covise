/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_MATERIAL_PARAM_H_
#define _CO_MATERIAL_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coMaterialParam Parameter handling class                            +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include <appl/CoviseBase.h>
#include "coUifPara.h"

namespace covise
{

/// parameter to choose values from a list
class APIEXPORT coMaterialParam : public coUifPara
{
public:
    /// Constructor
    coMaterialParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coMaterialParam();

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

    /// set for data
    int setValue(const string &value);

    /// get the value of a single color component
    const string getValue() const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coMaterialParam(const coMaterialParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coMaterialParam &operator=(const coMaterialParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coMaterialParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    string d_value;

    // Parameter type name
    static const char *s_type;
};
}
#endif
