/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_FLOAT_VECTOR_PARAM_H_
#define _CO_FLOAT_VECTOR_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFloatVectorParam Parameter handling class                                +
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

/// Multiple float parameters
class APIEXPORT coFloatVectorParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coFloatVectorParam(const coFloatVectorParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coFloatVectorParam &operator=(const coFloatVectorParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coFloatVectorParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    int d_length;

    float *d_data;

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coFloatVectorParam(const char *name, const char *desc, int length = 3);

    /// Destructor : virtual in case we derive objects
    virtual ~coFloatVectorParam();

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
    int setValue(int pos, float data);

    /// set/update the value with full field/size return 0 on error
    int setValue(int size, const float *data);

    /// set for 3d data
    int setValue(float data0, float data1, float data2);

    /// get the value
    float getValue(int pos) const;

    /// get the value : return 0 on errror, 1 on success
    int getValue(float &data0, float &data1, float &data2) const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
