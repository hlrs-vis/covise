/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_INT_VECTOR_PARAM_H_
#define _CO_INT_VECTOR_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coIntVectorParam Parameter handling class                                +
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

/// Parameter for multiple integers
class APIEXPORT coIntVectorParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coIntVectorParam(const coIntVectorParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coIntVectorParam &operator=(const coIntVectorParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coIntVectorParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    int d_length;

    long *d_data;

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coIntVectorParam(const char *name, const char *desc, int length = 3);

    /// Destructor : virtual in case we derive objects
    virtual ~coIntVectorParam();

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
    int setValue(int pos, long data);

    /// set/update the value with full field/size return 0 on error
    int setValue(int size, const long *data);

    /// set for 3d data
    int setValue(long data0, long data1, long data2);

    /// get the value
    long getValue(int pos) const;

    /// get for 3d data -> return -1 if not 3d
    int getValue(long &data0, long &data1, long &data2) const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
