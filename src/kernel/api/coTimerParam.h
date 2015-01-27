/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TIMER_PARAM_H_
#define _CO_TIMER_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coTimerParam Parameter handling class                                +
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

/// Integer slider parameter
class APIEXPORT coTimerParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coTimerParam(const coTimerParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coTimerParam &operator=(const coTimerParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coTimerParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    long d_start, d_delta, d_state;

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coTimerParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coTimerParam();

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
    int setValue(long start, long delta, long state);
    int setStart(long start);
    int setDelta(long delta);
    int setState(long state);

    /// get the value
    void getValue(long &start, long &delta, long &state) const;
    long getStart() const;
    long getDelta() const;
    long getState() const;

    /// The Timer is special: all these are not supported
    virtual void setImmediate(int isImmediate);
    virtual void setActive(int isActive);
    virtual void enable();
    virtual void disable();
    virtual void hide();
    virtual void show();

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
