/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_COLORMAP_PARAM_H_
#define _CO_COLORMAP_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColormapParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "coUifPara.h"
#include <covise/covise.h>
#include <covise/covise_msg.h>

namespace covise
{

/// File browser parameter
class APIEXPORT coColormapParam : public coUifPara
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coColormapParam(const coColormapParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coColormapParam &operator=(const coColormapParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coColormapParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    int d_len, d_lenData;
    float d_min;
    float d_max;
    float *d_rgbax;
    float *d_data;
    colormap_type d_colormapType;

    /// return buffer
    mutable char *valString;

    /// Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coColormapParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coColormapParam();

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
    int setValue(int len, float min, float max, const float *rgbax);

    /// set/update the value: return 0 on error
    int setData(int len, const float *data);

    /// set/update the value: return 0 on error
    int setMinMax(float min, float max);

    /// get the value
    int getValue(float *min, float *max, colormap_type *type, const float **rgbax) const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
