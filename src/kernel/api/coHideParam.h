/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS coHideParam
//
//  Objects of this class hide a parameter, values are read from it
//  or from their own variables loaded from an attribute string
//  Examples: Transform
//
//  Initial version: 2002-06-?? Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _CO_HIDE_PARAM_H_
#define _CO_HIDE_PARAM_H_

#include <covise/covise.h>
#include "coUifPara.h"

/**
 * an coHideParam object overrides the value of a parameter
 * if the load function has been used
 */

//template class APIEXPORT vector<float>;
//template class APIEXPORT vector<int> ;

namespace covise
{

class APIEXPORT coHideParam
{
public:
    /** Constructor
       * @param parameter to be (possibly) overridden
       */
    coHideParam(coUifPara *param);
    /// Destructor
    ~coHideParam();
    /** load is used in order to load a value that overrides the parameter param_
       * @param  values string with the value or values that override those of pram_
       */
    void load(const char *values);
    /// get the float value (applies for coFloatParam or coFloatSliderParam)
    float getFValue();
    /// get the int value (applies for coChoiceParam, coBooleanParam, coIntScalarParam or coIntSliderParam)
    int getIValue();
    /// get the float value (applies for coFloatVectorParam)
    float getFValue(int);
    /// get the int values (applies for coFloatVectorParam)
    int getIValue(int);
    /// get the float values (applies for coFloatVectorParam)
    void getFValue(float &data0, float &data1, float &data2);
    /// resets hidden_ to false
    void reset();

protected:
private:
    coUifPara *param_; // the hidden parameter
    bool hidden_; // flag; if true, the parameter value is overridden
    vector<float> fnumbers_; // here we keep overriding values
    vector<int> inumbers_; // here we keep overriding values
};
}
#endif
