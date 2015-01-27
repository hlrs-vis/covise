/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_FEEDBACK_H_
#define _CO_FEEDBACK_H_

// 25.04.00

#include <util/coTypes.h>

/**
 * Class to create Feedback messages
 *
 */

namespace covise
{

class coDistributedObject;
class coUifPara;

class APIEXPORT coFeedback
{

private:
    // Buffers for Parameters and Ports
    char *d_paraBuf, *d_userBuf;

    // the plugin's name
    char *d_pluginName;

    // number of Parameters and ports in this feedback stream
    int d_numParam;
    int d_numUser;

    // the separator string
    static const char *s_sep;

    void strAddMasked(char *str, const char *add);

public:
    /// Construct with name of Plugin
    coFeedback(const char *pluginName);

    /// add a parameter for feedback: return FAIL/SUCCESS
    void addPara(coUifPara *parameter);

    /// add a user string for feedback: return FAIL/SUCCESS
    void addString(const char *userString);

    /// apply the feedback to an Object: create Attribute: return FAIL/SUCCESS
    void apply(coDistributedObject *obj);

    /// Destructor : virtual in case we derive objects
    virtual ~coFeedback();
};
}
#endif
