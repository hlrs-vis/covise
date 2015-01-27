/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coFeedback.h"
#include "coUifPara.h"
#include "coModule.h"
#include <do/coDistributedObject.h>

using namespace covise;

coFeedback::~coFeedback()
{
    if (NULL != d_paraBuf)
        delete[] d_paraBuf;
    if (NULL != d_userBuf)
        delete[] d_userBuf;
    if (NULL != d_pluginName)
        delete[] d_pluginName;
}

// add a string, separated and masked
void coFeedback::strAddMasked(char *str, const char *add)
{
    // separator
    strcat(str, "!");

    // pointer to append
    char *cPtr = str + strlen(str);

    // copy including terminating \0 (by '<=') and with masking all '\' and '!'
    int i, addLen = (int)strlen(add);
    for (i = 0; i <= addLen; i++)
    {
        if (add[i] == '\\' || add[i] == '!')
        {
            *cPtr = '\\';
            cPtr++;
        }
        *cPtr = add[i];
        cPtr++;
    }
}

/// Construct with name of Plugin
coFeedback::coFeedback(const char *pluginName)
{
    d_pluginName = strcpy(new char[strlen(pluginName) + 1], (pluginName));
    d_paraBuf = NULL;
    d_userBuf = NULL;
    d_numParam = 0;
    d_numUser = 0;
}

/// add a parameter for feedback
void coFeedback::addPara(coUifPara *parameter)
{
    const char *paraName = parameter->getName();
    const char *paraType = parameter->getTypeString();
    const char *paraVal = parameter->getValString();
    char *newBuf;

    // create new buffer (big enough for crazy things like "!!!!" -> \!\!\!\!"
    // and copy old string if existing
    if (d_paraBuf)
    {
        newBuf = new char[strlen(d_paraBuf) + strlen(paraName)
                          + strlen(paraType) + 2 * strlen(paraVal) + 4];
        strcpy(newBuf, d_paraBuf);
    }
    else
    {
        newBuf = new char[strlen(paraName) + strlen(paraType) + 2 * strlen(paraVal) + 4];
        newBuf[0] = '\0';
    }
    strAddMasked(newBuf, paraName);
    strAddMasked(newBuf, paraType);
    strAddMasked(newBuf, paraVal);

    delete[] d_paraBuf;
    d_paraBuf = newBuf;
    d_numParam++;
}

/// add a user string for feedback: return FAIL/SUCCESS
void coFeedback::addString(const char *userString)
{
    char *newBuf;
    if (d_userBuf)
    {
        newBuf = new char[strlen(d_userBuf) + 2 * strlen(userString) + 2];
        strcpy(newBuf, d_userBuf);
    }
    else
    {
        newBuf = new char[2 * strlen(userString) + 2]; // worst case, all separated
        newBuf[0] = '\0';
    }
    strAddMasked(newBuf, userString);

    delete[] d_userBuf;
    d_userBuf = newBuf;
    d_numUser++;
}

/// apply the feedback to an Object: create Attribute: return FAIL/SUCCESS
void coFeedback::apply(coDistributedObject *obj)
{
    // always add a 'MODULE' parameter to load the plugin (aw: do it first)
    obj->addAttribute("MODULE", d_pluginName);

    char *buffer = new char[(d_paraBuf ? strlen(d_paraBuf) : 0)
                            + (d_userBuf ? strlen(d_userBuf) : 0)
                            + 64];

    sprintf(buffer, "coFeedback: %d %d %s%s", d_numParam, d_numUser,
            (d_paraBuf ? d_paraBuf : ""),
            (d_userBuf ? d_userBuf : ""));
    Covise::addInteractor(obj, d_pluginName, buffer);
}
