/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ATTRIBUTE_DEFINED
#define __ATTRIBUTE_DEFINED
#include <string.h>

class attribute
{
private:
    int lineNo;
    int i;
    double d;
    char *str;
    bool b;
    char *hostName;

public:
    attribute(int i);
    attribute(double d);
    //This constructor is used for strings or
    //boolean when boolType is true
    attribute(const char *s, bool isBoolean = false);
    attribute(const char *s, int enumToken);
    ~attribute();
    int getLineNo()
    {
        return lineNo;
    }
    int getInt()
    {
        return i;
    }
    double getDouble()
    {
        return d;
    }
    int getBoolInt()
    {
        if (b)
            return 1;
        else
            return 0;
    }
    bool getBoolean()
    {
        return b;
    }

    const char *getString()
    {
        return str;
    }
    const char *getHostName()
    {
        return hostName;
    }
};
#endif
