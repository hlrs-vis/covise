/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include "attribute.h"

attribute::~attribute()
{
    if (NULL != str)
    {
        delete str;
        str = NULL;
    }
}

attribute::attribute(const char *s, bool isBoolean)
{
    str = NULL;
    if (isBoolean)
    {
        if (0 == strcasecmp(s, "true"))
        {
            b = true;
            str = new char[5];
            strcpy(str, "TRUE");
        }
        else if (0 == strcasecmp(s, "on"))
        {
            b = true;
            str = new char[5];
            strcpy(str, "TRUE");
        }
        else if (0 == strcasecmp(s, "false"))
        {
            b = false;
            str = new char[5];
            strcpy(str, "FALSE");
        }
        else if (0 == strcasecmp(s, "off"))
        {
            b = false;
            str = new char[6];
            strcpy(str, "FALSE");
        }
    }
    else
    {
        if (NULL != s)
        {
            str = new char[1 + strlen(s)];
            strcpy(str, s);
        }
    }
}

attribute::attribute(const char *s, int enumToken)
{
    str = NULL;
    if (NULL != s)
    {
        str = new char[1 + strlen(s)];
        strcpy(str, s);
    }
    i = enumToken;
}

attribute::attribute(int i)
{
    this->i = i;
    this->d = static_cast<float>(i);
    str = new char[50];
    sprintf(str, "%d", this->i);
}

attribute::attribute(double d)
{
    this->d = d;
    str = new char[50];
    sprintf(str, "%f", this->d);
}
