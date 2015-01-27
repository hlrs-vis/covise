/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EnumTypes.h"

Tools::EnumTypes::EnumTypes()
{
    names.clear();
}

Tools::EnumTypes::~EnumTypes()
{
    names.clear();
}

int Tools::EnumTypes::toString(int typeNumber, string &typeName)
{
    map<int, string>::iterator i;

    i = names.find(typeNumber);

    if (i != names.end())
    {
        typeName = i->second;

        return 1;
    }
    else
    {
        typeName = "<none>";

        return 0;
    }
}

int Tools::EnumTypes::toInt(string typeName, int &typeNumber)
{
    map<int, string>::iterator i;

    for (i = names.begin(); i != names.end(); i++)
    {
        if (i->second == typeName)
        {
            typeNumber = i->first;
            return 1;
        }
    }

    return 0;
}
