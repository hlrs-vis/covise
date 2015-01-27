/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coPort.h"
#include <appl/ApplInterface.h>

/// ----- Never forget the Destructor !! -------

namespace covise
{
static bool checkPortName(const char *name)
{
    if (!*name)
        return false;
    if (isdigit(*name))
        return false;

    for (const char *p = name; *p; ++p)
    {
        if (!isascii(*p))
            return false;
        if (!isalnum(*p) && *p != '_')
            return false;
    }

    return true;
}
}

using namespace covise;

coPort::~coPort()
{
    if (NULL != d_name)
        delete[] d_name;
    if (NULL != d_desc)
        delete[] d_desc;
    if (NULL != d_defString)
        delete[] d_defString;
}

coPort::coPort(const char *name, const char *desc)
{
    if (!checkPortName(name))
    {
        fprintf(stderr,
                "illegal port or parameter name '%s': use only alphanumeric ASCII characters, '_' and no digit as the first character\n",
                name);
        assert("illegal port or parameter name" == NULL);
    }
    d_name = strcpy(new char[strlen(name) + 1], name);
    d_desc = strcpy(new char[strlen(desc) + 1], desc);
    d_defString = NULL;
    d_init = 0;
}

const char *coPort::getName() const
{
    return d_name;
}

const char *coPort::getDesc() const
{
    return d_desc;
}

/// Return whether this port is connected -> valid only in compute()
int coPort::isConnected() const
{
    return Covise::is_port_connected(d_name);
}

void coPort::setInfo(const char *value) const
{
    if (!value)
        return;

    char *buffer = new char[strlen(value) + strlen(d_name) + 2];

    // Start with port name + \n
    strcpy(buffer, d_name);
    strcat(buffer, "\n");

    // Rest: everything except \n or \t
    char *bufPtr = buffer + strlen(buffer);
    while (*value)
    {
        if (*value != '\n' && *value != '\t')
        {
            *bufPtr = *value;
            bufPtr++;
        }
        value++;
    }
    *bufPtr = '\0'; // terminate string
    Covise::send_ui_message("PORT_DESC", buffer);
    delete[] buffer;
}
