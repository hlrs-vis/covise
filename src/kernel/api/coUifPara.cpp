/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <appl/ApplInterface.h>
#include "coUifPara.h"
#include "coBlankConv.h"

#undef VERBOSE

using namespace covise;

//////////////////////////////////////////////////////////////////////////

coUifPara::coUifPara(const char *name, const char *desc)
    : coPort(name, desc)
{
    d_active = 1; // by default active
}

coUifPara::~coUifPara()
{
}

void coUifPara::setActive(int isActive)
{
    d_active = isActive;
}

// Hide everything below
void coUifPara::hide()
{
    if (d_init)
        Covise::hide_param(d_name);
}

/// Show everything below
void coUifPara::show()
{
    if (d_active && d_init)
        Covise::show_param(d_name);
}

/// return my type of element
coUifElem::Kind coUifPara::kind() const
{
    return PARAM;
}

/// whether this port can be switched : default is true
int coUifPara::switchable() const
{
    return 1;
}

coUifPara::Typeinfo coUifPara::numericType(const char *typeStr)
{
#ifdef VERBOSE
    cerr << "converted type string '" << typeStr;
#endif
    int val = 0;
    while (*typeStr)
    {
        val <<= 5;
        val |= *typeStr & 0x1f;
        typeStr++;
    }
#ifdef VERBOSE
    cerr << "' to value" << val << endl;
#endif
    return val;
}

/// print this to a stream
void coUifPara::print(ostream &str) const
{
    str << "\nParameter '" << d_name << "' : '" << d_desc
        << ", ACT=" << d_active
        << ", INIT=" << d_init
        << endl;
}

/// enable
void coUifPara::enable()
{
    Covise::enable_param(d_name);
}

/// disable
void coUifPara::disable()
{
    Covise::disable_param(d_name);
}

/// check whether this port is active
int coUifPara::isActive() const
{
    return d_active;
}

void coUifPara::para_error(const char *what)
{
    char buffer[256];
    sprintf(buffer, "Parameter '%s' %s", d_name, what);
}
