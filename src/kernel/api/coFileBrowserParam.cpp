/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFileBrowserParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coFileBrowserParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coFileBrowserParam::s_type = "BROWSE";
coUifPara::Typeinfo coFileBrowserParam::s_paraType = coUifPara::numericType("BROWSE");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coFileBrowserParam::coFileBrowserParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_filter = "*";
    d_path = ".";
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coFileBrowserParam::~coFileBrowserParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coFileBrowserParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coFileBrowserParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coFileBrowserParam::paramChange()
{
    const char *newVal;
    int retVal = Covise::get_reply_browser(&newVal);

    if (retVal)
        d_path = newVal;

    return retVal;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coFileBrowserParam::initialize()
{

    Covise::add_port(PARIN, d_name, "Browser", d_desc);
    Covise::set_port_default(d_name, d_path.c_str());

    string fname(d_name);
    fname.append("___filter");
    d_filter.insert(0, " ");
    d_filter.insert(0, d_name);
    Covise::add_port(PARIN, fname.c_str(), "BrowserFilter", d_desc);
    Covise::set_port_default(fname.c_str(), d_filter.c_str());

    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coFileBrowserParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "coFileBrowserParam : " << d_path << " " << d_filter << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coFileBrowserParam::setValue(const char *path, const char *filter)
{
    /// If we have been initialized, update the map
    if (d_init)
    {
        Covise::sendWarning("Cannot update file browsers");
        return 0;
    }
    else
    {
        d_filter = filter;
        d_path = path;
        return 1;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

const char *coFileBrowserParam::getValue() const
{
    return d_path.c_str();
}

/// ----- Prevent auto-generated functions by assert -------
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coFileBrowserParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coFileBrowserParam::getValString() const
{
    static char valString[512];
    sprintf(valString, "%s %s", d_path.c_str(), d_filter.c_str());
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coFileBrowserParam::setValString(const char *str)
{
    size_t retval;
    char *a = NULL, *b = NULL;
    a = new char[strlen(str) + 1];
    b = new char[strlen(str) + 1];
    retval = sscanf(str, "%s %s", a, b);
    d_path = a;
    d_filter = b;
    delete[] a;
    delete[] b;
    if (retval != 2)
        std::cerr << "coFileBrowserParam::setValString: sscanf failed" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the file name
void coFileBrowserParam::setFilename(const char *fName)
{
    d_path = fName;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the filter
void coFileBrowserParam::setFilter(const char *filter)
{
    d_filter = filter;
}
