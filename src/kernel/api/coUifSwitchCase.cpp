/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coUifSwitchCase.h"
#include "coUifElem.h"

/// ----- Prevent auto-generated functions by assert -------

using namespace covise;

/// Copy-Constructor: NOT IMPLEMENTED
coUifSwitchCase::coUifSwitchCase(const coUifSwitchCase &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
coUifSwitchCase &coUifSwitchCase::operator=(const coUifSwitchCase &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
coUifSwitchCase::coUifSwitchCase()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

coUifSwitchCase::~coUifSwitchCase()
{
    delete[] d_name;
}

coUifSwitchCase::coUifSwitchCase(const char *name, coUifSwitch *master)
{
    d_name = strcpy(new char[strlen(name) + 1], name);
    d_master = master;
    d_numElem = 0;
}

/// get the name of this object
const char *coUifSwitchCase::getName() const
{
    return d_name;
}

/// add one element to out group
void coUifSwitchCase::add(coUifElem *elem)
{
    d_elemList[d_numElem] = elem;
    d_numElem++;
}

/// Hide everything below
void coUifSwitchCase::hide()
{
    int i;
    for (i = 0; i < d_numElem; i++)
        d_elemList[i]->hide();
}

/// Show everything below
void coUifSwitchCase::show()
{
    int i;
    for (i = 0; i < d_numElem; i++)
        d_elemList[i]->show();
}

/// get my superior switch
coUifSwitch *coUifSwitchCase::getMaster() const
{
    return d_master;
}
