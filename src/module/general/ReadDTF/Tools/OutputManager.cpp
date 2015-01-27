/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OutputManager.h"

#include <cstdarg>
#include <cstdio>

using namespace std;
using namespace Tools;

CLASSINFO_OBJ(ClassInfo_ToolsOutputManager, OutputManager, "Tools::OutputManager", 1);

OutputManager::OutputManager()
{
#ifdef DEBUG_MODE
    cout << "OutputManager::OutputManager()" << endl;
#endif

    debug = false;
    INC_OBJ_COUNT(getClassName());
}

OutputManager::OutputManager(string className, int objectID)
    : OutputManagerBase(className, objectID)
{
#ifdef DEBUG_MODE
    cout << "OutputManager::OutputManager(ID)" << endl;
#endif

    debug = false;

    INC_OBJ_COUNT(getClassName());
}

OutputManager::~OutputManager()
{
#ifdef DEBUG_MODE
    cout << "OutputManager::~OutputManager()" << endl;
#endif

    DEC_OBJ_COUNT(getClassName());
}

void OutputManager::setDebug(bool debugOnOff)
{
    debug = debugOnOff;
}

void OutputManager::print(char *format, bool critical, ...)
{
    va_list ap;

    va_start(ap, critical);

    if (debug || critical)
        vprintf(format, ap);

    va_end(ap);
}

ostream *OutputManager::getOut()
{
    return new ostream(this->s.rdbuf());
}
