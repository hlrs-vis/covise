/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OutputManagerBase.h"

using namespace Tools;

ostream *Tools::out = NULL;

ostream &Tools::endLineDebug(ostream &ostr)
{
    stringbuf *buf;

    buf = (stringbuf *)ostr.rdbuf();

#ifdef DEBUG_MODE
    cout << buf << endl;
#endif

    buf->str("");

    buf = NULL;

    return ostr;
}

ostream &Tools::endDebug(ostream &ostr)
{
    stringbuf *buf;

    buf = (stringbuf *)ostr.rdbuf();

#ifdef DEBUG_MODE
    cout << buf;
#endif

    buf->str("");

    return ostr;
}

ostream &Tools::endLine(ostream &ostr)
{
    stringbuf *buf;

    buf = (stringbuf *)ostr.rdbuf();

    cout << buf << endl;

    buf->str("");

    return ostr;
}

ostream &Tools::end(ostream &ostr)
{
    stringbuf *buf;

    buf = (stringbuf *)ostr.rdbuf();

    cout << buf;

    buf->str("");

    return ostr;
}

OutputManagerBase::OutputManagerBase()
    : BaseObject()
{
#ifdef DEBUG_MODE
    cout << "OutputManagerBase::OutputManagerBase()" << endl;
#endif
}

OutputManagerBase::OutputManagerBase(string className, int objectID)
    : BaseObject(className, objectID)
{
#ifdef DEBUG_MODE
    cout << "OutputManagerBase::OutputManagerBase(ID)" << endl;
#endif
}

OutputManagerBase::~OutputManagerBase()
{
#ifdef DEBUG_MODE
    cout << "OutputManagerBase::OutputManagerBase(ID)" << endl;
#endif
}

void OutputManagerBase::setDebug(bool debugOnOff)
{
}

ostream *OutputManagerBase::getOut()
{
    stringstream s;

    return new ostream(s.rdbuf());
}

void OutputManagerBase::print(char *format, bool critical, ...)
{
}
