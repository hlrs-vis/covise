/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigRootErrorHandler.h"

using namespace covise;

coConfigRootErrorHandler::coConfigRootErrorHandler()
    : fSawErrors(false)
{
}

void coConfigRootErrorHandler::warning(const xercesc::SAXParseException &e)
{
    // if file could not be opend, handle that warning as an error
    if (QString::fromUtf16(reinterpret_cast<const ushort *>(e.getMessage())).contains("The primary document entity could not be opened."))
    {
        fSawErrors = true;
        COCONFIGDBG("coConfigRootErrorHandler::error err: schema file could not be opened");
        COCONFIGDBG("coConfigRootErrorHandler::error err: " << QString::fromUtf16(reinterpret_cast<const ushort *>(e.getMessage())));
    }
    else
    {
        COCONFIGLOG("coConfigRootErrorHandler::warning warn: parse warning in file (l,c): " << QString::fromUtf16(reinterpret_cast<const ushort *>(e.getSystemId())) << " (" << e.getLineNumber() << ", " << e.getColumnNumber() << ")");
        COCONFIGLOG("coConfigRootErrorHandler::warning warn: " << QString::fromUtf16(reinterpret_cast<const ushort *>(e.getMessage())));
    }
}

void coConfigRootErrorHandler::error(const xercesc::SAXParseException &e)
{
    COCONFIGLOG("coConfigRootErrorHandler::error err: parse warning in file (l,c): " << QString::fromUtf16(reinterpret_cast<const ushort *>(e.getSystemId())) << " (" << e.getLineNumber() << ", " << e.getColumnNumber() << ")");
    COCONFIGLOG("coConfigRootErrorHandler::error err: " << QString::fromUtf16(reinterpret_cast<const ushort *>(e.getMessage())));
}

void coConfigRootErrorHandler::fatalError(const xercesc::SAXParseException &e)
{
    error(e);
}

bool coConfigRootErrorHandler::getSawErrors() const
{
    return fSawErrors;
}

void coConfigRootErrorHandler::resetErrors()
{
    fSawErrors = false;
}
