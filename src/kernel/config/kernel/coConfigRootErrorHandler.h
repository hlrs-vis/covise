/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGROOTERRORHANDLER_H
#define COCONFIGROOTERRORHANDLER_H

#include <xercesc/sax2/DefaultHandler.hpp>
#include <config/coConfigLog.h>

namespace covise
{

// Receives Error notifications.
class coConfigRootErrorHandler : public xercesc::DefaultHandler
{
public:
    coConfigRootErrorHandler();

    void warning(const xercesc::SAXParseException &e);

    void error(const xercesc::SAXParseException &e);

    void fatalError(const xercesc::SAXParseException &e);
    bool getSawErrors() const;
    void resetErrors();

private:
    bool fSawErrors;
};
}
#endif
