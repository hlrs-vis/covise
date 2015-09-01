/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_HEADER_H
#define OSC_HEADER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscHeader: public oscObjectBase
{
public:
    oscShort revMajor;
    oscShort revMinor;
    oscString description;
    oscString date;
    oscString author;
    oscHeader(); ///< constructor
    virtual ~oscHeader(); ///< destructor

    virtual int parseFromXML(xercesc::DOMElement *currentElement);

};

typedef oscObjectVariable<oscHeader *> oscHeaderMember;

}

#endif //OSC_HEADER_H