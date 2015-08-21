/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBJECT_BASE_H
#define OSC_OBJECT_BASE_H
#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscMember.h>
#include <oscFactory.h>
#include <string>
#include <list>
#include <xercesc/parsers/XercesDOMParser.hpp>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectBase: oscMemberValue
{
public:
    static oscFactory<oscObjectBase> factory;
    typedef std::list<oscMember *> MemberList;
protected:
    OpenScenarioBase *base;
    MemberList members; ///< lost of all member variables

public:
    oscObjectBase(); ///< constructor
    virtual ~oscObjectBase(); ///< destructor
    virtual void initialize(OpenScenarioBase *b);

    virtual int parseFromXML(xercesc::DOMElement *currentElement);

};

}

#endif //OSC_OBJECT_BASE_H