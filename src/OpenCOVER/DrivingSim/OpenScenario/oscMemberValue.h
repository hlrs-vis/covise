/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MEMBER_VALUE_H
#define OSC_MEMBER_VALUE_H

#include "oscExport.h"

#include <string>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMAttr;
class DOMElement;
class DOMDocument;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario
{

/// \class This class represents a member variable value in an oscObject
class OPENSCENARIOEXPORT oscMemberValue 
{
public:
    enum MemberTypes
    {
        UINT = 0,
        INT = 1,
        USHORT = 2,
        SHORT = 3,
        STRING = 4,
        DOUBLE = 5,
        OBJECT = 6,
        DATE_TIME = 7,
        ENUM = 8,
        BOOL = 9,
        FLOAT = 10,
    };

protected:
    enum MemberTypes type;

public:
    oscMemberValue() : ///< constructor
            type(INT)
    {

    };

    virtual ~oscMemberValue() ///< destructor
    {

    };

    // set the value with the specified type instead of using initialize(), done in oscValue
    virtual void setValue(const int &t) { };
    virtual void setValue(const unsigned int &t) { };
    virtual void setValue(const short &t) { };
    virtual void setValue(const unsigned short &t) { };
    virtual void setValue(const std::string &t) { };
    virtual void setValue(const double &t) { };
    virtual void setValue(const time_t &t) { };
    virtual void setValue(const bool &t) { };
    virtual void setValue(const float &t) { };

    MemberTypes getType() const ///< return the type of this value
    {
        return type;
    };

    virtual bool initialize(xercesc::DOMAttr *) = 0;

    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, const char *name) = 0;
};

}

#endif //OSC_MEMBER_VALUE_H
