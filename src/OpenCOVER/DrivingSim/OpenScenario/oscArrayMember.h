/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ARRAY_MEMBER_H
#define OSC_ARRAY_MEMBER_H

#include <oscExport.h>
#include <oscMember.h>
#include <string>
#include <vector>
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a Member variable storing arrays of values
class OPENSCENARIOEXPORT oscArrayMember: public oscMember
{
protected:

    std::vector<oscObjectBase *>values;

public:
    //oscArrayMember(std::string &name, oscMemberValue::MemberTypes type, oscObjectBase* owner); ///< constructor
    oscArrayMember(); ///< constructor
    virtual ~oscArrayMember(); ///< destructor
    oscObjectBase * getValue(size_t i) {if(values.size()>i) return values[i]; else return NULL;};
    void setValue(size_t i,oscObjectBase *v) {values[i]=v;};
    void push_back(oscObjectBase *v) {values.push_back(v);};
};

}

#endif //OSC_ARRAY_MEMBER_H