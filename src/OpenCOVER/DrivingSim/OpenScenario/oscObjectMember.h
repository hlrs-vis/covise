/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBJECT_MEMBER_H
#define OSC_OBJECT_MEMBER_H

#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscMember.h>
#include <string>
#include <iostream>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END

namespace OpenScenario
{
    template<typename T>
    class OPENSCENARIOEXPORT oscObjectMember: public oscMember 
    {
    protected:
        T valueT;
    public:
        oscObjectMember(){type = oscMemberValue::OBJECT;};
        virtual bool initialize(xercesc::DOMAttr *){return false;};
        virtual T operator->(){return valueT;};
        virtual const T getObject(){return valueT;};
        virtual void setValue(oscObjectBase *t){valueT = dynamic_cast<T>(t);};
        virtual bool exists(){return valueT!=NULL;};
    };
    
}


#endif //OSC_OBJECT_MEMBER_H