/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_VARIABLES_H
#define OSC_VARIABLES_H

#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscMember.h>
#include <oscFactories.h>
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
    class oscValue: public oscMemberValue 
    {
    protected:
        T value;
    public:
        oscValue(){};
        oscValue(T &t){value = t;};
        OPENSCENARIOEXPORT virtual bool initialize(xercesc::DOMAttr *);
        virtual oscValue<T>& operator=(T t){value = t; return *this;};
        virtual const T &getValue() const {return value;};
        virtual void setValue(T &t){value = t;};
        OPENSCENARIOEXPORT virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, const char *name);
    };
    
    template<typename T>
    class oscVariable: public oscMember
    {
       T defaultValue;
    public:
        oscVariable(){};
        virtual bool initialize(xercesc::DOMAttr *){return false;};
        virtual T& operator=(T &tv){setValue(tv); return tv;};
        oscValue<T> *getMemberValue() const {return value;};
        void setMemberValue(oscMemberValue *v) {value = v;};
        const T& getValue() const {if(value) {oscValue<T> *ov = dynamic_cast<oscValue<T>*>(value); if(ov!=NULL) return ov->getValue();} return defaultValue;};
        bool exists(){return value!=NULL;};
        void setDefault(T &d) {defaultValue = d;};
        virtual void setValue(T &v) {if(value==NULL) {value = oscFactories::instance()->valueFactory->create(type);} if(value!=NULL) {oscValue<T> *ov = dynamic_cast<oscValue<T>*>(value); if(ov!=NULL) ov->setValue(v);}};
        virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document){if(value!=NULL) { value->writeToDOM(currentElement,document,name.c_str());} return true;};
        OPENSCENARIOEXPORT oscMemberValue::MemberTypes getType();

    };
    
    typedef oscVariable<std::string> oscString;
    typedef oscVariable<int> oscInt;
    typedef oscVariable<unsigned int> oscUInt;
    typedef oscVariable<short> oscShort;
    typedef oscVariable<unsigned short> oscUShort;
    typedef oscVariable<double> oscDouble;
	typedef oscVariable<bool> oscBool;
	typedef oscVariable<float> oscFloat;

    typedef oscValue<std::string> oscStringValue;
    typedef oscValue<int> oscIntValue;
    typedef oscValue<unsigned int> oscUIntValue;
    typedef oscValue<short> oscShortValue;
    typedef oscValue<unsigned short> oscUShortValue;
    typedef oscValue<double> oscDoubleValue;
    typedef oscValue<oscObjectBase *> oscObjectValue;
}


#endif //OSC_VARIABLES_H
