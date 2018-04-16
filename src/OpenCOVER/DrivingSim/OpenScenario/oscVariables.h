/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VARIABLES_H
#define OSC_VARIABLES_H

#include "oscExport.h"
#include "oscMemberValue.h"
#include "oscMember.h"
#include "oscFactories.h"

#include <string>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMAttr;
class DOMElement;
class DOMDocument;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario
{

//
template<typename T>
class OPENSCENARIOEXPORT oscValue: public oscMemberValue
{
protected:
    T value;

public:
	oscValue(); ///< constructor

    oscValue(T &t) ///< constructor
    {
        value = t;
    };

    virtual bool initialize(xercesc::DOMAttr *, OpenScenarioBase *base);

    virtual oscValue<T>& operator=(T t)
    {
        value = t;
        return *this;
    };

    virtual const T &getValue() const
    {
        return value;
    };

    virtual void setValue(const T &t)
    {
        value = t;
    };

    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, const char *name);
};

//
template<typename T>
class OPENSCENARIOEXPORT oscVariable: public oscMember
{
private:
    T defaultValue;

public:
    oscVariable()
    {

    };

    virtual const T &operator=(const T &tv)
    {
        setValue(tv);
        return tv;
    };

	operator T() { return getValue(); };

    oscValue<T> *getMemberValue() const
    {
        return dynamic_cast<oscValue<T>*>(value);
    };

    void setMemberValue(oscMemberValue *v)
    {
        value = v;
    };

    const T& getValue() const
    {
        if(value)
        {
            oscValue<T> *ov = dynamic_cast<oscValue<T>*>(value);
            if(ov != NULL)
            {
                return ov->getValue();
            }
        }
        return defaultValue;
    };

    bool exists() const
    {
        return value != NULL;
    };

    void setDefault(T &d)
    {
        defaultValue = d;
    };

    virtual void setValue(const T &v)
    {
        if(value == NULL)
        {
            value = oscFactories::instance()->valueFactory->create(type);
        }
        //value exists and creation was successful
        if(value != NULL)
        {
            oscValue<T> *ov = dynamic_cast<oscValue<T>*>(value);
            if(ov != NULL)
            {
                ov->setValue(v);
            }
        }
    };

    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
    {
        if(value != NULL)
        {
            value->writeToDOM(currentElement, document, name.c_str());
        }
        return true;
    };

    virtual oscMemberValue::MemberTypes getValueType() const;
};

//
class oscEnumType
{
public:
    std::map<std::string, int> enumValues;

    int getEnum(const std::string &n)
    {
        return enumValues[n];
    };

    void addEnum(const std::string &n, const int val)
    {
        enumValues[n] = val;
    };
};

template<>
inline oscMemberValue::MemberTypes oscVariable<int>::getValueType() const {return oscMemberValue::INT;};
template<>
inline oscMemberValue::MemberTypes oscVariable<unsigned int>::getValueType() const {return oscMemberValue::UINT;};
template<>
inline oscMemberValue::MemberTypes oscVariable<short>::getValueType() const {return oscMemberValue::SHORT;};
template<>
inline oscMemberValue::MemberTypes oscVariable<unsigned short>::getValueType() const {return oscMemberValue::USHORT;};
template<>
inline oscMemberValue::MemberTypes oscVariable<std::string>::getValueType() const {return oscMemberValue::STRING;};
template<>
inline oscMemberValue::MemberTypes oscVariable<double>::getValueType() const {return oscMemberValue::DOUBLE;};
template<>
inline oscMemberValue::MemberTypes oscVariable<time_t>::getValueType() const {return oscMemberValue::DATE_TIME;};
template<>
inline oscMemberValue::MemberTypes oscVariable<bool>::getValueType() const {return oscMemberValue::BOOL;};
template<>
inline oscMemberValue::MemberTypes oscVariable<float>::getValueType() const {return oscMemberValue::FLOAT;};

//
typedef oscVariable<std::string> oscString;
typedef oscVariable<int> oscInt;
typedef oscVariable<unsigned int> oscUInt;
typedef oscVariable<short> oscShort;
typedef oscVariable<unsigned short> oscUShort;
typedef oscVariable<double> oscDouble;
typedef oscVariable<bool> oscBool;
typedef oscVariable<float> oscFloat;
typedef oscVariable<time_t> oscDateTime;
//
class OPENSCENARIOEXPORT oscEnum: public oscVariable<int>
{
public:
    oscEnumType *enumType;
    virtual oscMemberValue::MemberTypes getValueType() const;
    void setValueWStr(const std::string &strVal); ///<set the value with the string of the value
    std::string getValueAsStr(const int val) const; ///<get the value as a string
};

//
typedef oscValue<std::string> oscStringValue;
typedef oscValue<int> oscIntValue;
typedef oscValue<unsigned int> oscUIntValue;
typedef oscValue<short> oscShortValue;
typedef oscValue<unsigned short> oscUShortValue;
typedef oscValue<double> oscDoubleValue;
typedef oscValue<bool> oscBoolValue;
typedef oscValue<float> oscFloatValue;
typedef oscValue<time_t> oscDateTimeValue;
typedef oscValue<oscObjectBase *> oscObjectValue;
//
class OPENSCENARIOEXPORT oscEnumValue: public oscValue<int>
{
public:
    oscEnumType *enumType;
    virtual bool initialize(xercesc::DOMAttr *, OpenScenarioBase *base);
    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name);
};

}

#endif //OSC_VARIABLES_H
