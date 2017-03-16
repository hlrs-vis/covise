/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscVariables.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/util/XMLString.hpp>


namespace OpenScenario
{

//oscVariable
//
template class  OPENSCENARIOEXPORT oscVariable<short>;
template class  OPENSCENARIOEXPORT oscVariable<int>;
template class  OPENSCENARIOEXPORT oscVariable<unsigned int>;
template class  OPENSCENARIOEXPORT oscVariable<unsigned short>;
template class  OPENSCENARIOEXPORT oscVariable<std::string>;
template class  OPENSCENARIOEXPORT oscVariable<double>;
template class  OPENSCENARIOEXPORT oscVariable<time_t>;
template class  OPENSCENARIOEXPORT oscVariable<bool>;
template class  OPENSCENARIOEXPORT oscVariable<float>;

template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<int>::getValueType() {return oscMemberValue::INT;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<unsigned int>::getValueType() {return oscMemberValue::UINT;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<short>::getValueType() {return oscMemberValue::SHORT;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<unsigned short>::getValueType() {return oscMemberValue::USHORT;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<std::string>::getValueType() {return oscMemberValue::STRING;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<double>::getValueType() {return oscMemberValue::DOUBLE;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<time_t>::getValueType() {return oscMemberValue::DATE_TIME;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<bool>::getValueType() {return oscMemberValue::BOOL;};
template<>
OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscVariable<float>::getValueType() {return oscMemberValue::FLOAT;};

OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscEnum::getValueType() {return oscMemberValue::ENUM;};


//oscEnum
//
void oscEnum::setValueWStr(const std::string &strVal)
{
    int val = enumType->getEnum(strVal);
    setValue(val);
}

std::string oscEnum::getValueAsStr(const int val) const
{
    std::string strVal;
    for(auto &it : enumType->enumValues)
    {
        if(it.second == val)
        {
            strVal = it.first;
        }
    }

    return strVal;
}


//oscValue.initialize()
//
template<>
OPENSCENARIOEXPORT bool oscValue<int>::initialize(xercesc::DOMAttr *attribute)
{
    try
    {
        value = std::stol(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of an int value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of an int value." << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned int>::initialize(xercesc::DOMAttr *attribute)
{
    try
    {
        value = std::stoul(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of an unsigned int value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of an unsigned int value." << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<short>::initialize(xercesc::DOMAttr *attribute)
{
    try
    {
        value = (short)std::stol(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of a shot value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of a short value." << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned short>::initialize(xercesc::DOMAttr *attribute)
{
    try
    {
        value = (unsigned short)std::stoul(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of an unsigned shot value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of an unsigned short value." << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<std::string>::initialize(xercesc::DOMAttr *attribute)
{
    value = xercesc::XMLString::transcode(attribute->getValue());
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<double>::initialize(xercesc::DOMAttr *attribute)
{
    try
    {
        value = std::stod(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of a double value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of a double value." << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<time_t>::initialize(xercesc::DOMAttr *attribute)
{

	std::string valueStr = xercesc::XMLString::transcode(attribute->getValue());
	struct tm t = {};
	//strptime(valueStr.c_str(),"%FT%TZ", gmtime(&value));
	int ns = sscanf(valueStr.c_str(), "%d-%d-%dT%d:%d:%d", &t.tm_year, &t.tm_mon, &t.tm_mday, &t.tm_hour, &t.tm_min, &t.tm_sec);
	t.tm_year -= 1900;
	t.tm_mon -= 1;
	t.tm_isdst = -1;
	if (ns < 6) {
		std::cerr << "Error! Trying to initialize a time_t value." << std::endl;
		return false;
	}
	else {
		value = std::mktime(&t);
		return true;
	}
    return false;
};
template<>
OPENSCENARIOEXPORT bool oscValue<bool>::initialize(xercesc::DOMAttr *attribute)
{
    std::string valueStr = xercesc::XMLString::transcode(attribute->getValue());
	
    //conversion of 'true' to '1' and 'false' to '0'
    if (valueStr == "true")
    {
        valueStr = "1";
    }
    else if (valueStr == "false")
    {
        valueStr = "0";
    }

    try
    {
        value = (std::stol(valueStr) != 0);
    }
    catch (...)
    {
        std::cerr << " Error during conversion of string value \"" << valueStr << "\" to boolean." << std::endl;
        std::cerr << " Known values are: 'true', 'false', '1', '0'" << std::endl;
        return false;
    }

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<float>::initialize(xercesc::DOMAttr *attribute)
{

    try
    {
        value = std::stof(xercesc::XMLString::transcode(attribute->getValue()));
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Error during initialization of a float value. Invalid argument: " << ia.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Error during initialization of a float value." << std::endl;
        return false;
    }

    return true;
};

//oscEnumValue.initialize()
//
OPENSCENARIOEXPORT bool oscEnumValue::initialize(xercesc::DOMAttr *attribute)
{
    std::string valstr = xercesc::XMLString::transcode(attribute->getValue()); 
	std::string enumName = nameMapping::instance()->getEnumName(valstr);
    value = enumType->getEnum(enumName);
    return true;
};


//oscValue.writeToDOM()
//
template<>
OPENSCENARIOEXPORT bool oscValue<int>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%d", value);
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(buf));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned int>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%u", value);
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(buf));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<short>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%hd",value);
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(buf));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned short>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%hu", value);
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(buf));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<std::string>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(value.c_str()));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<double>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(std::to_string(value).c_str()));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<time_t>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
	strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", localtime(&value));
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(buf));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<bool>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    if(value)
    {
        currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode("true"));
    }
    else
    {
        currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode("false"));
    }
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<float>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(std::to_string(value).c_str()));
    return true;
};

//oscEnumValue.writeToDOM()
//
OPENSCENARIOEXPORT bool oscEnumValue::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    for(std::map<std::string,int>::iterator it = enumType->enumValues.begin();it != enumType->enumValues.end(); it++)
    {
        if(it->second == value)
        {
			std::string s = it->first.c_str();
			std::string schemaEnumName = nameMapping::instance()->getSchemaEnumName(s);
            currentElement->setAttribute(xercesc::XMLString::transcode(name), xercesc::XMLString::transcode(schemaEnumName.c_str()));
        }
    }
    return true;
};

}
