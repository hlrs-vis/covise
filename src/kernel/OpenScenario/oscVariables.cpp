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
#include <xercesc/util/TransService.hpp>

#include "oscObjectBase.h"
#include "OpenScenarioBase.h"


namespace OpenScenario
{

//oscVariable
//
template class  oscVariable<short>;
template class  oscVariable<int>;
template class  oscVariable<unsigned int>;
template class  oscVariable<unsigned short>;
template class  oscVariable<std::string>;
template class  oscVariable<double>;
template class  oscVariable<time_t>;
template class  oscVariable<bool>;
template class  oscVariable<float>;


OPENSCENARIOEXPORT oscMemberValue::MemberTypes oscEnum::getValueType() const {return oscMemberValue::ENUM;};


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

template<>
OPENSCENARIOEXPORT oscValue<double>::oscValue()
{
	type = DOUBLE;
}
template<>
OPENSCENARIOEXPORT oscValue<float>::oscValue()
{
	type = FLOAT;
}
template<>
OPENSCENARIOEXPORT oscValue<bool>::oscValue()
{
	type = BOOL;
}
template<>
OPENSCENARIOEXPORT oscValue<std::string>::oscValue()
{
	type = STRING;
}
template<>
OPENSCENARIOEXPORT oscValue<int>::oscValue()
{
	type = INT;
}
template<>
OPENSCENARIOEXPORT oscValue<unsigned int>::oscValue()
{
	type = UINT;
}
template<>
OPENSCENARIOEXPORT oscValue<short>::oscValue()
{
	type = SHORT;
}
template<>
OPENSCENARIOEXPORT oscValue<unsigned short>::oscValue()
{
	type = USHORT;
}
template<>
OPENSCENARIOEXPORT oscValue<time_t>::oscValue()
{
	type = DATE_TIME;
}
/*
template<>
OPENSCENARIOEXPORT oscValue<oscObjectBase *>::oscValue()
{
	type = OBJECT;
}*/

//oscValue.initialize()
//
template<>
OPENSCENARIOEXPORT bool oscValue<int>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
            value = std::stol(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscValue<unsigned int>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
			value = std::stoul(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscValue<short>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
			value = (short)std::stol(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscValue<unsigned short>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
			value = (unsigned short)std::stoul(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscValue<std::string>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
	char *val = XMLChTranscodeUtf(attribute->getValue());

	if (strcmp(val, "$owner")==0)
	{
		value = val;
	}
	else
	{
	  if (val[0] == '$')
	  {
                base->addParameter(val + 1, this);
	  }
	  else
	  {
	    value = val;
	  }
	}
    xercesc::XMLString::release(&val);
    
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<double>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
			value = std::stod(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscValue<time_t>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
	char *ch;
	std::string valueStr = ch = XMLChTranscodeUtf(attribute->getValue()); xercesc::XMLString::release(&ch);
	if (valueStr[0] == '$')
	{
		base->addParameter(valueStr.c_str() + 1, this);
		return true;
	}
	else
	{
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
	}
    return false;
};
template<>
OPENSCENARIOEXPORT bool oscValue<bool>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
	char *ch;
    std::string valueStr = ch = XMLChTranscodeUtf(attribute->getValue()); xercesc::XMLString::release(&ch);
	if (valueStr[0] == '$')
	{
		base->addParameter(valueStr.c_str() + 1, this);
	}
	else
	{
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
	}

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<float>::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{

    try
    {
		char * val = XMLChTranscodeUtf(attribute->getValue());
		if (val[0] == '$')
		{
			base->addParameter(val + 1, this);
		}
		else
			value = std::stof(val);
		xercesc::XMLString::release(&val);
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
OPENSCENARIOEXPORT bool oscEnumValue::initialize(xercesc::DOMAttr *attribute, OpenScenarioBase *base)
{
	char *ch;
    std::string valstr = ch = XMLChTranscodeUtf(attribute->getValue()); xercesc::XMLString::release(&ch);
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
	XMLCh *t1 = NULL, *t2 = NULL;
    currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(buf)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned int>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%u", value);
	XMLCh *t1 = NULL, *t2 = NULL;
	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(buf)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<short>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%hd",value);
	XMLCh *t1 = NULL, *t2 = NULL;
	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(buf)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<unsigned short>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
    sprintf(buf, "%hu", value);
	XMLCh *t1 = NULL, *t2 = NULL;
	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(buf)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<std::string>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;

	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(value.c_str())); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);

    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<double>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    currentElement->setAttribute(XMLChTranscodeUtf(name), XMLChTranscodeUtf(std::to_string(value).c_str()));
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<time_t>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
    char buf[100];
	if (value <= 0)
		value = time(NULL);
	strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", localtime(&value));
	XMLCh *t1 = NULL, *t2 = NULL;
	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(buf)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<bool>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
	XMLCh *t1 = NULL, *t2 = NULL;
    if(value)
    {
		currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf("true")); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    }
    else
    {
		currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf("false")); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    }
    return true;
};
template<>
OPENSCENARIOEXPORT bool oscValue<float>::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *, const char *name)
{
	XMLCh *t1 = NULL, *t2 = NULL;
	currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(std::to_string(value).c_str())); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
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
			XMLCh *t1 = NULL, *t2 = NULL;
			currentElement->setAttribute(t1 = XMLChTranscodeUtf(name), t2 = XMLChTranscodeUtf(schemaEnumName.c_str())); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        }
    }
    return true;
};

}
