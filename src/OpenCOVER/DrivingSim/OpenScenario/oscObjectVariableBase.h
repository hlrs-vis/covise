/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_VARIABLE_BASE_H
#define OSC_OBJECT_VARIABLE_BASE_H

#include "oscExport.h"
#include "oscMemberValue.h"
#include "oscFactories.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/util/XMLString.hpp>


namespace OpenScenario
{

template<typename T, typename TBase>
class oscObjectVariableBase : public TBase
{
protected:
    T valueT;

public:
    oscObjectVariableBase() : ///< constructor
            valueT(NULL)
    {
        TBase::type = oscMemberValue::OBJECT;
    };

    T operator->()
    {
        return valueT;
    };

    T getObject() const
    {
        return valueT;
    };

    T createObject()
	{
		T obj = static_cast<T>(oscFactories::instance()->objectFactory->create(TBase::typeName));
		if(obj)
		{
			obj->initialize(TBase::owner->getBase(), TBase::owner, this, TBase::owner->getSource());
			setValue(obj);
		}

		return obj;
	};

	T getOrCreateObject()
    {
        if (valueT == NULL)
        {
			createObject();
        }
        return valueT;
    };

    void setValue(T t)
    {
        valueT = t;
    };
    void setValue(oscObjectBase *t)
    {
        valueT = dynamic_cast<T>(t);
    };

    void deleteValue()
    {
        delete valueT;
        valueT = NULL;
    };

    bool exists() const
    {
        return valueT != NULL;
    };

    bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
    {
        if(valueT != NULL)
        {
            xercesc::DOMElement *memberElement = document->createElement(xercesc::XMLString::transcode(TBase::name.c_str()));
            currentElement->appendChild(memberElement);
            valueT->writeToDOM(memberElement,document);
            return true;
        }
        return false;
    };

    oscMemberValue::MemberTypes getValueType() const
    {
        return TBase::type;
    };
};

}

#endif /* OSC_OBJECT_VARIABLE_BASE_H */
