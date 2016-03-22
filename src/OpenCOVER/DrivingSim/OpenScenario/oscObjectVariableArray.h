/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_VARIABLE_ARRAY_H
#define OSC_OBJECT_VARIABLE_ARRAY_H

#include "oscExport.h"
#include "oscMemberArray.h"
#include "oscMemberValue.h"


namespace OpenScenario
{

template<typename T>
class OPENSCENARIOEXPORT oscObjectVariableArray: public oscMemberArray
{
protected:
    T valueT;

public:
    oscObjectVariableArray() ///< constructor
    {
        type = oscMemberValue::OBJECT;
        valueT = NULL;
    };

    T operator->()
    {
        return valueT;
    };

    oscObjectBase* getObject() const
    {
        return valueT;
    };

    oscObjectBase* getGenerateObject()
    {
        if (!valueT)
        {
            oscObjectBase *obj = oscFactories::instance()->objectFactory->create(typeName);
            if(obj)
            {
                oscMember *member = static_cast<oscMember *>((oscObjectVariable<T>*)this);
                obj->initialize(owner->getBase(), owner, member, owner->getSource());
                setValue(obj);
            }
        }

        return valueT;
    };

    void setValue(oscObjectBase *t)
    {
        if (t != NULL)
        {
            valueT = dynamic_cast<T>(t);
        }
        else
        {
            valueT = NULL;
        }
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

    oscMemberValue::MemberTypes getValueType() const
    {
        return type;
    };
};

}

#endif //OSC_OBJECT_VARIABLE_ARRAY_H
