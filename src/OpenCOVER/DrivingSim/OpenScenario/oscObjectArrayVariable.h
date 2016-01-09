/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_ARRAY_VARIABLE_H
#define OSC_OBJECT_ARRAY_VARIABLE_H

#include <oscExport.h>
#include <oscArrayMember.h>
#include <oscMemberValue.h>


namespace OpenScenario
{
    template<typename T>
    class OPENSCENARIOEXPORT oscObjectArrayVariable: public oscArrayMember
    {
    protected:
        T valueT;
    public:
        oscObjectArrayVariable() {type = oscMemberValue::OBJECT; valueT = NULL;}; ///< constructor
        T operator->() {return valueT;};
        oscObjectBase* getObject() const {return valueT;};
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
        void deleteValue() {delete valueT; valueT = NULL;};
        bool exists() const {return valueT != NULL;};
        oscMemberValue::MemberTypes getValueType() const {return type;};
    };
}

#endif //OSC_OBJECT_ARRAY_VARIABLE_H
