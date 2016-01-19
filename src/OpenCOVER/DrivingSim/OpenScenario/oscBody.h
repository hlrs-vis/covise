/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_BODY_H
#define OSC_BODY_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT sexType: public oscEnumType
{
public:
    static sexType *instance(); 
private:
    sexType();
    static sexType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscBody: public oscObjectBase
{
public:
    oscBody()
    {
        OSC_ADD_MEMBER(weight);
        OSC_ADD_MEMBER(height);
        OSC_ADD_MEMBER(eyeDistance);
        OSC_ADD_MEMBER(sex);

        sex.enumType = sexType::instance();
    };

    oscDouble weight;
    oscDouble height;
    oscDouble eyeDistance;
    oscEnum sex;

    enum sex
    {
        male,
        female,
    };
};

typedef oscObjectVariable<oscBody *> oscBodyMember;

}

#endif //OSC_BODY_H
