/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPRECIPITATION_H
#define OSCPRECIPITATION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Precipitation_typeType : public oscEnumType
{
public:
static Enum_Precipitation_typeType *instance();
    private:
		Enum_Precipitation_typeType();
	    static Enum_Precipitation_typeType *inst; 
};
class OPENSCENARIOEXPORT oscPrecipitation : public oscObjectBase
{
public:
oscPrecipitation()
{
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(intensity);
        type.enumType = Enum_Precipitation_typeType::instance();
    };
    oscEnum type;
    oscDouble intensity;

    enum Enum_Precipitation_type
    {
dry,
rain,
snow,

    };

};

typedef oscObjectVariable<oscPrecipitation *> oscPrecipitationMember;
typedef oscObjectVariableArray<oscPrecipitation *> oscPrecipitationArrayMember;


}

#endif //OSCPRECIPITATION_H
