/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ROAD_CONDITIONS_H
#define OSC_ROAD_CONDITIONS_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;
class oscRoadConditions;

class OPENSCENARIOEXPORT effectType: public oscEnumType
{
public:
    static effectType *instance(); 
private:
    effectType();
    static effectType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadConditions: public oscObjectBase
{
public:
	oscDouble frictionScale;
	oscDouble intensity;
    enum effect
    {
        dry,
        water,
        snow,
        oil,
        dirt,
		leaves,
    };
	
    oscRoadConditions()
    {
        OSC_ADD_MEMBER(frictionScale);
		OSC_ADD_MEMBER(intensity);
		OSC_ADD_MEMBER(effect);
		effect.enumType = effectType::instance();
    };
	oscEnum effect;
};

typedef oscObjectVariable<oscRoadConditions *> oscRoadConditionsMember;

}

#endif //OSC_ROAD_CONDITIONS_H