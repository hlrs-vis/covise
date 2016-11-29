/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMING_H
#define OSCTIMING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_domain_absolute_relativeType : public oscEnumType
{
public:
static Enum_domain_absolute_relativeType *instance();
    private:
		Enum_domain_absolute_relativeType();
	    static Enum_domain_absolute_relativeType *inst; 
};
class OPENSCENARIOEXPORT oscTiming : public oscObjectBase
{
public:
    oscTiming()
    {
        OSC_ADD_MEMBER(domain);
        OSC_ADD_MEMBER(scale);
        OSC_ADD_MEMBER(offset);
    };
    oscEnum domain;
    oscDouble scale;
    oscDouble offset;

    enum Enum_domain_absolute_relative
    {
absolute,
relative,

    };

};

typedef oscObjectVariable<oscTiming *> oscTimingMember;


}

#endif //OSCTIMING_H
