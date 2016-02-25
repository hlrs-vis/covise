/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TIMING_H
#define OSC_TIMING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT domainType: public oscEnumType
{
public:
    static domainType *instance();
private:
    domainType();
    static domainType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTiming: public oscObjectBase
{
public:
    oscTiming()
    {
        OSC_ADD_MEMBER(scale);
        OSC_ADD_MEMBER(offset);
        OSC_ADD_MEMBER(domain);

        domain.enumType = domainType::instance();
    };

    oscDouble scale;
    oscDouble offset;
    oscEnum domain;

    enum domain
    {
        absolute,
        relative
    };
};

typedef oscObjectVariable<oscTiming *> oscTimingMember;

}

#endif /* OSC_TIMING_H */
