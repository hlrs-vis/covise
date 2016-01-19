/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROUTE_H
#define OSC_ROUTE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT strategyType: public oscEnumType
{
public:
    static strategyType *instance(); 
private:
    strategyType();
    static strategyType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoute: public oscObjectBase
{
public:
    oscRoute()
    {
        OSC_ADD_MEMBER(strategy);

        strategy.enumType = strategyType::instance();
    };
    oscEnum strategy;

    enum strategy
    {
        fastest,
        shortest,
        leastIntersections,
        random,
    };
};

typedef oscObjectVariable<oscRoute *> oscRouteMember;

}

#endif //OSC_ROUTE_H
