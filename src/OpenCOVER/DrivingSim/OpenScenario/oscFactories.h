/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FACTORIES_H
#define OSC_FACTORIES_H

#include "oscFactory.h"
#include "oscMemberValue.h"


namespace OpenScenario
{

class oscObjectBase;

class OPENSCENARIOEXPORT oscFactories
{
    oscFactories();
    ~oscFactories();

    static oscFactories* inst;

public:
    static oscFactories *instance();

    oscFactory<oscObjectBase, std::string> *objectFactory;
    oscFactory<oscMemberValue, oscMemberValue::MemberTypes> *valueFactory;

    void setObjectFactory(oscFactory<oscObjectBase, std::string> *factory); ///< set your own factory in order to create your own classes derived from the original OpenScenario ones
                                                                            ///set back to the default factory if factory is NULL
    void setValueFactory(oscFactory<oscMemberValue, oscMemberValue::MemberTypes> *factory); ///< set your own factory in order to create your own classes derived from the original OpenScenario ones
                                                                                            ///set back to the default factory if factory is NULL

};

}

#endif //OSC_FACTORIES_H
