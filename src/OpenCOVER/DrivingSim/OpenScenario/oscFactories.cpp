/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscFactories.h>
#include <oscHeader.h>
#include <oscFile.h>
#include <oscDirectory.h>
#include <oscEnvironment.h>
#include <oscWeather.h>
#include <oscTimeOfDay.h>
#include <oscRoadNetwork.h>
#include <oscVariables.h>
#include <oscObjectBase.h>


using namespace OpenScenario;


oscFactory<oscObjectBase,std::string> staticObjectFactory;
oscFactory<oscMemberValue,oscMemberValue::MemberTypes> staticValueFactory;

oscFactories* oscFactories::inst = NULL;

oscFactories::oscFactories()
{
    // set default factories
    objectFactory = &staticObjectFactory;
    valueFactory = &staticValueFactory;

    // register all builtin types
    staticValueFactory.registerType<oscStringValue>(oscMemberValue::STRING);
    staticValueFactory.registerType<oscIntValue>(oscMemberValue::INT);
    staticValueFactory.registerType<oscUIntValue>(oscMemberValue::UINT);
    staticValueFactory.registerType<oscShortValue>(oscMemberValue::SHORT);
    staticValueFactory.registerType<oscUShortValue>(oscMemberValue::USHORT);
    staticValueFactory.registerType<oscDoubleValue>(oscMemberValue::DOUBLE);
    staticValueFactory.registerType<oscEnumValue>(oscMemberValue::ENUM);
    
    staticObjectFactory.registerType<oscHeader>("oscHeader");
    staticObjectFactory.registerType<oscFile>("oscFile");
    staticObjectFactory.registerType<oscDirectory>("oscDirectory");
    staticObjectFactory.registerType<oscEnvironment>("oscEnvironment");
    staticObjectFactory.registerType<oscWeather>("oscWeather");
    staticObjectFactory.registerType<oscTimeOfDay>("oscTimeOfDay");
    staticObjectFactory.registerType<oscRoadNetwork>("oscRoadNetwork");
}


void oscFactories::setObjectFactory(oscFactory<oscObjectBase,std::string> *f)
{
    objectFactory = f;
    if(objectFactory == NULL)
        objectFactory = &staticObjectFactory;
}

void oscFactories::setValueFactory(oscFactory<oscMemberValue,oscMemberValue::MemberTypes> *f)
{
    valueFactory = f;
    if(valueFactory == NULL)
        valueFactory = &staticValueFactory;
}
