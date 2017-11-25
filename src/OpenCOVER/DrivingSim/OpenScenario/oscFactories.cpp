/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// sort alphabetically
#include "schemaHeaders.h"

using namespace OpenScenario;


oscFactory<oscObjectBase, std::string> oscFactories::staticObjectFactory;
oscFactory<oscMemberValue, oscMemberValue::MemberTypes> oscFactories::staticValueFactory;

oscFactories* oscFactories::inst = NULL;



/*****
 * constructor
 *****/

oscFactories::oscFactories() :
        // set default factories
        objectFactory(&staticObjectFactory),
        valueFactory(&staticValueFactory)
{
    // register all built-in types
    staticValueFactory.registerType<oscStringValue>(oscMemberValue::STRING);
    staticValueFactory.registerType<oscIntValue>(oscMemberValue::INT);
    staticValueFactory.registerType<oscUIntValue>(oscMemberValue::UINT);
    staticValueFactory.registerType<oscShortValue>(oscMemberValue::SHORT);
    staticValueFactory.registerType<oscUShortValue>(oscMemberValue::USHORT);
    staticValueFactory.registerType<oscDoubleValue>(oscMemberValue::DOUBLE);
    staticValueFactory.registerType<oscBoolValue>(oscMemberValue::BOOL);
	staticValueFactory.registerType<oscFloatValue>(oscMemberValue::FLOAT);
	staticValueFactory.registerType<oscDateTimeValue>(oscMemberValue::DATE_TIME);
    staticValueFactory.registerType<oscEnumValue>(oscMemberValue::ENUM);
    
    //register all object types
    // from the schema
#include "registerSchemaObjects.h"
//    staticObjectFactory.registerType<>("");
}



/*****
 * destructor
 *****/

oscFactories::~oscFactories()
{

}



/*****
 * public functions
 *****/

oscFactories *oscFactories::instance()
{
    if(inst == NULL)
    {
        inst = new oscFactories();
    }

    return inst;
}


void oscFactories::setObjectFactory(oscFactory<oscObjectBase, std::string> *f)
{
    objectFactory = f;
    if(objectFactory == NULL)
        objectFactory = &staticObjectFactory;
}

void oscFactories::setValueFactory(oscFactory<oscMemberValue, oscMemberValue::MemberTypes> *f)
{
    valueFactory = f;
    if(valueFactory == NULL)
        valueFactory = &staticValueFactory;
}
