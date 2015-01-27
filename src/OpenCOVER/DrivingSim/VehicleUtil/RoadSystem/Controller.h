/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Controller_h
#define Controller_h

#include "Element.h"
#include "RoadSignal.h"
#include "RoadSensor.h"
#include "Control.h"

#include <vector>
#ifdef None
#undef None
#endif
#ifdef True
#undef True
#endif
#ifdef False
#undef False
#endif
#include <v8.h>

class Controller : public Element
{
public:
    Controller(const std::string &, const std::string &, const std::string &, const double &);

    void addControl(Control *);

    void addTrigger(RoadSensor *, const std::string &);

    const std::string &getName();

    unsigned int getNumControls();
    Control *getControl(unsigned int i);

    void init();
    void update(const double &dt);

protected:
    v8::Handle<v8::String> readScriptFile(const std::string &);
    static v8::Handle<v8::Value> getGreenLight(v8::Local<v8::String>, const v8::AccessorInfo &);
    static void switchGreenLight(v8::Local<v8::String>, v8::Local<v8::Value>, const v8::AccessorInfo &);
    static v8::Handle<v8::Value> getYellowLight(v8::Local<v8::String>, const v8::AccessorInfo &);
    static void switchYellowLight(v8::Local<v8::String>, v8::Local<v8::Value>, const v8::AccessorInfo &);
    static v8::Handle<v8::Value> getRedLight(v8::Local<v8::String>, const v8::AccessorInfo &);
    static void switchRedLight(v8::Local<v8::String>, v8::Local<v8::Value>, const v8::AccessorInfo &);

    std::string name;

    std::vector<Control *> controlVector;

    double timer;
    double nextScriptUpdate;

    std::string scriptName;
    double cycleTime;

    v8::HandleScope handle_scope;
    v8::Persistent<v8::Context> context;
    v8::Context::Scope context_scope;
    v8::Local<v8::Script> script;
    v8::Local<v8::Function> initFunction;
    v8::Local<v8::Function> updateFunction;
    v8::Persistent<v8::ObjectTemplate> controlTemplate;
};

class ControllerRoadSensorTriggerAction : public RoadSensorTriggerAction
{
public:
    ControllerRoadSensorTriggerAction(const v8::Local<v8::Function> &triggerJSFunction_)
        : triggerJSFunction(triggerJSFunction_)
    {
    }

    void operator()(const std::string &info)
    {
        v8::Handle<v8::Value> args[] = { v8::String::New(info.c_str()) };
        triggerJSFunction->Call(triggerJSFunction, 1, args);
    }

protected:
    v8::Local<v8::Function> triggerJSFunction;
};

#endif
