/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Controller.h"

Controller::Controller(const std::string &setId, const std::string &setName, const std::string &setScriptName, const double &setCycleTime)
    : Element(setId)
    , name(setName)
    , timer(0.0)
    , nextScriptUpdate(0.0)
    , scriptName(setScriptName)
    , cycleTime(setCycleTime)
    , scriptInitialized(false)
    , context(v8::Context::New())
    , context_scope(context)
    , controlTemplate(v8::ObjectTemplate::New())
{
    if (scriptName != "")
    {
        initScript(scriptName);
    }
}

bool Controller::initScript(const std::string &scriptName)
{
    v8::Handle<v8::String> source = readScriptFile(scriptName);
    if (source.IsEmpty())
    {
        std::cout << "Script file " << scriptName << " not found";
        return false;
    }

    script = v8::Script::Compile(source);
    script->Run();

    initFunction = v8::Local<v8::Function>::Cast(context->Global()->Get(v8::String::New("init")));
    updateFunction = v8::Local<v8::Function>::Cast(context->Global()->Get(v8::String::New("update")));

    controlTemplate->SetInternalFieldCount(1);
    controlTemplate->SetAccessor(v8::String::New("green"), Controller::getGreenLight, Controller::switchGreenLight);
    controlTemplate->SetAccessor(v8::String::New("yellow"), Controller::getYellowLight, Controller::switchYellowLight);
    controlTemplate->SetAccessor(v8::String::New("red"), Controller::getRedLight, Controller::switchRedLight);

    scriptInitialized = true;

    return true;
}

void Controller::addTrigger(RoadSensor *sensor, const std::string &functionName)
{
    std::cout << "Adding Trigger function " << functionName << " to sensor " << sensor->getId() << " at s=" << sensor->getS() << std::endl;
    sensor->setTriggerAction(new ControllerRoadSensorTriggerAction(v8::Local<v8::Function>::Cast(context->Global()->Get(v8::String::New(functionName.c_str())))));
}

void Controller::addControl(Control *control)
{
    controlVector.push_back(control);

    v8::Local<v8::Object> controlObject = controlTemplate->NewInstance();
    controlObject->SetInternalField(0, v8::External::New(control));

    std::string controlName = std::string("signal_") + control->getSignal()->getId();
    context->Global()->Set(v8::String::New(controlName.c_str()), controlObject);

    std::cout << "addControl: added control id=" << controlName << std::endl;
}

bool Controller::setScriptParams(const std::string &name, const double &time)
{
    scriptName = name;
    cycleTime = time;

    return initScript(scriptName);
}

const std::string &Controller::getName()
{
    return name;
}

unsigned int Controller::getNumControls()
{
    return controlVector.size();
}

Control *Controller::getControl(unsigned int i)
{
    return controlVector[i];
}

void Controller::init()
{
    v8::Handle<v8::Value> args[] = { v8::Integer::New(0) };
    initFunction->Call(initFunction, 0, args);
}

void Controller::update(const double &dt)
{
    //hard coded script update call at every second
    const double updateTime = 1.0;

    timer += dt;
    if (timer > nextScriptUpdate)
    {
        /*if(timer > cycleTime) {
         timer -= cycleTime;
         nextScriptUpdate += updateTime - cycleTime;
      }
      else {*/
        nextScriptUpdate += updateTime;
        //}

        int time = (int)timer;
        v8::Handle<v8::Value> args[] = { v8::Integer::New(time) };
        updateFunction->Call(updateFunction, 1, args);
    }
}

v8::Handle<v8::String> Controller::readScriptFile(const std::string &name)
{
    FILE *file = fopen(name.c_str(), "rb");
    if (file == NULL)
        return v8::Handle<v8::String>();

    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    rewind(file);

    char *chars = new char[size + 1];
    chars[size] = '\0';
    for (int i = 0; i < size;)
    {
        int read = fread(&chars[i], 1, size - i, file);
        i += read;
    }
    fclose(file);
    v8::Handle<v8::String> result = v8::String::New(chars, size);
    delete[] chars;
    return result;
}

v8::Handle<v8::Value> Controller::getGreenLight(v8::Local<v8::String> property, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    int value = (int)(static_cast<Control *>(wrap->Value())->getGreenLight());
    return v8::Integer::New(value);
}
void Controller::switchGreenLight(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    void *ptr = wrap->Value();
    static_cast<Control *>(ptr)->switchGreenLight(value->Int32Value() != 0);
}

v8::Handle<v8::Value> Controller::getYellowLight(v8::Local<v8::String> property, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    int value = (int)(static_cast<Control *>(wrap->Value())->getYellowLight());
    return v8::Integer::New(value);
}
void Controller::switchYellowLight(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    void *ptr = wrap->Value();
    static_cast<Control *>(ptr)->switchYellowLight(value->Int32Value() != 0);
}

v8::Handle<v8::Value> Controller::getRedLight(v8::Local<v8::String> property, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    int value = (int)(static_cast<Control *>(wrap->Value())->getRedLight());
    return v8::Integer::New(value);
}
void Controller::switchRedLight(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo &info)
{
    v8::Local<v8::Object> self = info.Holder();
    v8::Local<v8::External> wrap = v8::Local<v8::External>::Cast(self->GetInternalField(0));
    void *ptr = wrap->Value();
    static_cast<Control *>(ptr)->switchRedLight(value->Int32Value() != 0);
}
