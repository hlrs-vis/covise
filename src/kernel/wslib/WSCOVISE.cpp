/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSCoviseStub.h"

using namespace covise;

//========== Modules & Parameters ==========

covise::covise__Module::covise__Module()
{
}

covise::covise__Module::covise__Module(const covise::covise__Module &m)
    : covise::xsd__anyType()
{
    this->name = m.name;
    this->category = m.category;
    this->host = m.host;
    this->description = m.description;
    this->instance = m.instance;
    this->id = m.id;
    this->title = m.title;
    this->position = m.position;
    for (std::vector<covise__Parameter *>::const_iterator i = m.parameters.begin(); i != m.parameters.end(); ++i)
        this->parameters.push_back((*i)->clone());
    this->inputPorts = m.inputPorts;
    this->outputPorts = m.outputPorts;
}

covise::covise__Module::~covise__Module()
{
}

covise::covise__Parameter::~covise__Parameter()
{
}

covise::covise__Parameter *covise::covise__Parameter::clone() const
{
    return new covise::covise__Parameter(*this);
}

covise::covise__Parameter *covise::covise__BooleanParameter::clone() const
{
    return new covise::covise__BooleanParameter(*this);
}

covise::covise__Parameter *covise::covise__ChoiceParameter::clone() const
{
    return new covise::covise__ChoiceParameter(*this);
}

covise::covise__Parameter *covise::covise__FileBrowserParameter::clone() const
{
    return new covise::covise__FileBrowserParameter(*this);
}

covise::covise__Parameter *covise::covise__FloatScalarParameter::clone() const
{
    return new covise::covise__FloatScalarParameter(*this);
}

covise::covise__Parameter *covise::covise__FloatSliderParameter::clone() const
{
    return new covise::covise__FloatSliderParameter(*this);
}

covise::covise__Parameter *covise::covise__FloatVectorParameter::clone() const
{
    return new covise::covise__FloatVectorParameter(*this);
}

covise::covise__Parameter *covise::covise__IntScalarParameter::clone() const
{
    return new covise::covise__IntScalarParameter(*this);
}

covise::covise__Parameter *covise::covise__IntSliderParameter::clone() const
{
    return new covise::covise__IntSliderParameter(*this);
}

covise::covise__Parameter *covise::covise__IntVectorParameter::clone() const
{
    return new covise::covise__IntVectorParameter(*this);
}

covise::covise__Parameter *covise::covise__StringParameter::clone() const
{
    return new covise::covise__StringParameter(*this);
}

covise::covise__Parameter *covise::covise__ColormapChoiceParameter::clone() const
{
    return new covise::covise__ColormapChoiceParameter(*this);
}

bool covise::operator==(const covise::covise__Colormap &c1, const covise::covise__Colormap &c2)
{
    return std::equal(c1.pins.begin(), c1.pins.end(), c2.pins.begin());
}

bool covise::operator==(const covise::covise__ColormapPin &p1, const covise::covise__ColormapPin &p2)
{
    return p1.a == p2.a && p1.b == p2.b && p1.g == p2.g && p1.a == p2.a && p1.position == p2.position;
}

bool covise::operator!=(const covise::covise__Colormap &c1, const covise::covise__Colormap &c2)
{
    return !(c1 == c2);
}

bool covise::operator!=(const covise::covise__ColormapPin &p1, const covise::covise__ColormapPin &p2)
{
    return !(p1 == p2);
}

//========== Event ==========

covise::covise__Event::covise__Event(const std::string &type)
    : type(type)
{
}

covise::covise__Event::~covise__Event()
{
}

covise::covise__Event *covise::covise__Event::clone() const
{
    return new covise::covise__Event(*this);
}

//========== LinkAddEvent ==========

covise::covise__LinkAddEvent::covise__LinkAddEvent(const covise::covise__Link &link)
    : covise::covise__Event("LinkAdd")
    , link(link)
{
}

covise::covise__LinkAddEvent::~covise__LinkAddEvent()
{
}

covise::covise__Event *covise::covise__LinkAddEvent::clone() const
{
    return new covise::covise__LinkAddEvent(*this);
}

//========== LinkDelEvent ==========

covise::covise__LinkDelEvent::covise__LinkDelEvent(const std::string &linkID)
    : covise::covise__Event("LinkDel")
    , linkID(linkID)
{
}

covise::covise__LinkDelEvent::~covise__LinkDelEvent()
{
}

covise::covise__Event *covise::covise__LinkDelEvent::clone() const
{
    return new covise::covise__LinkDelEvent(*this);
}

//========== AddModuleEvent ==========

covise::covise__ModuleAddEvent::covise__ModuleAddEvent(const covise::covise__Module &module)
    : covise::covise__Event("ModuleAdd")
    , module(module)
{
}

covise::covise__ModuleAddEvent::~covise__ModuleAddEvent()
{
}

covise::covise__Event *covise::covise__ModuleAddEvent::clone() const
{
    return new covise::covise__ModuleAddEvent(*this);
}

//========== DelModuleEvent ==========

covise::covise__ModuleDelEvent::covise__ModuleDelEvent(const std::string &module)
    : covise::covise__Event("ModuleDel")
{
    this->moduleID = module;
}

covise::covise__ModuleDelEvent::~covise__ModuleDelEvent()
{
}

covise::covise__Event *covise::covise__ModuleDelEvent::clone() const
{
    return new covise::covise__ModuleDelEvent(*this);
}

//========== ModuleChangeEvent ==========

covise::covise__ModuleChangeEvent::covise__ModuleChangeEvent(const covise::covise__Module &module)
    : covise::covise__Event("ModuleChange")
    , module(module)
{
}

covise::covise__ModuleChangeEvent::~covise__ModuleChangeEvent()
{
}

covise::covise__Event *covise::covise__ModuleChangeEvent::clone() const
{
    return new covise::covise__ModuleChangeEvent(*this);
}

//========== ModuleDiedEvent ==========

covise::covise__ModuleDiedEvent::covise__ModuleDiedEvent(const std::string &module)
    : covise::covise__Event("ModuleDied")
{
    this->moduleID = module;
}

covise::covise__ModuleDiedEvent::~covise__ModuleDiedEvent()
{
}

covise::covise__Event *covise::covise__ModuleDiedEvent::clone() const
{
    return new covise::covise__ModuleDiedEvent(*this);
}

//========== ModuleExecuteStartEvent ==========

covise::covise__ModuleExecuteStartEvent::covise__ModuleExecuteStartEvent(const std::string &module)
    : covise::covise__Event("ModuleExecuteStart")
{
    this->moduleID = module;
}

covise::covise__ModuleExecuteStartEvent::~covise__ModuleExecuteStartEvent()
{
}

covise::covise__Event *covise::covise__ModuleExecuteStartEvent::clone() const
{
    return new covise::covise__ModuleExecuteStartEvent(*this);
}

//========== ModuleExecuteFinishEvent ==========

covise::covise__ModuleExecuteFinishEvent::covise__ModuleExecuteFinishEvent(const std::string &module)
    : covise::covise__Event("ModuleExecuteFinish")
{
    this->moduleID = module;
}

covise::covise__ModuleExecuteFinishEvent::~covise__ModuleExecuteFinishEvent()
{
}

covise::covise__Event *covise::covise__ModuleExecuteFinishEvent::clone() const
{
    return new covise::covise__ModuleExecuteFinishEvent(*this);
}

//========== ExecuteStartEvent ==========

covise::covise__ExecuteStartEvent::covise__ExecuteStartEvent()
    : covise::covise__Event("ExecuteStart")
{
}

covise::covise__ExecuteStartEvent::~covise__ExecuteStartEvent()
{
}

covise::covise__Event *covise::covise__ExecuteStartEvent::clone() const
{
    return new covise::covise__ExecuteStartEvent(*this);
}

//========== ExecuteFinishEvent ==========

covise::covise__ExecuteFinishEvent::covise__ExecuteFinishEvent()
    : covise::covise__Event("ExecuteFinish")
{
}

covise::covise__ExecuteFinishEvent::~covise__ExecuteFinishEvent()
{
}

covise::covise__Event *covise::covise__ExecuteFinishEvent::clone() const
{
    return new covise::covise__ExecuteFinishEvent(*this);
}

//========== ParameterChangeEvent ==========

covise::covise__ParameterChangeEvent::covise__ParameterChangeEvent(const std::string &moduleID, const covise::covise__Parameter *parameter)
    : covise::covise__Event("ParameterChange")
    , moduleID(moduleID)
    , parameter(parameter->clone())
{
}

covise::covise__ParameterChangeEvent::covise__ParameterChangeEvent(const covise::covise__ParameterChangeEvent &parameter)
    : covise::covise__Event("ParameterChange")
{
    this->moduleID = parameter.moduleID;
    this->parameter = parameter.parameter->clone();
    this->type = parameter.type;
}

covise::covise__ParameterChangeEvent::~covise__ParameterChangeEvent()
{
}

covise::covise__Event *covise::covise__ParameterChangeEvent::clone() const
{
    return new covise::covise__ParameterChangeEvent(*this);
}

//========== OpenNetEvent ==========

covise::covise__OpenNetEvent::covise__OpenNetEvent(const std::string &mapname)
    : covise::covise__Event("OpenNet")
    , mapname(mapname)
{
}

covise::covise__OpenNetEvent::~covise__OpenNetEvent()
{
}

covise::covise__Event *covise::covise__OpenNetEvent::clone() const
{
    return new covise::covise__OpenNetEvent(*this);
}

//========== OpenNetDoneEvent ==========

covise::covise__OpenNetDoneEvent::covise__OpenNetDoneEvent(const std::string &mapname)
    : covise::covise__Event("OpenNetDone")
    , mapname(mapname)
{
}

covise::covise__OpenNetDoneEvent::~covise__OpenNetDoneEvent()
{
}

covise::covise__Event *covise::covise__OpenNetDoneEvent::clone() const
{
    return new covise::covise__OpenNetDoneEvent(*this);
}

//========== QuitEvent ==========

covise::covise__QuitEvent::covise__QuitEvent()
    : covise::covise__Event("Quit")
{
}

covise::covise__QuitEvent::~covise__QuitEvent()
{
}

covise::covise__Event *covise::covise__QuitEvent::clone() const
{
    return new covise::covise__QuitEvent(*this);
}
