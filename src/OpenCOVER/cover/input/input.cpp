/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * input.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRDynLib.h>
#include <cover/coVRPluginSupport.h>
#include <net/tokenbuffer.h>

#include "coMousePointer.h"
#include "input.h"
#include "inputdevice.h"
#include "input_const.h"

#include <iostream>
#include <sstream>
#include <cassert>

using namespace std;
using namespace covise;

namespace opencover
{

Input *Input::instance()
{
    static Input *singleton = NULL;
    if (!singleton)
        singleton = new Input;
    return singleton;
}

Input::Input()
{
    activePerson = NULL;
}

bool Input::init()
{
    activePerson = NULL;

    initHardware();

    initObjects();
    initPersons();

    setActivePerson(0);

    update();

    if (coVRMSController::instance()->isMaster())
    {
        if (cover->debugLevel(2))
            printConfig();
    }

    return true;
}

coMousePointer *Input::mouse() const
{

    return m_mouse;
}

namespace
{

    template <typename Map>
    void clearMap(Map &map)
    {
        for (typename Map::iterator it = map.begin();
             it != map.end();
             ++it)
        {
            delete it->second;
        }
        map.clear();
    }
}

Input::~Input()
{
    persons.clear();

    delete m_mouse;

    clearMap(buttondevices);
    clearMap(trackingbodies);
    clearMap(valuators);
    clearMap(drivers);
    clearMap(plugins);
}

void Input::printConfig() const
{

    std::cout << "plugins:";
    for (std::map<std::string, DriverFactoryBase *>::const_iterator p = plugins.begin(); p != plugins.end(); p++)
    {
        std::cout << " " << p->first;
    }
    std::cout << std::endl;

    std::cout << "devices:";
    for (std::map<std::string, InputDevice *>::const_iterator d = drivers.begin(); d != drivers.end(); d++)
    {
        std::cout << " " << d->first << "(b:" << d->second->numButtons() << ", v:" << d->second->numValuators() << ", m:" << d->second->numBodies() << ")";
    }
    std::cout << std::endl;

    std::cout << "buttons:";
    for (std::map<std::string, ButtonDevice *>::const_iterator b = buttondevices.begin(); b != buttondevices.end(); b++)
    {
        std::cout << " " << b->first;
    }
    std::cout << std::endl;

    std::cout << "bodies:";
    for (std::map<std::string, TrackingBody *>::const_iterator b = trackingbodies.begin(); b != trackingbodies.end(); b++)
    {
        std::cout << " " << b->first;
    }
    std::cout << std::endl;

    std::cout << "persons: num=" << persons.size() << ", active=" << activePerson->name() << ", head=" << hasHead() << ", hand=" << hasHand() << std::endl;
}

bool Input::isTrackingOn() const
{

    if (!activePerson)
        return false;

    return (hasHand() || hasHead()) && activePerson->isVarying();
}

bool Input::hasHead() const
{

    if (!activePerson)
        return false;
    return activePerson->hasHead();
}

bool Input::hasHand(int num) const
{

    if (!activePerson)
        return false;
    return activePerson->hasHand(num);
}

const osg::Matrix &Input::getHeadMat() const
{

    return activePerson->getHeadMat();
}

const osg::Matrix &Input::getHandMat(int num) const
{

    return activePerson->getHandMat(num);
}

unsigned int Input::getButtonState(int num) const
{

    return activePerson->getButtonState(num);
}

double Input::getValuatorValue(size_t idx) const
{

    return activePerson->getValuatorValue(idx);
}

/**
 * @brief Input::configPath Helper function for config file reading
 * @param src Config section string
 * @param n "name" number
 * @return concat string+number
 */
string Input::configPath(string src, int n)
{
    stringstream sstr;
    sstr << "COVER.Input";
    if (!src.empty())
        sstr << "." << src;
    if (n >= 0)
        sstr << ":" << n;
    return sstr.str();
}

bool Input::initHardware()
{
    plugins["const"] = new DriverFactory<ConstInputDevice>("const");

    m_mouse = new coMousePointer;

    return true;
}

namespace
{

    template <typename Map>
    typename Map::mapped_type findInMap(const Map &map, const typename Map::key_type &key)
    {
        typename Map::const_iterator it = map.find(key);
        if (it == map.end())
            return NULL;

        return it->second;
    }
}

Person *Input::getPerson(const std::string &name)
{

    if (name.empty())
        return NULL;

    Person *person = findInMap(persons, name);
    if (!person)
    {
        person = new Person(name);
        persons[name] = person;
    }
    return person;
}

TrackingBody *Input::getBody(const std::string &name)
{

    if (name.empty())
        return NULL;

    TrackingBody *body = findInMap(trackingbodies, name);
    if (!body)
    {
        body = new TrackingBody(name);
        trackingbodies[name] = body;
    }
    return body;
}

ButtonDevice *Input::getButtons(const std::string &name)
{

    if (name.empty())
        return NULL;

    ButtonDevice *buttons = findInMap(buttondevices, name);
    if (!buttons)
    {
        buttons = new ButtonDevice(name);
        buttondevices[name] = buttons;
    }
    return buttons;
}

Valuator *Input::getValuator(const std::string &name)
{

    if (name.empty())
        return NULL;

    Valuator *val = findInMap(valuators, name);
    if (!val)
    {
        val = new Valuator(name);
        valuators[name] = val;
    }
    return val;
}

DriverFactoryBase *Input::getDriverPlugin(const std::string &type)
{

    DriverFactoryBase *fact = findInMap(plugins, type);
    if (!fact)
    {
        CO_SHLIB_HANDLE handle = coVRDynLib::dlopen(coVRDynLib::libName(std::string("input_") + type));
        if (!handle)
        {
            std::cerr << "failed to open device driver plugin: " << coVRDynLib::dlerror() << std::endl;
            return NULL;
        }
        typedef DriverFactoryBase *(*NewDriverFactory)(void);
        NewDriverFactory newDriverFactory = (NewDriverFactory)coVRDynLib::dlsym(handle, "newDriverFactory");
        if (!newDriverFactory)
        {
            std::cerr << "malformed device driver plugin: no newDriverFactory function" << std::endl;
            coVRDynLib::dlclose(handle);
            return NULL;
        }
        fact = newDriverFactory();
        fact->setLibHandle(handle);
        plugins[type] = fact;
    }
    return fact;
}

InputDevice *Input::getDevice(const std::string &name)
{

    //std::cerr << "Input: looking for dev " << name << std::endl;
    InputDevice *dev = findInMap(drivers, name);
    if (!dev)
    {
        std::string conf = configPath("Device." + name);
        std::string type = coCoviseConfig::getEntry("driver", conf, "const");
        //std::cerr << "Input: creating dev " << name << ", driver " << type << std::endl;
        DriverFactoryBase *plug = getDriverPlugin(coVRMSController::instance()->isMaster() ? type : "const");
        if (!plug)
        {
            std::cerr << "Input: replaced driver " << name << " with \"const\"" << std::endl;
            plug = getDriverPlugin("const");
        }
        if (plug)
        {
            dev = plug->newInstance("COVER.Input.Device." + name);
            drivers[name] = dev;
            if (dev->needsThread())
                dev->start();
        }
    }
    return dev;
}

bool Input::initPersons()
{
    std::vector<std::string> personNames = coCoviseConfig::getScopeNames(configPath("Persons"), "Person");

    if (personNames.empty())
    {
        cout << "Input: Persons must be configured!" << endl;
        personNames.push_back("default");
    }

    activePerson = NULL;

    for (size_t i = 0; i < personNames.size(); ++i)
    {

        if (!activePerson)
            activePerson = getPerson(personNames[i]);
    }

    return true;
}

bool Input::initObjects()
{

    std::vector<std::string> objects = coCoviseConfig::getScopeNames(configPath("Objects"), "Object");

    for (int n = 0; n < objects.size(); ++n)
    {

        string conf = configPath("Objects.Object:" + objects[n]); //config string for reading this object config
        string bodyName = coCoviseConfig::getEntry("body", conf, "");
        TrackingBody *body = getBody(bodyName);
        if (!body)
            std::cerr << "Input: did not find body " << bodyName << std::endl;
    }

    return true;
}

//=======================================End of init section================================================

/**
* @brief Input::setActivePerson Sets an active person
* @param numPerson Number of person
* @return 0 on success
*/
bool Input::setActivePerson(size_t num)
{
    Person *p = getPerson(num);
    if (!p)
        return false;

    activePerson = p;
    return true;
}

Person *Input::getPerson(size_t num) const
{

    if (num >= persons.size())
        return NULL;

    size_t idx = 0;
    for (PersonMap::const_iterator it = persons.begin();
         it != persons.end();
         ++it)
    {
        if (num == idx)
        {
            return it->second;
        }
    }

    return NULL;
}

/**
 * @brief Input::update Updates all device data. Must be called at the main loop at every frame once
 * @return 0
 */
void Input::update()
{
    unsigned nBodies = trackingbodies.size(), nButtons = buttondevices.size(), nValuators = valuators.size();
    unsigned int len = 0;
    osg::Matrix mouse = osg::Matrix::identity();
    if (coVRMSController::instance()->isMaster())
    {

        for (auto d : drivers)
            d.second->update();

        TokenBuffer tb;
        tb << nButtons << nValuators << nBodies;

        m_mouse->update();
        tb << m_mouse->wheel(0) << m_mouse->wheel(1);
        mouse = m_mouse->getMatrix();
        for (int i = 0; i < 16; ++i)
        {
            tb << mouse(i / 4, i % 4);
        }

        for (ButtonDeviceMap::iterator ob = buttondevices.begin(); ob != buttondevices.end(); ++ob)
        {
            ButtonDevice *b = ob->second;
            b->update();
            tb << b->getButtonState();
        }

        for (ValuatorMap::iterator it = valuators.begin(); it != valuators.end(); ++it)
        {
            Valuator *v = it->second;
            v->update();
            tb << v->getValue();
            std::pair<double, double> range = v->getRange();
            tb << range.first << range.second;
        }

        for (TrackingBodyMap::iterator ob = trackingbodies.begin(); ob != trackingbodies.end(); ++ob)
        {
            ob->second->update();
            int isVar = ob->second->isVarying(), is6Dof = ob->second->is6Dof();
            tb << isVar;
            tb << is6Dof;
            for (int i = 0; i < 16; ++i)
                tb << ob->second->getMat().ptr()[i];
        }

        len = tb.get_length();
        coVRMSController::instance()->sendSlaves(&len, sizeof(len));
        coVRMSController::instance()->sendSlaves(tb.get_data(), len);
    }
    else
    {
        coVRMSController::instance()->readMaster(&len, sizeof(len));
        char *data = new char[len];
        coVRMSController::instance()->readMaster(data, len);
        TokenBuffer tb(data, len);
        tb >> nButtons >> nValuators >> nBodies;

        if (nButtons != buttondevices.size())
        {
            std::cerr << "Input (id=" << coVRMSController::instance()->getID() << "): buttondevices.size() is " << buttondevices.size() << ", should be " << nButtons << std::endl;
            exit(1);
        }

        if (nValuators != valuators.size())
        {
            std::cerr << "Input (id=" << coVRMSController::instance()->getID() << "): valuators.size() is " << valuators.size() << ", should be " << nValuators << std::endl;
            exit(1);
        }

        if (nBodies != trackingbodies.size())
        {
            std::cerr << "Input (id=" << coVRMSController::instance()->getID() << "): trackingbodies.size() is " << trackingbodies.size() << ", should be " << nBodies << std::endl;
            exit(1);
        }

        tb >> m_mouse->wheelCounter[0] >> m_mouse->wheelCounter[1];
        for (int i = 0; i < 16; ++i)
        {
            tb >> mouse(i / 4, i % 4);
        }
        m_mouse->setMatrix(mouse);

        for (ButtonDeviceMap::iterator ob = buttondevices.begin(); ob != buttondevices.end(); ++ob)
        {
            unsigned int bs;
            tb >> bs;
            ob->second->setButtonState(bs);
        }

        for (ValuatorMap::iterator it = valuators.begin(); it != valuators.end(); ++it)
        {
            Valuator *v = it->second;
            double value = 0., min = 0., max = 0.;
            tb >> value >> min >> max;
            v->setValue(value);
            v->setRange(min, max);
        }

        for (TrackingBodyMap::iterator ob = trackingbodies.begin(); ob != trackingbodies.end(); ++ob)
        {
            osg::Matrix mat;
            int isVar = 0, is6Dof = 0;
            tb >> isVar;
            tb >> is6Dof;
            for (int i = 0; i < 16; ++i)
                tb >> mat.ptr()[i];
            ob->second->setMat(mat);
            ob->second->setVarying(isVar != 0);
            ob->second->set6Dof(is6Dof != 0);
        }
    }
}
}
