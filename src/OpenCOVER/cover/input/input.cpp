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

#include <osg/io_utils>
#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRDynLib.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPluginList.h>
#include <net/tokenbuffer.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

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
Input *Input::s_singleton = NULL;

Input *Input::instance()
{
    if (!s_singleton)
        s_singleton = new Input;
    return s_singleton;
}

Input::Input()
: activePerson(NULL)
, m_mouse(NULL)
, m_debug(0)
{
    assert(!s_singleton);
}

bool Input::init()
{
    if (cover->debugLevel(2))
        m_debug |= Input::Config;

    activePerson = NULL;

    initHardware();

    initObjects();
    initPersons();

    if (coVRMSController::instance()->isMaster())
    {
        if (debug(Input::Config))
            printConfig();
    }

    return true;
}

void Input::setDebug(int debugFlags)
{
    m_debug = debugFlags;
}

int Input::getDebug()
{
    return m_debug;
}

bool Input::debug(DebugBits facility)
{
    return (Input::instance()->getDebug() & facility)!=0;
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
    personNames.clear();

    delete m_mouse;

    clearMap(buttondevices);
    clearMap(trackingbodies);
    clearMap(valuators);
    clearMap(drivers);

	for (std::map<std::string, DriverFactoryBase *>::iterator it = plugins.begin();
		it != plugins.end();
		++it)
	{
		DriverFactoryBase *dfb = it->second;
		CO_SHLIB_HANDLE handle = dfb->getLibHandle();
		delete dfb;
		if (handle)
		{
			opencover::coVRDynLib::dlclose(handle); // this has to be called after the destructor, otherwise the code gets unloaded while it is still executing
		}
	}
	plugins.clear();

    s_singleton = NULL;
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

bool Input::isHeadValid() const
{
    if (!activePerson)
        return false;
    return activePerson->isHeadValid();
}

bool Input::hasHand(int num) const
{

    if (!activePerson)
        return false;
    return activePerson->hasHand(num);
}

bool Input::isHandValid(int num) const
{
    if (!activePerson)
        return false;
    return activePerson->isHandValid(num);
}

bool Input::hasRelative() const
{
    if (!activePerson)
        return false;
    return activePerson->hasRelative();
}

bool Input::isRelativeValid() const
{
    if (!activePerson)
        return false;
    return activePerson->isRelativeValid();
}

const osg::Matrix &Input::getHeadMat() const
{

    return activePerson->getHeadMat();
}

const osg::Matrix &Input::getRelativeMat() const
{
    return activePerson->getRelativeMat();
}

const osg::Matrix &Input::getHandMat(int num) const
{

    return activePerson->getHandMat(num);
}

unsigned int Input::getButtonState(int num) const
{

    return activePerson->getButtonState(num);
}

unsigned int Input::getRelativeButtonState(int num) const
{

    return activePerson->getRelativeButtonState(num);
}

double Input::getValuatorValue(size_t idx) const
{

    return activePerson->getValuatorValue(idx);
}

float Input::eyeDistance() const
{
    if (!activePerson)
        return 0.f;
    return activePerson->eyeDistance();
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
        personNames.push_back(name);
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

void Input::addDevice(const std::string &name, InputDevice *dev)
{
	drivers[name] = dev;
	if (dev->needsThread())
		dev->start();
}
void Input::removeDevice(const std::string &name, InputDevice *dev)
{
	drivers.erase(name);
}


InputDevice *Input::getDevice(const std::string &name)
{

    //std::cerr << "Input: looking for dev " << name << std::endl;
    InputDevice *dev = findInMap(drivers, name);
    if (!dev)
    {
        std::string conf = configPath("Device." + name);
        std::string type = coCoviseConfig::getEntry("driver", conf, "const");
        if (type.empty())
        {
            if (coVRPlugin *coverPlugin = coVRPluginList::instance()->addPlugin(name.c_str(), coVRPluginList::Input))
            {
                dev = findInMap(drivers, name);
                if (dev)
                    return dev;
            }
        }
        else if (type != "const")
        {
            if (coVRPlugin *coverPlugin = coVRPluginList::instance()->addPlugin(type.c_str(), coVRPluginList::Input))
            {
                dev = findInMap(drivers, name);
                if (dev)
                    return dev;
            }
        }
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
    std::vector<std::string> names = coCoviseConfig::getScopeNames(configPath("Persons"), "Person");

    if (names.empty())
    {
        if (cover->debugLevel(3))
            cout << "Input: Persons must be configured!" << endl;
        names.push_back("default");
    }

    activePerson = NULL;

    for (size_t i = 0; i < names.size(); ++i)
    {
        Person *p = getPerson(names[i]);

        if (!activePerson)
            activePerson = p;
    }

    return true;
}

bool Input::initObjects()
{

    std::vector<std::string> objects = coCoviseConfig::getScopeNames(configPath("Objects"), "Object");

    for (size_t n = 0; n < objects.size(); ++n)
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

size_t Input::getNumPersons() const
{
    assert(persons.size() == personNames.size());
    return persons.size();
}

size_t Input::getNumBodies() const
{
    return trackingbodies.size();
}

size_t Input::getNumDevices() const
{
    return drivers.size();
}

InputDevice *Input::getDevice(size_t num) //< get driver instance
{
    if (num >= drivers.size())
        return NULL;

    size_t idx = 0;
    for (DriverMap::const_iterator it = drivers.begin();
         it != drivers.end();
         ++it)
    {
        if (num == idx)
        {
            return it->second;
        }
        ++idx;
    }
    return NULL;
}

TrackingBody *Input::getBody(size_t num)
{
    if (num >= trackingbodies.size())
        return NULL;

    size_t idx = 0;
    for (TrackingBodyMap::const_iterator it = trackingbodies.begin();
         it != trackingbodies.end();
         ++it)
    {
        if (num == idx)
        {
            return it->second;
        }
        ++idx;
    }
    return NULL;
}
/**
* @brief Input::setActivePerson Sets an active person
* @param numPerson Number of person
* @return 0 on success
*/
bool Input::setActivePerson(size_t num)
{
    Person *p = getPerson(num);
    if (!p)
    {
        std::cerr << "Input: no person for index " << num << std::endl;
        return false;
    }

    activePerson = p;
    cover->personSwitched(num);
    std::cerr << "Input: switched to person " << activePerson->name() << std::endl;
    return true;
}

Person *Input::getPerson(size_t num)
{

    if (num >= personNames.size())
        return NULL;

    const std::string &name = personNames[num];
    return getPerson(name);
}

size_t Input::getActivePerson() const
{
    const std::string name = activePerson->name();
    std::vector<std::string>::const_iterator it = std::find(personNames.begin(), personNames.end(), name);
    if (it == personNames.end())
    {
        std::cerr << "Input: did not find active person" << std::endl;
        return 0;
    }

    return (it - personNames.begin());
}

/**
 * @brief Input::update Updates all device data. Must be called at the main loop at every frame once
 * @return 0
 */
bool Input::update()
{
    unsigned activePerson = getActivePerson();
    unsigned nBodies = trackingbodies.size(), nButtons = buttondevices.size(), nValuators = valuators.size();
    unsigned int len = 0;
    osg::Matrix mouse = osg::Matrix::identity();

    bool changed = false;

    if (coVRMSController::instance()->isMaster())
    {

        for (DriverMap::iterator it = drivers.begin(); it != drivers.end(); ++it)
        {
            it->second->update();
        }

        TokenBuffer tb;
        tb << activePerson;
        tb << nButtons << nValuators << nBodies;

        { 
            const int oxres = m_mouse->xres, oyres = m_mouse->yres;
            const float owidth = m_mouse->width, oheight = m_mouse->height;
            const int ow0 = m_mouse->wheel(0), ow1 = m_mouse->wheel(1);
            const osg::Matrix omat = m_mouse->getMatrix();
            m_mouse->update();
            if (omat != m_mouse->getMatrix())
                changed = true;
            if (oxres != m_mouse->xres || oyres != m_mouse->yres || owidth != m_mouse->width || oheight != m_mouse->height)
                changed = true;
            if (ow0 != m_mouse->wheel(0) || ow1 != m_mouse->wheel(1))
                changed = true;
            if (debug(Input::Mouse))
            {
                if (debug(Input::Matrices))
                {
                    if (omat != m_mouse->getMatrix())
                    {
                        std::cerr << "Input: transformed Mouse: mat=" << m_mouse->getMatrix() << std::endl;
                    }
                }
                if (oxres != m_mouse->xres || oyres != m_mouse->yres || owidth != m_mouse->width || oheight != m_mouse->height)
                {
                    std::cerr << "Input: transformed Mouse: xres=" << m_mouse->xres << ", yres=" << m_mouse->yres << ", width=" << m_mouse->width << ", height=" << m_mouse->height << std::endl;
                }
                if (debug(Input::Valuators))
                {
                    if (ow0 != m_mouse->wheel(0) || ow1 != m_mouse->wheel(1))
                    {
                        std::cerr << "Input: transformed Mouse: wheel(0)=" << m_mouse->wheel(0) << ", wheel(1)=" << m_mouse->wheel(1) << std::endl;
                    }
                }
            }
            tb << m_mouse->xres << m_mouse->yres << m_mouse->width << m_mouse->height;
            mouse = m_mouse->getMatrix();
            tb << m_mouse->wheel(0) << m_mouse->wheel(1) << m_mouse->x() << m_mouse->y() <<  mouse;
        }

        for (ButtonDeviceMap::iterator ob = buttondevices.begin(); ob != buttondevices.end(); ++ob)
        {
            ButtonDevice *b = ob->second;
            const unsigned old = b->getButtonState();
            b->update();
            if (old != b->getButtonState())
            {
                changed = true;
                if (debug(Input::Transformed) && debug(Input::Buttons))
                {
                    std::cerr << "Input: transformed " << ob->second->name() << " buttons=0x" << std::hex << b->getButtonState() << std::dec << std::endl;
                }
            }
            tb << b->getButtonState();
        }

        for (ValuatorMap::iterator it = valuators.begin(); it != valuators.end(); ++it)
        {
            Valuator *v = it->second;
            const double old = v->getValue();
            v->update();
            if (old != v->getValue())
            {
                changed = true;
                if (debug(Input::Transformed) && debug(Input::Valuators))
                {
                    std::cerr << "Input: transformed " << it->second->name() << " valuator=" << v->getValue() << std::endl;
                }
            }
            tb << v->getValue();
            std::pair<double, double> range = v->getRange();
            tb << range.first << range.second;
        }

        for (TrackingBodyMap::iterator ob = trackingbodies.begin(); ob != trackingbodies.end(); ++ob)
        {
            const osg::Matrix old = ob->second->getMat();
            ob->second->update();
            ob->second->updateRelative();
            if (old != ob->second->getMat())
            {
                changed = true;
                if (debug(Input::Transformed) && debug(Input::Matrices))
                {
                    std::cerr << "Input: transformed " << ob->second->name() << " matrix=" << ob->second->getMat() << std::endl;
                }
            }
            int isVal = ob->second->isValid(), isVar = ob->second->isVarying(), is6Dof = ob->second->is6Dof();
            tb << isVal;
            tb << isVar;
            tb << is6Dof;
            tb << ob->second->getMat();
        }

        tb << changed;

        len = tb.get_length();
        coVRMSController::instance()->syncData(&len, sizeof(len));
        coVRMSController::instance()->syncData(const_cast<char *>(tb.get_data()), len);
    }
    else
    {
        coVRMSController::instance()->syncData(&len, sizeof(len));
        std::vector<char> data(len);
        coVRMSController::instance()->syncData(&data[0], len);
        TokenBuffer tb(&data[0], len);
        tb >> activePerson;
        tb >> nButtons >> nValuators >> nBodies;

        if (activePerson != getActivePerson())
        {
            std::cerr << "Input (id=" << coVRMSController::instance()->getID() << "): active persion is " << getActivePerson() << ", should be " << activePerson << std::endl;
            exit(1);
        }

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

        tb >> m_mouse->xres >> m_mouse->yres >> m_mouse->width >> m_mouse->height;
        tb >> m_mouse->wheelCounter[0] >> m_mouse->wheelCounter[1] >> m_mouse->mouseX >> m_mouse->mouseY >> mouse;
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
            int isVal = 0, isVar = 0, is6Dof = 0;
            tb >> isVal;
            tb >> isVar;
            tb >> is6Dof;
            tb >> mat;
            ob->second->setMat(mat);
            ob->second->setValid(isVal != 0);
            ob->second->setVarying(isVar != 0);
            ob->second->set6Dof(is6Dof != 0);
        }

        tb >> changed;
    }

    for (size_t i=0; i<personNames.size(); ++i) {
        Person *p = getPerson(i);
        if (p->activateOnAction() && (p->isHandValid(0) || p->isHeadValid())) {
            if (i != getActivePerson()) {
                setActivePerson(i);
                p->setActivateOnAction(false);
            }
        }
    }

    return changed;
}

} // namespace opencover
