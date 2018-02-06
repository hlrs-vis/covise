/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * input.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <osg/Matrix>

#include "person.h"
#include "trackingbody.h"
#include "buttondevice.h"
#include "valuator.h"

namespace opencover
{

class InputDevice;
class DriverFactoryBase;
class coMousePointer;

/**
 * @brief The Input class
 *
 * An interface class for input devices
 */
class COVEREXPORT Input
{
public:
    static Input *instance(); //< access input subsystem
    ~Input();
    bool init(); //< initialize input subsytem, legacy driver requires cover/coVRPluginSupport to exist

    enum DebugBits {
        Config = 1,
        Mouse = 2,
        Driver = 4,
        Raw = 8,
        Transformed = 16,
        Matrices = 32,
        Buttons = 64,
        Valuators = 128,
    };
    static bool debug(DebugBits kind);
    void setDebug(int debugFlags /* bitwise or of some DebugBits */);
    int getDebug();

    void printConfig() const; //< debug output

    bool update(); //< global update, call once per frame

    bool isTrackingOn() const;

    bool hasHead() const; //< whether active person's head is tracked
    bool isHeadValid() const; //< whether active person's head matrix is valid
    bool hasHand(int num = 0) const; //< whether active person's hand is tracked
    bool isHandValid(int num = 0) const; //< whether active person's hand matrix is valid
    bool hasRelative() const; //< whether active person has a relative input matrix
    bool isRelativeValid() const; //< whether active person's relative matrix is valid

    //Persons control
    size_t getNumPersons() const; //< number of configured persons
    size_t getNumBodies() const; //< number of bodies
    size_t getNumDevices() const; //< number of Devices
    size_t getActivePerson() const; //< number of current person
    bool setActivePerson(size_t numPerson); //< set active person
    Person *getPerson(const std::string &name);
    Person *getPerson(size_t idx);

    size_t getNumObjects() const
    {
        return objects.size();
    } //< number of tracked objects (not part of persons)

    //Interface for the users of input devices
    const osg::Matrix &getHeadMat() const; //< head matrix of active persion
    const osg::Matrix &getRelativeMat() const; //< relative matrix of active persion
    const osg::Matrix &getHandMat(int num = 0) const; //< hand matrix of active person
    unsigned int getButtonState(int num = 0) const; //< button states of active person's device
    unsigned int getRelativeButtonState(int num = 0) const; //< button states of active person's device
    double getValuatorValue(size_t idx) const; //< valuator values corresponding to current person
    float eyeDistance() const; //< eye distance (in mm) of active person

    DriverFactoryBase *getDriverPlugin(const std::string &name);
	void addDevice(const std::string &name, InputDevice *dev); //< add internal device e.g. from a cover plugin
	void removeDevice(const std::string &name, InputDevice *dev); //< remove internal device e.g. from a cover plugin if the plugin is deleted
    InputDevice *getDevice(const std::string &name); //< get driver instance
    InputDevice *getDevice(size_t idx); //< get driver instance
    TrackingBody *getBody(const std::string &name); //< a single tracked body (matrix)
    TrackingBody *getBody(size_t idx); //< a single tracked body (matrix)
    ButtonDevice *getButtons(const std::string &name); //< state of a set of buttons (e.g. mouse)
    Valuator *getValuator(const std::string &name); //< a single analog value

    coMousePointer *mouse() const;

private:
    Input();
    static Input *s_singleton;

    int m_debug;
    coMousePointer *m_mouse;

    typedef std::map<std::string, Person *> PersonMap;
    PersonMap persons; //< configured persons
    Person *activePerson; //< active person
    std::vector<std::string> personNames; //< ordered list of names of configured persons

    std::vector<TrackingBody *> objects; ///Objects list

    typedef std::map<std::string, TrackingBody *> TrackingBodyMap;
    TrackingBodyMap trackingbodies; //< all configured tracking bodies

    typedef std::map<std::string, ButtonDevice *> ButtonDeviceMap;
    ButtonDeviceMap buttondevices; //< all configured button devices

    typedef std::map<std::string, Valuator *> ValuatorMap;
    ValuatorMap valuators; //< all configured valuators

    typedef std::map<std::string, InputDevice *> DriverMap;
    DriverMap drivers; //< all driver instances

    std::map<std::string, DriverFactoryBase *> plugins; //< all loaded driver plugins

    bool initHardware();
    bool initPersons();
    bool initObjects();

    std::string configPath(std::string src, int n = -1); //< access configuration values
};
}
#endif
