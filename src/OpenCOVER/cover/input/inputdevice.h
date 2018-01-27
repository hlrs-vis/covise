/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputdevice.h
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */

#ifndef INPUT_DEVICE_H
#define INPUT_DEVICE_H

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <osg/Matrix>
#include <osg/Vec3>
#include <vector>
#include <cover/coVRDynLib.h>

namespace opencover
{

/**
 * @brief The InputDevice class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */
class COVEREXPORT InputDevice : public OpenThreads::Thread
{
    friend class Input;
    friend class ButtonDevice;
    friend class Valuator;
    friend class TrackingBody;

public:
    InputDevice(const std::string &configPath);
    virtual ~InputDevice();

    std::string configPath(const std::string &ent = "") const; //< path to config values for this device
    virtual bool poll(); //< called regularly to retrieve new values from hardware - reimplement
    virtual void run(); //< regularly calls poll() on another thread - reimplement if this is not appropriate

    virtual bool needsThread() const; //< whether a thread should be spawned - reimplement if not necessary
    void stopLoop(); //< request run()/other thread to terminate

    bool isValid() const;
    bool isVarying() const;
    bool is6Dof() const;
    const std::string &getName() const;
    const osg::Matrix &getOffsetMat() const;
    void setOffsetMat(const osg::Matrix &m);

    std::string &getCalibrationPointName(int i);
    osg::Vec3 &getCalibrationPoint(int i);

protected:
    static osg::Matrix s_identity; //< identity matrix, for returning a valid reference
    bool loop_is_running; /// If true, the main loop will run
    OpenThreads::Mutex m_mutex; //< protect state data structures

    const std::string m_config; //< path to config values for this device
    std::string m_name;
    osg::Matrix m_offsetMatrix; //< system matrix
    bool m_isVarying; //< whether returned values can change
    bool m_is6Dof; //< whether matrices represent position and orientation
	
    // state data, update during poll(), create them with the correct size
    bool m_valid;
    std::vector<bool> m_buttonStates;
    std::vector<double> m_valuatorValues;
    std::vector<std::pair<double, double> > m_valuatorRanges;
    std::vector<bool> m_bodyMatricesValid;
    std::vector<bool> m_bodyMatricesRelative;
    std::vector<osg::Matrix> m_bodyMatrices;
    osg::Vec3 m_calibrationPoints[3];
    std::string m_calibrationPointNames[3];

    // these are called by Input
    size_t numButtons() const
    {
        return m_buttonStates.size();
    }
    bool getButtonState(size_t idx) const;

    size_t numValuators() const
    {
        return m_valuatorValues.size();
    }
    double getValuatorValue(size_t idx) const;
    std::pair<double, double> getValuatorRange(size_t idx) const;

    size_t numBodies() const
    {
        return m_bodyMatrices.size();
    }
    bool isBodyMatrixValid(size_t idx) const;
    bool isBodyMatrixRelative(size_t idx) const;
    const osg::Matrix &getBodyMatrix(size_t idx) const;

    virtual void update(); //< called by Input::update()

private:
    // per-frame state
    bool m_validFrame;
    std::vector<bool> m_buttonStatesFrame;
    std::vector<double> m_valuatorValuesFrame;
    std::vector<std::pair<double, double> > m_valuatorRangesFrame;
    std::vector<bool> m_bodyMatricesValidFrame;
    std::vector<bool> m_bodyMatricesRelativeFrame;
    std::vector<osg::Matrix> m_bodyMatricesFrame;
};

class COVEREXPORT DriverFactoryBase
{

    friend class Input;

public:
    DriverFactoryBase(const std::string &name);
    virtual ~DriverFactoryBase();

    virtual InputDevice *newInstance(const std::string &name) = 0;
    const std::string &name() const;
    CO_SHLIB_HANDLE getLibHandle() const;

private:
    void setLibHandle(CO_SHLIB_HANDLE handle);

    std::string m_name;
    CO_SHLIB_HANDLE m_handle;
};

template <class Driver>
class DriverFactory : public DriverFactoryBase
{

public:
    DriverFactory(const std::string &name)
        : DriverFactoryBase(name)
    {
    }

    Driver *newInstance(const std::string &instanceName)
    {

        return new Driver(instanceName);
    }
};
}

#define INPUT_PLUGIN(Driver)                                                     \
    extern "C" PLUGINEXPORT opencover::DriverFactory<Driver> *newDriverFactory() \
    {                                                                            \
        return new opencover::DriverFactory<Driver>(#Driver);                    \
    }

#endif /* INPUTHDW_H_ */
