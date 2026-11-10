/* This file is part of COVISE, licensed under LGPL 2.1 or later. */
#ifndef COVISE_OPENCOOVER_SURFBOARD_H
#define COVISE_OPENCOOVER_SURFBOARD_H

#include <cover/coVRPlugin.h>
#include <cover/coVRNavigationProvider.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <osg/Matrix>
#include <string>

namespace covise { class SerialCom; }

class PLUGINEXPORT Surfboard : public opencover::coVRPlugin,
                               public opencover::coVRNavigationProvider,
                               public opencover::ui::Owner,
                               public OpenThreads::Thread
{
public:
    Surfboard();
    ~Surfboard() override;
    bool init() override;
    bool update() override;
    void run() override;
    void setEnabled(bool enabled) override;

private:
    bool parseLine(const std::string &line, double &angle, int &button) const;
    std::string device_;
    int baudRate_ = 115200;
    double maxSpeed_ = 1.5;          // m/s
    double acceleration_ = 1.0;      // m/s^2
    double deceleration_ = 1.5;      // m/s^2
    double steeringRate_ = 0.7;      // direct lean-to-heading scale
    double turnRate_ = 4.0;          // rad/s for the >45 degree carve
    double lowAngleDeadZone_ = 2.0;  // degrees
    double turnThreshold_ = 45.0;    // degrees
    bool buttonRequired_ = true;
    volatile bool running_ = false;
    covise::SerialCom *serial_ = nullptr;
    OpenThreads::Mutex dataMutex_;
    double angleDegrees_ = 0.0;
    int button_ = 0;
    double heading_ = 0.0;           // radians, zero points along +Y
    double speed_ = 0.0;             // m/s
    double zeroAngleDegrees_ = 0.0;
    double turnAcceleration_ = 10.0;  // rad/s^2
    double turnDeceleration_ = 20.0;  // rad/s^2
    double turnVelocity_ = 0.0;      // rad/s
    double carveTarget_ = 0.0;       // radians
    bool carveLatched_ = false;
    opencover::ui::Menu *menu_ = nullptr;
    opencover::ui::Button *zeroAngleButton_ = nullptr;
    opencover::ui::Slider *turnAccelerationSlider_ = nullptr;
    opencover::ui::Slider *turnDecelerationSlider_ = nullptr;
};

#endif
