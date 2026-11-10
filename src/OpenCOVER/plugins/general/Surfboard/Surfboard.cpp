/* This file is part of COVISE, licensed under LGPL 2.1 or later. */
#include "Surfboard.h"

#include <config/CoviseConfig.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRCollaboration.h>
#include <util/SerialCom.h>
#include <osg/Math>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cerrno>

using namespace opencover;

Surfboard::Surfboard()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , coVRNavigationProvider("Surfboard", this)
    , ui::Owner("Surfboard_owner", cover->ui)
    , menu_(new ui::Menu("Surfboard", this))
    , zeroAngleButton_(new ui::Button(menu_, "Zero angle"))
    , turnAccelerationSlider_(new ui::Slider(menu_, "Turn acceleration"))
    , turnDecelerationSlider_(new ui::Slider(menu_, "Turn deceleration"))
{
    turnAccelerationSlider_->setBounds(0.1, 40.0);
    turnAccelerationSlider_->setPresentation(ui::Slider::AsDial);
    turnAccelerationSlider_->setShared(true);
    turnAccelerationSlider_->setValue(turnAcceleration_);
    turnAccelerationSlider_->setCallback([this](double value, bool) {
        turnAcceleration_ = std::max(0.1, value);
    });
    turnDecelerationSlider_->setBounds(0.1, 60.0);
    turnDecelerationSlider_->setPresentation(ui::Slider::AsDial);
    turnDecelerationSlider_->setShared(true);
    turnDecelerationSlider_->setValue(turnDeceleration_);
    turnDecelerationSlider_->setCallback([this](double value, bool) {
        turnDeceleration_ = std::max(0.1, value);
    });

    zeroAngleButton_->setCallback([this](bool pressed) {
        if (!pressed || !serial_ || !coVRMSController::instance()->isMaster())
            return;

        char command[] = "zero\n";
        const int commandLength = static_cast<int>(sizeof(command) - 1);
        if (serial_->write(command, commandLength) != commandLength)
            std::cerr << "Surfboard: failed to send angle zero command" << std::endl;
    });
    coVRNavigationManager::instance()->registerNavigationProvider(this);
}

Surfboard::~Surfboard()
{
    running_ = false;
    if (isRunning())
        join();
    delete serial_;
    coVRNavigationManager::instance()->unregisterNavigationProvider(this);
}

bool Surfboard::init()
{
    device_ = configString("Surfboard", "device",
#ifdef WIN32
                           "\\\\.\\COM3"
#else
                           "/dev/ttyUSB0"
#endif
                           )->value();
    baudRate_ = configInt("Surfboard", "baudRate", 115200)->value();
    maxSpeed_ = std::max(0.0, double(configFloat("Surfboard", "maxSpeed", 1.5f)->value()));
    acceleration_ = std::max(0.0, double(configFloat("Surfboard", "acceleration", 1.0f)->value()));
    deceleration_ = std::max(0.0, double(configFloat("Surfboard", "deceleration", 1.5f)->value()));
    steeringRate_ = std::max(0.0, double(configFloat("Surfboard", "steeringRate", 0.7f)->value()));
    turnRate_ = std::max(0.0, double(configFloat("Surfboard", "turnRate", 4.0f)->value()));
    turnAcceleration_ = std::max(0.1, double(configFloat("Surfboard", "turnAcceleration", 10.0f)->value()));
    turnDeceleration_ = std::max(0.1, double(configFloat("Surfboard", "turnDeceleration", 20.0f)->value()));
    turnAccelerationSlider_->setValue(turnAcceleration_);
    turnDecelerationSlider_->setValue(turnDeceleration_);
    lowAngleDeadZone_ = std::max(0.0, double(configFloat("Surfboard", "deadZone", 2.0f)->value()));
    turnThreshold_ = std::clamp(double(configFloat("Surfboard", "turnThreshold", 45.0f)->value()), 1.0, 179.0);
    buttonRequired_ = configBool("Surfboard", "buttonRequired", true)->value();

    if (!coVRMSController::instance()->isMaster())
        return true;

    serial_ = new covise::SerialCom(device_.c_str(), baudRate_);
    if (serial_->isBad())
    {
        std::cerr << "Surfboard: cannot open serial port '" << device_ << "': "
                  << serial_->errorMessage() << std::endl;
        delete serial_;
        serial_ = nullptr;
        return false;
    }
    running_ = true;
    start();
    std::cerr << "Surfboard: reading USBEncoder lines 'Angle: <degrees> ... GPIO4/5: <state>' from " << device_
              << " at " << baudRate_ << " baud" << std::endl;
    return true;
}

bool Surfboard::parseLine(const std::string &line, double &angle, int &button) const
{
    const auto angleLabel = line.find("Angle:");
    const auto buttonLabel = line.find("GPIO4/5:");
    if (angleLabel == std::string::npos || buttonLabel == std::string::npos)
        return false;

    const char *angleStart = line.c_str() + angleLabel + sizeof("Angle:") - 1;
    char *angleEnd = nullptr;
    errno = 0;
    const double parsedAngle = std::strtod(angleStart, &angleEnd);
    if (angleEnd == angleStart || errno == ERANGE || !std::isfinite(parsedAngle))
        return false;

    const char *buttonStart = line.c_str() + buttonLabel + sizeof("GPIO4/5:") - 1;
    char *buttonEnd = nullptr;
    errno = 0;
    const long parsedButton = std::strtol(buttonStart, &buttonEnd, 10);
    if (buttonEnd == buttonStart || errno == ERANGE || parsedButton < 0 || parsedButton > 255)
        return false;

    angle = parsedAngle;
    button = static_cast<int>(parsedButton);
    return true;
}

void Surfboard::run()
{
    std::string line;
    char c = 0;
    while (running_ && serial_)
    {
        const int count = serial_->read(&c, 1, 1);
        if (count <= 0) continue;
        if ( c == '\r')
        {
            double angle = 0.0;
            int button = 0;
            if (!line.empty() && parseLine(line, angle, button))
            {
                OpenThreads::ScopedLock<OpenThreads::Mutex> lock(dataMutex_);
                angleDegrees_ = std::clamp(angle, -180.0, 180.0);
                button_ = button;
            }
            line.clear();
        }
        else if (line.size() < 128)
            line.push_back(c);
        else
            line.clear();
    }
}

bool Surfboard::update()
{
    if (!isEnabled()) return false;

    double angle;
    int button;
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(dataMutex_);
        angle = angleDegrees_;
        button = button_;
    }

    const double dt = std::clamp(double(cover->frameDuration()), 0.0, 0.1);
    const bool moving = !buttonRequired_ || button != 0;
    const double relativeAngle = std::remainder(angle - zeroAngleDegrees_, 360.0);
    const double absAngle = std::abs(relativeAngle);
    const double sign = relativeAngle < 0.0 ? 1.0 : -1.0;
    const double previousHeading = heading_;

    constexpr double recenterThreshold = 5.0;
    const double maxTurnVelocity = std::max(0.1, turnRate_);
    if (absAngle <= recenterThreshold)
    {
        carveLatched_ = false;
        turnVelocity_ = 0.0;
        carveTarget_ = 0.0;
    }

    if (!carveLatched_ && absAngle > turnThreshold_)
    {
        // Crossing 45 degrees starts a complete, latched turn. The angular
        // speed follows an accelerate/brake profile and settles at +/-180.
        carveLatched_ = true;
        carveTarget_ = sign * osg::PI;
        turnVelocity_ = 0.0;
    }

    if (carveLatched_)
    {
        const double remaining = carveTarget_ - heading_;
        const double direction = remaining < 0.0 ? -1.0 : 1.0;
        const double distance = std::abs(remaining);
        const double brakingVelocity = std::sqrt(2.0 * turnDeceleration_ * distance);
        const double desiredVelocity = std::min(maxTurnVelocity, brakingVelocity);
        const double currentMagnitude = std::abs(turnVelocity_);
        const double velocityStep = (desiredVelocity > currentMagnitude ? turnAcceleration_ : turnDeceleration_) * dt;
        const double newMagnitude = currentMagnitude + std::clamp(desiredVelocity - currentMagnitude,
                                                                   -velocityStep, velocityStep);
        turnVelocity_ = direction * newMagnitude;
        const double headingStep = turnVelocity_ * dt;
        if (std::abs(headingStep) >= distance)
        {
            heading_ = carveTarget_;
            turnVelocity_ = 0.0;
        }
        else
            heading_ += headingStep;
    }
    else
    {
        // Below the carve threshold, heading tracks the measured lean
        // directly, so a small left/right angle changes direction immediately.
        heading_ = osg::DegreesToRadians(relativeAngle * steeringRate_);
        turnVelocity_ = 0.0;
    }

    const double targetSpeed = moving ? maxSpeed_ : 0.0;
    const double speedDelta = targetSpeed - speed_;
    const double speedLimit = (targetSpeed > speed_) ? acceleration_ : deceleration_;
    speed_ += std::clamp(speedDelta, -speedLimit * dt, speedLimit * dt);

    // Small lean drives laterally only. Forward motion starts for the hard
    // carve, as requested; the left/right sign follows the measured angle.
    double dx = 0.0, dy = 0.0;
    if (moving && (carveLatched_ || absAngle > turnThreshold_))
    {
        dx = 0.0;
        dy = speed_ * dt;
    }
    else if (moving && absAngle > lowAngleDeadZone_ && !carveLatched_)
    {
        dx = sign * speed_ * (absAngle / turnThreshold_) * dt;
    }

    osg::Matrix relativeTranslation;
    relativeTranslation.makeTranslate(osg::Vec3(-dx * 1000.0, -dy * 1000.0, 0.0));
    osg::Matrix relativeRotation;
    relativeRotation.makeRotate(-(heading_ - previousHeading), osg::Vec3(0, 0, 1));
    osg::Matrix transform;
    if (coVRMSController::instance()->isMaster())
    {
        // Follow Skateboard: apply the per-frame motion to the current scene
        // transform so navigation from other sources is preserved.
        transform = VRSceneGraph::instance()->getTransform()->getMatrix();
        transform = transform * relativeTranslation * relativeRotation;
        coVRMSController::instance()->sendSlaves(transform.ptr(), sizeof(transform));
    }
    else
        coVRMSController::instance()->readMaster(transform.ptr(), sizeof(transform));
    VRSceneGraph::instance()->getTransform()->setMatrix(transform);
    coVRCollaboration::instance()->SyncXform();
    return (absAngle > 0.01);
}

void Surfboard::setEnabled(bool enabled)
{
    if (enabled && !isEnabled())
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(dataMutex_);
        zeroAngleDegrees_ = angleDegrees_;
        heading_ = 0.0;
        turnVelocity_ = 0.0;
        carveTarget_ = 0.0;
        carveLatched_ = false;
    }
    coVRNavigationProvider::setEnabled(enabled);
    if (!enabled) speed_ = 0.0;
}

COVERPLUGIN(Surfboard)
