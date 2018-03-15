/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "person.h"
#include "input.h"
#include "../coVRConfig.h"
#include <config/CoviseConfig.h>
#include <sstream>

using namespace covise;

namespace opencover
{

const osg::Matrix Person::s_identity = osg::Matrix::identity();

Person::Person(const std::string &name)
    : m_name(name)
    , m_head(NULL)
    , m_buttondev(NULL)
    , m_eyeDistance(0.)
    , m_activateOnAction(false)
{
    const std::string conf = "COVER.Input.Persons.Person:" + name;

    m_eyeDistance = coVRConfig::instance()->stereoSeparation();
    if (coVRConfig::instance()->stereoState())
    {
        m_eyeDistance = coCoviseConfig::getFloat("eyeDistance", conf, m_eyeDistance);
    }

    m_head = Input::instance()->getBody(coCoviseConfig::getEntry("head", conf, ""));
    m_relative = Input::instance()->getBody(coCoviseConfig::getEntry("relative", conf, ""));
    m_activateOnAction = coCoviseConfig::isOn("activateOnAction", conf, m_activateOnAction);

    for (int i = 0; i < 4; ++i)
    {
        std::stringstream str;
        str << "hand" << i;
        std::string hand = coCoviseConfig::getEntry(str.str(), conf, "");
        if (TrackingBody *body = Input::instance()->getBody(hand))
        {
            addHand(body);
        }
        else
        {
            break;
        }
    }
    if (m_hands.empty())
    {
        TrackingBody *hand = Input::instance()->getBody(coCoviseConfig::getEntry("hand", conf, ""));
        if (hand)
            addHand(hand);
        TrackingBody *hand2 = Input::instance()->getBody(coCoviseConfig::getEntry("secondHand", conf, ""));
        if (hand2)
            addHand(hand2);
    }

    m_buttondev = Input::instance()->getButtons(coCoviseConfig::getEntry("buttons", conf, ""));
    m_relativebuttondev = Input::instance()->getButtons(coCoviseConfig::getEntry("relativeButtons", conf, ""));

    for (int i = 0; i < 4; ++i)
    {
        std::stringstream str;
        str << "valuator" << i;
        std::string val = coCoviseConfig::getEntry(str.str(), conf, "");
        if (Valuator *valuator = Input::instance()->getValuator(val))
        {
            addValuator(valuator);
        }
        else
        {
            break;
        }
    }
}

std::string Person::name() const
{

    return m_name;
}

void Person::addHand(TrackingBody *hand)
{

    m_hands.push_back(hand);
}

void Person::addValuator(Valuator *val)
{

    m_valuators.push_back(val);
}

bool Person::hasHead() const
{
    return m_head != NULL;
}

bool Person::isHeadValid() const
{
    return hasHead() && m_head->isValid();
}

bool Person::hasHand(size_t num) const
{
    return m_hands.size() > num && m_hands[num];
}

bool Person::isHandValid(size_t idx) const
{
    return hasHand(idx) && m_hands[idx]->isValid();
}

bool Person::isVarying() const
{
    if (hasHead() && m_head->isVarying())
    {
        return true;
    }

    for (size_t i = 0; i < m_hands.size(); ++i)
    {
        if (m_hands[i]->isVarying())
        {
            return true;
        }
    }

    return false;
}

bool Person::hasRelative() const
{
    return m_relative != nullptr;
}

bool Person::isRelativeValid() const
{
    return hasRelative() && m_relative->isValid();
}

TrackingBody *Person::getHead() const
{

    return m_head;
}

TrackingBody *Person::getHand(size_t num) const
{

    if (m_hands.size() <= num)
        return NULL;

    return m_hands[num];
}

TrackingBody *Person::getRelative() const
{
    return m_relative;
}

const osg::Matrix &Person::getHeadMat() const
{

    if (!hasHead())
        return s_identity;

    return getHead()->getMat();
}

const osg::Matrix &Person::getHandMat(size_t num) const
{

    if (!hasHand(num))
        return s_identity;

    return m_hands[num]->getMat();
}

const osg::Matrix &Person::getRelativeMat() const
{
    if (!hasRelative())
        return s_identity;
    return m_relative->getMat();
}

unsigned int Person::getButtonState(size_t num) const
{
    if (!m_buttondev)
        return 0;

    return m_buttondev->getButtonState();
}

unsigned int Person::getRelativeButtonState(size_t num) const
{
    if (!m_relativebuttondev)
        return 0;

    return m_relativebuttondev->getButtonState();
}

double Person::getValuatorValue(size_t idx) const
{

    if (idx >= m_valuators.size())
        return 0.;

    return m_valuators[idx]->getValue();
}

float Person::eyeDistance() const {

    return m_eyeDistance;
}

void Person::setEyeDistance(float dist)
{
    m_eyeDistance = dist;
}

bool Person::activateOnAction() const
{
    return m_activateOnAction;
}

void Person::setActivateOnAction(bool enable)
{
    m_activateOnAction = enable;
}

}
