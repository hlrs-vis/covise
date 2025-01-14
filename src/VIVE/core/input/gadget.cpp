/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gadget.h"
#include "input.h"
#include "../vvConfig.h"
#include <config/CoviseConfig.h>
#include <sstream>

using namespace covise;

namespace vive
{

const vsg::dmat4 Gadget::s_identity = vsg::dmat4::identity();

Gadget::Gadget(const std::string &name): m_name(name)
{
    const std::string conf = "COVER.Input.Gadgets.Gadget:" + name;

    for (int i = 0;; ++i)
    {
        std::stringstream str;
        str << "body" << i;
        std::string bname = coCoviseConfig::getEntry(str.str(), conf, "");
        if (i == 0 && bname.empty())
            bname = coCoviseConfig::getEntry("body", conf, "");
        TrackingBody *body = Input::instance()->getBody(bname);
        if (!body)
        {
            break;
        }
        addBody(body);
    }

    for (int i = 0;; ++i)
    {
        std::stringstream str;
        str << "valuator" << i;
        std::string vname = coCoviseConfig::getEntry(str.str(), conf, "");
        if (i == 0 && vname.empty())
            vname = coCoviseConfig::getEntry("valuator", conf, "");
        Valuator *val = Input::instance()->getValuator(vname);
        if (!val)
            break;
        addValuator(val);
    }

    for (int i = 0;; ++i)
    {
        std::stringstream str;
        str << "buttons" << i;
        std::string bname = coCoviseConfig::getEntry(str.str(), conf, "");
        if (i == 0 && bname.empty())
            bname = coCoviseConfig::getEntry("buttons", conf, "");
        ButtonDevice *buttons = Input::instance()->getButtons(bname);
        if (!buttons)
            break;
        addButtons(buttons);
    }
}

std::string Gadget::name() const
{
    return m_name;
}

void Gadget::addBody(TrackingBody *hand)
{
    m_body.push_back(hand);
}

void Gadget::addValuator(Valuator *val)
{
    m_valuator.push_back(val);
}

void Gadget::addButtons(ButtonDevice *buttons)
{
    m_button.push_back(buttons);
}

TrackingBody *Gadget::getBody(size_t num) const
{
    if (m_body.size() <= num)
        return nullptr;

    return m_body[num];
}

bool Gadget::isBodyValid(size_t idx) const
{
    auto body = getBody(idx);
    return body && body->isValid();
}

const vsg::dmat4 &Gadget::getBodyMat(size_t num) const
{
    auto body = getBody(num);
    if (!body)
        return s_identity;

    return m_body[num]->getMat();
}

Valuator *Gadget::getValuator(size_t num) const
{
    if (m_valuator.size() <= num)
        return nullptr;

    return m_valuator[num];
}

bool Gadget::isValuatorValid(size_t idx) const
{
    auto val = getValuator(idx);
    return val && val->isValid();
}

double Gadget::getValuatorValue(size_t idx) const
{
    auto val = getValuator(idx);
    if (!val)
        return 0.;

    return m_valuator[idx]->getValue();
}

ButtonDevice *Gadget::getButtons(size_t num) const
{
    if (m_button.size() <= num)
        return nullptr;
    return m_button[num];
}

bool Gadget::isButtonsValid(size_t num) const
{
    auto buttons = getButtons(num);
    return buttons && buttons->isValid();
}

unsigned int Gadget::getButtonState(size_t num) const
{
    auto buttons = getButtons(num);
    if (!buttons)
        return 0;

    return m_button[num]->getButtonState();
}

} // namespace vive
