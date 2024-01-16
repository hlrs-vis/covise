/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_INPUT_GADGET_H
#define COVER_INPUT_GADGET_H

#include <osg/Matrix>
#include <vector>

#include "trackingbody.h"
#include "buttondevice.h"
#include "valuator.h"
#include <util/coExport.h>

namespace opencover
{

class COVEREXPORT Gadget
{
    friend class Input;

public:
    std::string name() const;

    TrackingBody *getBody(size_t idx) const;
    Valuator *getValuator(size_t idx) const;
    ButtonDevice *getButtons(size_t idx) const;

    bool isBodyValid(size_t idx) const;
    bool isValuatorValid(size_t idx) const;
    bool isButtonsValid(size_t idx) const;

    const osg::Matrix &getBodyMat(size_t idx) const;
    double getValuatorValue(size_t idx) const;
    unsigned int getButtonState(size_t idx) const;

private:
    Gadget(const std::string &name);
    void addBody(TrackingBody *body);
    void addButtons(ButtonDevice *);
    void addValuator(Valuator *val);

    std::string m_name;
    std::vector<TrackingBody *> m_body;
    std::vector<Valuator *> m_valuator;
    std::vector<ButtonDevice *> m_button;

    static const osg::Matrix s_identity;
};

} // namespace opencover
#endif
