/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * person.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef __PERSON_H_
#define __PERSON_H_

#include <osg/Matrix>
#include <vector>

#include "trackingbody.h"
#include "buttondevice.h"
#include "valuator.h"
#include <util/coExport.h>

namespace opencover
{

class COVEREXPORT Person
{
    friend class Input;

public:
    std::string name() const;

    bool hasHead() const;
    bool isHeadValid() const;
    bool hasHand(size_t num) const;
    bool isHandValid(size_t idx) const;
    bool isVarying() const;
    bool hasRelative() const;
    bool isRelativeValid() const;

    TrackingBody *getHead() const;
    TrackingBody *getHand(size_t num) const;
    TrackingBody *getRelative() const;

    const osg::Matrix &getHeadMat() const;
    const osg::Matrix &getHandMat(size_t num) const;
    const osg::Matrix &getRelativeMat() const;

    unsigned int getButtonState(size_t num) const;
    unsigned int getRelativeButtonState(size_t num) const;
    double getValuatorValue(size_t idx) const;
    
    float eyeDistance() const;
    void setEyeDistance(float dist);

    bool activateOnAction() const;
    void setActivateOnAction(bool enable);

private:
    Person(const std::string &name);
    void addHand(TrackingBody *hand);
    void addValuator(Valuator *val);

    std::string m_name;
    TrackingBody *m_head = nullptr;
    TrackingBody *m_relative = nullptr;
    std::vector<TrackingBody *> m_hands;
    ButtonDevice *m_buttondev = nullptr;
    ButtonDevice *m_relativebuttondev = nullptr;
    std::vector<Valuator *> m_valuators;
    float m_eyeDistance = 0.f;
    bool m_activateOnAction = false;

    static const osg::Matrix s_identity;
};
}
#endif /* PERSON_H_ */
