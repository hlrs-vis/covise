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

    TrackingBody *getHead() const;
    TrackingBody *getHand(size_t num) const;

    const osg::Matrix &getHeadMat() const;
    const osg::Matrix &getHandMat(size_t num) const;

    unsigned int getButtonState(size_t num) const;
    double getValuatorValue(size_t idx) const;
    
    const std::string &getName() const{return m_name;};

private:
    Person(const std::string &name);
    void addHand(TrackingBody *hand);
    void addValuator(Valuator *val);

    std::string m_name;
    TrackingBody *m_head;
    std::vector<TrackingBody *> m_hands;
    ButtonDevice *m_buttondev;
    std::vector<Valuator *> m_valuators;

    static const osg::Matrix s_identity;
};
}
#endif /* PERSON_H_ */
