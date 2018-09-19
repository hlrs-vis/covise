/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * buttondevice.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef BUTTONDEVICE_H
#define BUTTONDEVICE_H

#include <iostream>
#include <map>
#include "inputsource.h"

namespace opencover
{

class InputDevice;

class COVEREXPORT ButtonDevice: public InputSource
{
    friend class Input;
    friend class coMousePointer;

public:
    unsigned getButtonState() const; ///call this to get button state

private:
    ButtonDevice(const std::string &name);

    void update();
    void setButtonState(unsigned int state, bool isRaw = false);

    typedef std::map<unsigned, unsigned> ButtonMap;
    ButtonMap buttonMap, multiButtonMap;
    void createButtonMap(const std::string &confbase);
    unsigned mapButton(unsigned raw) const;

    unsigned int m_raw, m_oldRaw;
    unsigned int m_btnstatus; ///Saved buttonstatus
};
}
#endif /* BUTTONDEVICE_H_ */
