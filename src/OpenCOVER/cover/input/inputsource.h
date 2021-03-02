/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputsource.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef INPUTSOURCE_H
#define INPUTSOURCE_H

#include <util/coExport.h>
#include <string>
#include <vector>

namespace opencover
{

class InputDevice;

class COVEREXPORT InputSource
{
public:
    InputSource(const std::string &name, const std::string &kind);
    virtual ~InputSource();
    bool isValid() const;
    int validDevice();
    virtual void update() = 0;
    const std::string &name() const;
    const std::string &config() const;
    InputDevice *device() const;
    InputDevice *device(int i) const;
    size_t numDevices() const {return m_dev.size();};

protected:
    void setValid(bool);
    bool m_valid = false, m_oldValid = false;
    int m_validDevice=0;

private:
    const std::string m_name;
    const std::string m_conf;
    std::vector<InputDevice *>m_dev;
};

}
#endif
