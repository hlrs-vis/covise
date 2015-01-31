/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * trackingbody.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef __TRACKINGBODY_H_
#define __TRACKINGBODY_H_

#include <osg/Matrix>
#include <iostream>

namespace opencover
{

class InputDevice;

class TrackingBody
{
    friend class Input;

public:
    bool isValid() const;
    const osg::Matrix &getMat() const;
    bool isVarying() const;
    bool is6Dof() const;

private:
    TrackingBody(const std::string &name);

    void update();
    void setValid(bool);
    void setMat(const osg::Matrix &mat);
    void setVarying(bool isVar);
    void set6Dof(bool is6Dof);

    InputDevice *m_dev;
    bool m_valid;
    size_t m_idx;
    osg::Matrix m_mat, m_deviceOffsetMat;
    bool m_varying, m_6dof;
};
}
#endif /* TRACKINGBODY_H_ */
