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
#include <util/coExport.h>
#include <iostream>
#include "inputsource.h"

namespace opencover
{

class InputDevice;

class COVEREXPORT TrackingBody: public InputSource
{
    friend class Input;

public:
    const osg::Matrix &getMat() const;
    const osg::Matrix &getOffsetMat() const;
    void setOffsetMat(const osg::Matrix &m);
    bool isVarying() const;
    bool is6Dof() const;

private:
    TrackingBody(const std::string &name);

    void update();
    void updateRelative();
    void setMat(const osg::Matrix &mat);
    void setVarying(bool isVar);
    void set6Dof(bool is6Dof);

    TrackingBody *m_baseBody = nullptr;
    size_t m_idx = 0;
    osg::Matrix m_mat, m_oldMat;
    osg::Matrix m_deviceOffsetMat;
    bool m_varying = true, m_6dof = false;

    struct Assemble
    {
        InputDevice *device = nullptr;
        int valuator = -1;
        double scale = 1.;
        double shift = 0.;
    };

    bool m_assemble = false;
    bool m_assembleWithRotationAxis = false; // assemble from 3 valuators desrcibing a rotation axis, angle is proportional to axis length
    Assemble m_valuator[9];
};
}
#endif /* TRACKINGBODY_H_ */
