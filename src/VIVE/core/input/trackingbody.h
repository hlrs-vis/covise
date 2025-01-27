/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#pragma once


#include <vsg/maths/mat4.h>
#include <util/coExport.h>
#include <iostream>
#include "inputsource.h"

namespace vive
{

class InputDevice;

class VVCORE_EXPORT TrackingBody: public InputSource
{
    friend class Input;
    friend class vvMousePointer;

public:
    const vsg::dmat4 &getMat() const;
    const vsg::dmat4 &getOffsetMat() const;
    void setOffsetMat(const vsg::dmat4 &m);
    bool isVarying() const;
    bool is6Dof() const;

private:
    TrackingBody(const std::string &name);

    void update();
    void updateRelative();
    void setMat(const vsg::dmat4 &mat);
    void setVarying(bool isVar);
    void set6Dof(bool is6Dof);

    TrackingBody *m_baseBody = nullptr;
    size_t m_idx = 0;
    vsg::dmat4 m_mat, m_oldMat;
    vsg::dmat4 m_deviceOffsetMat;
    bool m_varying = true, m_6dof = false;

    struct Assemble
    {
        InputDevice *device = nullptr;
        int valuator = -1;
        double scale = 1.;
        double shift = 0.;
    };
    int m_lastDevice = 0;
    bool m_assemble = false;
    bool m_assembleWithRotationAxis = false; // assemble from 3 valuators desrcibing a rotation axis, angle is proportional to axis length
    Assemble m_valuator[9];
};
}
