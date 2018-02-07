/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "trackingbody.h"
#include "input.h"
#include "inputdevice.h"
#include <config/CoviseConfig.h>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT
#include <osg/io_utils>

using namespace covise;

namespace opencover
{

TrackingBody::TrackingBody(const std::string &name)
    : InputSource(name, "Body")
    , m_baseBody(NULL)
{
    const std::string conf = config();

    m_assemble = false;
    m_6dof = false;
    for (int i=0; i<9; ++i)
    {
        std::string c = conf+".Assemble";
        switch (i) {
        case 0: c+="X"; break;
        case 1: c+="Y"; break;
        case 2: c+="Z"; break;
        case 3: c+="H"; break;
        case 4: c+="P"; break;
        case 5: c+="R"; break;
        case 6: c+="AxisX"; break;
        case 7: c+="AxisY"; break;
        case 8: c+="AxisZ"; break;
        }

        auto &v = m_valuator[i];
        const std::string dev = covise::coCoviseConfig::getEntry("device", c, "");
        if (!dev.empty())
        {
            v.device = Input::instance()->getDevice(dev);
        }
        if (!v.device)
            v.device = device();

        v.valuator = coCoviseConfig::getInt("valuator", c, -1);
        if (v.valuator >= 0)
        {
            m_assemble = true;
            if (i >= 6)
                m_assembleWithRotationAxis = true;
            if (v.valuator >= device()->numValuators())
                std::cerr << "TrackingBody: valuator index " << i << "=" << v.valuator << " out of range - " << v.device->numValuators() << " valuators" << std::endl;
        }
        v.scale = coCoviseConfig::getFloat("scale", c, 1.);
        v.shift = coCoviseConfig::getFloat("shift", c, 0.);
    }

    if (!m_assemble)
    {
        m_idx = coCoviseConfig::getInt("bodyIndex", conf, 0);
        if (m_idx >= device()->numBodies())
        {
            std::cerr << "TrackingBody: body index " << m_idx << " out of range - " << device()->numBodies() << " bodies" << std::endl;
        }
    }

    //Offset&Orientation reading and matrix creating

    float trans[3];
    float rot[3];

    trans[0] = coCoviseConfig::getFloat("x", conf + ".Offset", 0);
    trans[1] = coCoviseConfig::getFloat("y", conf + ".Offset", 0);
    trans[2] = coCoviseConfig::getFloat("z", conf + ".Offset", 0);

    rot[0] = coCoviseConfig::getFloat("h", conf + ".Orientation", 0);
    rot[1] = coCoviseConfig::getFloat("p", conf + ".Orientation", 0);
    rot[2] = coCoviseConfig::getFloat("r", conf + ".Orientation", 0);

    std::string baseBody = coCoviseConfig::getEntry("bodyOffset", conf);
    if (!baseBody.empty())
    {
        std::cout << "body " << name << " is based on " << baseBody << std::endl;
        m_baseBody = Input::instance()->getBody(baseBody);
        if (!m_baseBody)
        {
            std::cout << "did not find base body " << baseBody << " for body " << name << std::endl;
            std::cerr << "did not find base body " << baseBody << " for body " << name << std::endl;
        }
    }

#if 0
   std::cout<<name<<" is "<<device<<"; dev body no "<<m_idx
      <<" Offset=("<<trans[0]<<" "<<trans[1]<<" "<<trans[2]<<") "
      <<" Orientation=("<<rot[0]<<" "<<rot[1]<<" "<<rot[2]<<") "<<std::endl;
#endif

    //Create rotation matrix (from OpenVRUI/osg/mathUtils.h)
    MAKE_EULER_MAT(m_deviceOffsetMat, rot[0], rot[1], rot[2]);
    //fprintf(stderr, "offset from device('%d) %f %f %f\n", device_ID, deviceOffsets[device_ID].trans[0], deviceOffsets[device_ID].trans[1], deviceOffsets[device_ID].trans[2]);

    osg::Matrix translationMat;
    translationMat.makeTranslate(trans[0], trans[1], trans[2]);
    m_deviceOffsetMat.postMult(translationMat); //add translation
    m_mat = m_deviceOffsetMat;
    m_oldMat = m_mat;

    updateRelative();
}

/**
 * @brief TrackingBody::update Must be called from Input::update()
 * @return 0
 */
void TrackingBody::update()
{
    if (m_assemble)
    {
        m_varying = false;
        m_valid = true;
        m_mat.makeIdentity();

        double value[9];
        bool is6dof = false;
        for (int i=0; i<9; ++i)
        {
            const auto &v = m_valuator[i];

            m_varying |= v.device->isVarying();
            m_valid &= v.device->isValid();

            value[i] = 0.;
            int idx = v.valuator;
            if (idx >= 0)
            {
                auto val = v.device->getValuatorValue(idx);
                val += v.shift;
                val *= v.scale;
                value[i] = val;
            }
            if (idx >= 3)
                is6dof = true;
        }
        if (is6dof && m_valid)
            m_6dof = is6dof;

        if (m_assembleWithRotationAxis)
        {
            osg::Vec3 rotaxis(value[7], value[8], value[6]);
            m_mat.makeRotate(rotaxis.length()*0.01, rotaxis);
        }
        else
        {
            double hpr[3] = {value[3], value[4], value[5]};
            MAKE_EULER_MAT(m_mat, hpr[0], hpr[1], hpr[2]);
        }
        m_mat.setTrans(value[0], value[1], value[2]);
    }
    else
    {
        m_varying = device()->isVarying();

        m_valid = device()->isBodyMatrixValid(m_idx);
        m_6dof = device()->is6Dof();

        if (m_valid)
        {
            m_mat = device()->getBodyMatrix(m_idx);
        }
    }

    if (Input::debug(Input::Raw) && m_valid != m_oldValid)
    {
        std::cerr << "Input: raw " << name() << " valid=" << m_valid << std::endl;
    }
    m_oldValid = m_valid;

    if (m_valid)
    {
        if (Input::debug(Input::Raw) && Input::debug(Input::Matrices) && m_mat!=m_oldMat)
        {
            std::cerr << "Input: raw " << name() << " matrix=" << m_mat << std::endl;
        }
        m_oldMat = m_mat;

        //std::cerr << "TrackingBody::update: getting dev idx " << m_idx << ": " << m_mat << std::endl;
        m_mat.preMult(m_deviceOffsetMat);
    }
}

void TrackingBody::updateRelative()
{
    if (m_baseBody)
    {
        m_mat = m_mat * m_baseBody->getMat();
    }
}

const osg::Matrix &TrackingBody::getMat() const
{
    return m_mat;
}

const osg::Matrix &TrackingBody::getOffsetMat() const
{
    return m_deviceOffsetMat;
}

void TrackingBody::setOffsetMat(const osg::Matrix &m)
{
    m_deviceOffsetMat = m;
}

void TrackingBody::setMat(const osg::Matrix &mat)
{

    m_mat = mat;
}

bool TrackingBody::isVarying() const
{

    return m_varying;
}

bool TrackingBody::is6Dof() const
{

    return m_6dof;
}

void TrackingBody::setVarying(bool isVar)
{

    m_varying = isVar;
}

void TrackingBody::set6Dof(bool is6Dof)
{

    m_6dof = is6Dof;
}
}
