/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "trackingbody.h"
#include "input.h"
#include "inputdevice.h"
#include <config/CoviseConfig.h>

#include <OpenVRUI/vsg/mathUtils.h> //for MAKE_EULER_MAT
#include <vsg/io/read.h>

using namespace covise;

namespace vive
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
        if (device() && m_idx >= device()->numBodies())
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

    //Create rotation matrix (from OpenVRUI/vsg/mathUtils.h)
    m_deviceOffsetMat = makeEulerMat(rot[0], rot[1], rot[2]);
    //fprintf(stderr, "offset from device('%d) %f %f %f\n", device_ID, deviceOffsets[device_ID].trans[0], deviceOffsets[device_ID].trans[1], deviceOffsets[device_ID].trans[2]);

    vsg::dmat4 translationMat = vsg::translate((double)trans[0], (double)trans[1], (double)trans[2]);
    m_deviceOffsetMat = m_deviceOffsetMat * translationMat;
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
        m_mat = vsg::dmat4();

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
            vsg::dvec3 rotaxis((double)value[7], (double)value[8], (double)value[6]);
            m_mat = rotate(length(rotaxis)*0.01, rotaxis);
        }
        else
        {
            double hpr[3] = {value[3], value[4], value[5]};
            m_mat = makeEulerMat(hpr[0], hpr[1], hpr[2]);
        }
        setTrans(m_mat,vsg::dvec3((double)value[0], (double)value[1], (double)value[2]));
    }
    else if (device())
    {
        m_varying = device()->isVarying();
	/*if(numDevices()>1)
	{
	    if(device(0)->isBodyMatrixValid(m_idx) && device(1)->isBodyMatrixValid(m_idx))
	    {
               m_mat = device(0)->getBodyMatrix(m_idx);
               vsg::dmat4 m_mat2 = device(1)->getBodyMatrix(m_idx);
	       vsg::vec3 v1,v2;
	       v1 = m_mat.getTrans();
	       v2 = m_mat2.getTrans();
	       vsg::vec3 vd = v1 -v2;
	       fprintf(stderr,"1: %f %f %f 2: %f %f %f 1-2: %f %f %f\n",v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],vd[0],vd[1],vd[2]);
	    }
	}*/
	if(m_lastDevice < numDevices() && device(m_lastDevice)->isBodyMatrixValid(m_idx))
	{
	    m_valid=true;
            m_mat = device(m_lastDevice)->getBodyMatrix(m_idx);
	}
	else
	{
	    for(int i=0;i<numDevices();i++)
	    {
                m_valid = device(i)->isBodyMatrixValid(m_idx);
                m_6dof = device(i)->is6Dof();

                if (m_valid)
                {
	            m_lastDevice = i;
                    m_mat = device(i)->getBodyMatrix(m_idx);
	            break;
                }
	    }
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
        m_mat = m_deviceOffsetMat*m_mat;
    }
}

void TrackingBody::updateRelative()
{
    if (m_baseBody)
    {
        m_mat = m_baseBody->getMat() * m_mat;
    }
}

const vsg::dmat4 &TrackingBody::getMat() const
{
    return m_mat;
}

const vsg::dmat4 &TrackingBody::getOffsetMat() const
{
    return m_deviceOffsetMat;
}

void TrackingBody::setOffsetMat(const vsg::dmat4 &m)
{
    m_deviceOffsetMat = m;
}

void TrackingBody::setMat(const vsg::dmat4 &mat)
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
