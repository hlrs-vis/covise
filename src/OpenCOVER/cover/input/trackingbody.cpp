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
    , m_valid(false)
    , m_oldValid(m_valid)
    , m_baseBody(NULL)
{
    const std::string conf = config();

    m_idx = coCoviseConfig::getInt("bodyIndex", conf, 0);
    if (m_idx >= device()->numBodies())
    {
        std::cerr << "TrackingBody: body index " << m_idx << " out of range - " << device()->numBodies() << " bodies" << std::endl;
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
    m_oldValid = m_valid;
    m_valid = device()->isBodyMatrixValid(m_idx);
    if (Input::debug(Input::Raw) && m_valid != m_oldValid)
    {
        std::cerr << "Input: raw " << name() << " valid=" << m_valid << std::endl;
    }
    if (m_valid)
    {
        m_mat = device()->getBodyMatrix(m_idx);
        if (Input::debug(Input::Raw) && Input::debug(Input::Matrices) && m_mat!=m_oldMat)
        {
            std::cerr << "Input: raw " << name() << " matrix=" << m_mat << std::endl;
        }
        m_oldMat = m_mat;
        //std::cerr << "TrackingBody::update: getting dev idx " << m_idx << ": " << m_mat << std::endl;
        m_mat.preMult(m_deviceOffsetMat);

    }
    m_varying = device()->isVarying();
    m_6dof = device()->is6Dof();
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

bool TrackingBody::isValid() const
{
    return m_valid;
}

bool TrackingBody::isVarying() const
{

    return m_varying;
}

bool TrackingBody::is6Dof() const
{

    return m_6dof;
}

void TrackingBody::setValid(bool valid)
{
    m_valid = valid;
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
