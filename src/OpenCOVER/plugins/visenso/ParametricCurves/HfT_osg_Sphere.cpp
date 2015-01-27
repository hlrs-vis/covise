/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Class to generate a sphere                                **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** cpp file                                                               **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

/* Parametric representation of a sphere
 *
 * x = r*cos(u)*cos(v)
 * y = r*sin(u)*cos(v)
 * z = r*sin(v)
*  u = [-PI, PI] ; v = [-PI/2, PI/2]
 *
 */

#include "HfT_osg_Sphere.h"
#include <osg/Array>
#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Geometry>
#include <stdexcept>
#include <config/CoviseConfig.h>

using namespace osg;
using namespace std;
using namespace covise;

//---------------------------------------------------
//Implements HfT_osg_Sphere::HfT_osg_Sphere()
//---------------------------------------------------
HfT_osg_Sphere::HfT_osg_Sphere()
    : HfT_osg_Parametric_Surface(-PI, PI, -PI / 2, PI / 2)
    , m_radian(1.0)
{
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::HfT_osg_Sphere(const HfT_osg_Sphere& iSphere)
//---------------------------------------------------
HfT_osg_Sphere::HfT_osg_Sphere(const HfT_osg_Sphere &iSphere)
    : HfT_osg_Parametric_Surface(iSphere.PatchesU(),
                                 iSphere.PatchesV(),
                                 iSphere.LowerBoundU(),
                                 iSphere.UpperBoundU(),
                                 iSphere.LowerBoundV(),
                                 iSphere.UpperBoundV(),
                                 iSphere.Mode())
    , m_radian(iSphere.Radian())
{
    setSurface(*iSphere.SurfaceColor());
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::HfT_osg_Sphere(double iRad)
//---------------------------------------------------
HfT_osg_Sphere::HfT_osg_Sphere(double iRad)
    : HfT_osg_Parametric_Surface(-PI, PI, -PI / 2, PI / 2)
{
    if ((iRad > 0.0) && (!isNaN(iRad)))
    {
        m_radian = iRad;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::HfT_osg_Sphere(int iMode, double iLowU,
//                                          double iUpU, double iLowV,
//                                          double iUpV)
//---------------------------------------------------
HfT_osg_Sphere::HfT_osg_Sphere(int iMode, double iLowU,
                               double iUpU, double iLowV,
                               double iUpV)
    : HfT_osg_Parametric_Surface(iMode, iLowU, iUpU, iLowV, iUpV)
    , m_radian(1.0)
{
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::HfT_osg_Sphere(double iRad, int iPatchesU,
//                                          int iPatchesV, double iLowU,
//                                          double iUpU, double iLowV,
//                                          double iUpV, int iMode)
//---------------------------------------------------
HfT_osg_Sphere::HfT_osg_Sphere(double iRad, int iPatchesU,
                               int iPatchesV, double iLowU,
                               double iUpU, double iLowV,
                               double iUpV, int iMode)
    : HfT_osg_Parametric_Surface(iPatchesU, iPatchesV, iLowU, iUpU,
                                 iLowV, iUpV, iMode)
{
    if ((iRad > 0.0) && (!isNaN(iRad)))
    {
        m_radian = iRad;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::~HfT_osg_Sphere()
//---------------------------------------------------
HfT_osg_Sphere::~HfT_osg_Sphere()
{
    m_radian = 0.0;
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::Radian()
//---------------------------------------------------
double HfT_osg_Sphere::Radian() const
{
    return m_radian;
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setRadian(const double& iRadian)
//---------------------------------------------------
void HfT_osg_Sphere::setRadian(const double &iRadian)
{
    if ((iRadian > 0.0) && (!isNaN(iRadian)))
    {
        m_radian = iRadian;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    m_rpLineRadian->setHeight(float(m_radian));
    m_rpLineRadian->setCenter(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5);
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setRadianAndBoundries(const double& iRadian, const double& iLowU,
//                                                 const double& iUpU, const double& iLowV,
//                                                 const double& iUpV)
//---------------------------------------------------
void HfT_osg_Sphere::setRadianAndBoundries(const double &iRadian, const double &iLowU,
                                           const double &iUpU, const double &iLowV,
                                           const double &iUpV)
{
    if ((iRadian > 0.0) && (!isNaN(iRadian)))
    {
        m_radian = iRadian;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    m_rpLineRadian->setHeight(float(m_radian));
    m_rpLineRadian->setCenter(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5);
    setBoundries(iLowU, iUpU, iLowV, iUpV);
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setRadianBoundriesAndPatches(const double& iRadian, const double& iLowU,
//                                                       const double& iUpU, const double& iLowV,
//                                                       const double& iUpV, int iPatchesU,
//                                                       int iPatchesV)
//---------------------------------------------------
void HfT_osg_Sphere::setRadianBoundriesAndPatches(const double &iRadian, const double &iLowU,
                                                  const double &iUpU, const double &iLowV,
                                                  const double &iUpV, int iPatchesU,
                                                  int iPatchesV)
{
    if ((iRadian > 0.0) && (!isNaN(iRadian)))
    {
        m_radian = iRadian;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    m_rpLineRadian->setHeight(float(m_radian));
    m_rpLineRadian->setCenter(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5);
    setBoundriesAndPatches(iLowU, iUpU, iLowV, iUpV, iPatchesU, iPatchesV);
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setRadianAndMode(const double& iRadian, int iNewMode)
//---------------------------------------------------
void HfT_osg_Sphere::setRadianAndMode(const double &iRadian, int iNewMode)
{
    if ((iRadian > 0.0) && (!isNaN(iRadian)))
    {
        m_radian = iRadian;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    m_rpLineRadian->setHeight(float(m_radian));
    m_rpLineRadian->setCenter(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5);
    setBoundriesAndMode(m_lowerBoundU, m_upperBoundU,
                        m_lowerBoundV, m_upperBoundV, iNewMode);
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setRadianBoundriesPatchesAndMode(const double& iRadian, const double& iLowU,
//                                                            const double& iUpU, const double& iLowV,
//                                                            const double& iUpV, int iPatchesU,
//                                                            int iPatchesV, int iNewMode)
//---------------------------------------------------
void HfT_osg_Sphere::setRadianBoundriesPatchesAndMode(const double &iRadian, const double &iLowU,
                                                      const double &iUpU, const double &iLowV,
                                                      const double &iUpV, int iPatchesU,
                                                      int iPatchesV, int iNewMode)
{
    if ((iRadian > 0.0) && (!isNaN(iRadian)))
    {
        m_radian = iRadian;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a value greater than 0.0. \n");
    }
    m_rpLineRadian->setHeight(float(m_radian));
    m_rpLineRadian->setCenter(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5);
    setBoundriesPatchesAndMode(iLowU, iUpU, iLowV, iUpV, iPatchesU, iPatchesV, iNewMode);
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setImageAndTexture()
//---------------------------------------------------
void HfT_osg_Sphere::setImageAndTexture()
{
    const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("sphere", "COVER.Plugin.ParametricCurves.Image",
                                                                          "/work/ac_te/Weltkarte.jpg");
    m_rpImage = osgDB::readImageFile(m_imagePath);
    m_rpTexture->setImage(m_rpImage.get());
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setAuxiliarGeometrys()
//---------------------------------------------------
void HfT_osg_Sphere::setAuxiliarGeometrys()
{
    m_rpGeode->addDrawable(m_rpSphereOriginDrawable);
    m_rpGeode->addDrawable(m_rpLineRadianDrawable);
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::setCallbacks(bool iIsVisible)
//---------------------------------------------------
void HfT_osg_Sphere::setCallbacks(bool iIsVisible)
{
    if (iIsVisible)
    {
        m_rpSphereOriginDrawable->setDrawCallback(NULL);
        m_rpLineRadianDrawable->setDrawCallback(NULL);
    }
    else
    {
        m_rpSphereOriginDrawable->setDrawCallback(new Drawable::DrawCallback());
        m_rpLineRadianDrawable->setDrawCallback(new Drawable::DrawCallback());
        m_rpSphereOriginDrawable->dirtyDisplayList();
        m_rpLineRadianDrawable->dirtyDisplayList();
    }
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::digitalize()
//---------------------------------------------------
void HfT_osg_Sphere::digitalize()
{
    double paramU = 0.0;
    double paramV = 0.0;
    int counterU = 0;
    int counterV = 0;
    //Length of the vector
    double length;

    // Loop the latitude, then along the meridian
    // External loop with v, sweep direction from bottom to top
    paramV = m_lowerBoundV;

    while (paramV <= (m_upperBoundV + m_epsilon))
    {
        counterU = 0;
        paramU = m_lowerBoundU;

        // Internal loop with u, sweep direction to the right
        while (paramU <= (m_upperBoundU + m_epsilon))
        {
            // parametric representation of a sphere
            // compute supporting points
            double x = m_radian * cos(paramU) * cos(paramV);
            double y = m_radian * sin(paramU) * cos(paramV);
            double z = m_radian * sin(paramV);
            m_rpSupportingPoints->push_back(Vec3(x, y, z));

            // Derivation with respect to u and derivation with respect to v
            // The cross product of both results in the normal vector
            // compute normal for each supporting point
            double xn = (m_radian * m_radian) * cos(paramU) * (cos(paramV) * cos(paramV));
            double yn = (m_radian * m_radian) * sin(paramU) * (cos(paramV) * cos(paramV));
            double zn = (m_radian * m_radian) * sin(paramV) * cos(paramV);
            length = xn * xn + yn * yn + zn * zn;
            //Normalized normal vector
            m_rpNormals->push_back(Vec3(-xn / sqrt(length), -yn / sqrt(length), -zn / sqrt(length)));

            // Texture coordinates are in a interval from 0 to 1
            // compute texture coordinate for each supporting point
            double ut = double(counterU) / double(m_patchesU);
            double vt = double(counterV) / double(m_patchesV);
            m_rpTexCoords->push_back(Vec2(ut, vt));

            //fprintf(stderr,"ArrayList %f %f %i %i \t %f %f %f \n",
            //paramU,paramV,counterU,counterV,x,y,z);
            //fprintf(stderr,"TexCoords %i %i %i %i \t %f %f  \n",
            //counterU,counterV,m_patchesU, m_patchesV, ut,vt);
            counterU++;
            paramU = m_lowerBoundU + ((m_upperBoundU - m_lowerBoundU) / m_patchesU) * counterU;
        }
        if ((counterV + 1) > m_patchesV)
        {
            break;
        }
        else
        {
            counterV++;
            paramV = m_lowerBoundV + ((m_upperBoundV - m_lowerBoundV) / m_patchesV) * counterV;
        }
    }
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::computeDirectrixU(int iSlideNumberU)
//---------------------------------------------------
void HfT_osg_Sphere::computeDirectrixU(int iSlideNumberU)
{
    int numSuppPointsU = m_patchesU + 1;
    int numSuppPointsV = m_patchesV + 1;
    int beginningPosition;
    int position;

    //The whole sphere
    // V parameter interval has a length of PI
    if (fabs(PI - (m_upperBoundV - m_lowerBoundV)) <= m_epsilon)
    {
        beginningPosition = numSuppPointsU * iSlideNumberU;
        //Line is created between the following points
        for (int i = 0; i < numSuppPointsU; i++)
        {
            m_rpDirectrixUEdges->push_back(i + beginningPosition);
        }
    }
    //Not entire closed sphere
    else
    {

        //lower hemisphere -->directrix in u direction begins at the upper border
        if (m_upperBoundV < (PI / 2))
        {
            position = (numSuppPointsV - 1) * numSuppPointsU;
            beginningPosition = numSuppPointsU * (m_patchesV - iSlideNumberU);
            //Line is created between the following points
            for (int i = 0; i < numSuppPointsU; i++)
            {
                m_rpDirectrixUEdges->push_back(position - beginningPosition);
            }
        }

        //upper hemisphere -->directrix in u direction begins at the lower border
        else
        {
            beginningPosition = iSlideNumberU * numSuppPointsU;
            //Line is created between the following points
            for (int i = 0; i < numSuppPointsU; i++)
            {
                m_rpDirectrixUEdges->push_back(i + beginningPosition);
            }
        }
    }
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::computeDirectrixV(int iSlideNumberV)
//---------------------------------------------------
void HfT_osg_Sphere::computeDirectrixV(int iSlideNumberV)
{
    int numSuppPointsU = m_patchesU + 1;
    int numSuppPointsV = m_patchesV + 1;

    //Line is created between the following points
    for (int i = 0; i < numSuppPointsV; i++)
    {
        m_rpDirectrixVEdges->push_back(i * numSuppPointsU + iSlideNumberV);
    }
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::computeEquator()
//---------------------------------------------------
bool HfT_osg_Sphere::computeEquator()
{
    int numSuppPointsV = m_patchesV + 1;

    // Compute the equator only if this is a whole sphere
    if (fabs(PI - (m_upperBoundV - m_lowerBoundV)) <= m_epsilon)
    {
        m_rpDirectrixUEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
        this->computeDirectrixU((int)floor((float)(numSuppPointsV / 2)));
        m_rpEquatorEdges = m_rpDirectrixUEdges;
        return true;
    }
    else
    {
        return false;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Sphere::createAuxiliarObjects()
//---------------------------------------------------
void HfT_osg_Sphere::createAuxiliarObjects()
{
    m_rpSphereOriginPoint = new Sphere(Vec3(0.0, 0.0, 0.0), 0.1);
    m_rpLineRadian = new Cylinder(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5,
                                  (m_rpSphereOriginPoint->getRadius()) / 5, (float)m_radian);
    m_rpLineRadian->setRotation(Quat(PI / 2, Vec3d(0.0, 1.0, 0.0)));

    m_rpSphereOriginDrawable = new ShapeDrawable(m_rpSphereOriginPoint);
    m_rpLineRadianDrawable = new ShapeDrawable(m_rpLineRadian);
}
