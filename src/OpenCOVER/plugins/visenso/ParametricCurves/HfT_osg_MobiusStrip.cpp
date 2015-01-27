/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Class to generate a mobius strip                          **
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
/*  Parametric representation of a mobius strip
 *
 *   x(u,v) = r* cos(u)* (1+v*cos(u/2))
 *   y(u,v) = r* sin(u)* (1+v*cos(u/2))
 *   z(u,v) = r* v* sin(u/2)
 *
 *   u = [0, 2*PI] ; v = [-1.0, 1.0]
 */
#include "HfT_osg_MobiusStrip.h"
#include <osg/Vec3>
#include <osg/Array>
#include <stdexcept>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>

using namespace std;
using namespace covise;
using namespace opencover;

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::HfT_osg_MobiusStrip()
//---------------------------------------------------
HfT_osg_MobiusStrip::HfT_osg_MobiusStrip()
    : HfT_osg_Parametric_Surface(0.0, 2 * PI, -1.0, 1.0)
    , m_radian(1.0)
{
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
    setFrontFaceGeode();
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(const HfT_osg_MobiusStrip& iMobiusStrip)
//---------------------------------------------------
HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(const HfT_osg_MobiusStrip &iMobiusStrip)
    : HfT_osg_Parametric_Surface(iMobiusStrip.PatchesU(),
                                 iMobiusStrip.PatchesV(),
                                 iMobiusStrip.LowerBoundU(),
                                 iMobiusStrip.UpperBoundU(),
                                 iMobiusStrip.LowerBoundV(),
                                 iMobiusStrip.UpperBoundV(),
                                 iMobiusStrip.Mode())
    , m_radian(iMobiusStrip.Radian())
{
    setSurface(*iMobiusStrip.SurfaceColor());
    this->createAuxiliarObjects();
    setFrontFaceGeode();
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(double iRad)
//---------------------------------------------------
HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(double iRad)
    : HfT_osg_Parametric_Surface(0.0, 2 * PI, -1.0, 1.0)
{
    if ((iRad > 0.0) && (!isNaN(iRad)))
    {
        m_radian = iRad;
    }
    else
    {
        throw out_of_range("Invalid value for radian. Choose a number greater than 0.0. \n");
    }
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
    setFrontFaceGeode();
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(int iMode, double iLowU,
//                                                    double iUpU, double iLowV,
//                                                    double iUpV)
//---------------------------------------------------
HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(int iMode, double iLowU,
                                         double iUpU, double iLowV,
                                         double iUpV)
    : HfT_osg_Parametric_Surface(iMode, iLowU, iUpU, iLowV, iUpV)
{
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    this->createAuxiliarObjects();
    setFrontFaceGeode();
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(double iRad, int iPatchesU,
//                                                    int iPatchesV, double iLowU,
//                                                    double iUpU, double iLowV,
//                                                    double iUpV, int iMode)
//---------------------------------------------------
HfT_osg_MobiusStrip::HfT_osg_MobiusStrip(double iRad, int iPatchesU,
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
    setFrontFaceGeode();
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::~HfT_osg_MobiusStrip()
//---------------------------------------------------
HfT_osg_MobiusStrip::~HfT_osg_MobiusStrip()
{
    m_radian = 0.0;
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::Radian()
//---------------------------------------------------
double HfT_osg_MobiusStrip::Radian() const
{
    return m_radian;
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::setRadian(const double& iRadian)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setRadian(const double &iRadian)
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
//Implements HfT_osg_MobiusStrip::setRadianAndBoundries(const double& iRadian, const double& iLowU,
//                                                      const double& iUpU, const double& iLowV,
//                                                      const double& iUpV)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setRadianAndBoundries(const double &iRadian, const double &iLowU,
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
//Implements HfT_osg_MobiusStrip::setRadianBoundriesAndPatches(const double& iRadian, const double& iLowU,
//                                                             const double& iUpU, const double& iLowV,
//                                                             const double& iUpV, int iPatchesU,
//                                                             int iPatchesV)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setRadianBoundriesAndPatches(const double &iRadian, const double &iLowU,
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
//Implements HfT_osg_MobiusStrip::setRadianAndMode(const double& iRadian, int iNewMode)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setRadianAndMode(const double &iRadian, int iNewMode)
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
//Implements HfT_osg_MobiusStrip::setRadianBoundriesPatchesAndMode(const double& iRadian, const double& iLowU,
//                                                                 const double& iUpU, const double& iLowV,
//                                                                 const double& iUpV, int iPatchesU,
//                                                                 int iPatchesV, int iNewMode)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setRadianBoundriesPatchesAndMode(const double &iRadian, const double &iLowU,
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
//Implements HfT_osg_MobiusStrip::setImageAndTexture()
//---------------------------------------------------
void HfT_osg_MobiusStrip::setImageAndTexture()
{
    //Image and texture for the frontside
    const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("mobiusBack", "COVER.Plugin.ParametricCurves.Image", "/work/ac_te/Visenso2b.JPG");
    m_rpImage = osgDB::readImageFile(m_imagePath);
    m_rpTexture->setImage(m_rpImage.get());
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::setFrontFaceGeode()
//---------------------------------------------------
void HfT_osg_MobiusStrip::setFrontFaceGeode()
{
    // Set the frontFace geode as child to the trafo node.
    // Set the state sets, these are two different state sets.
    // Both state sets share the same geometry which stores the geometry
    // data for the mobius strip.
    m_rpFrontFaceGeode = new Geode();
    m_rpTrafoNode->addChild(m_rpFrontFaceGeode);
    m_rpFrontFaceGeode->addDrawable(m_rpGeom);
    m_rpFrontFaceGeode->setNodeMask(m_rpFrontFaceGeode->getNodeMask()
                                    & (~Isect::Intersection) & (~Isect::Pick));
    m_rpStateSetGeode = m_rpGeode->getOrCreateStateSet();
    m_rpStateSetFrontFaceGeode = m_rpFrontFaceGeode->getOrCreateStateSet();

    //Set cullFace to back for the main geode
    m_rpCullFaceGeode = new CullFace(CullFace::BACK);
    m_rpStateSetGeode->setAttributeAndModes(m_rpCullFaceGeode,
                                            osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    //Set the image for the frontFace geode
    m_frontFaceImagePath = (std::string)coCoviseConfig::getEntry("mobiusFront", "COVER.Plugin.ParametricCurves.Image", "/work/ac_te/Visenso1.JPG.JPG");
    m_rpFrontFaceImage = osgDB::readImageFile(m_frontFaceImagePath);

    if (m_rpFrontFaceImage)
    {
        m_rpFrontFaceTexture = new Texture2D(m_rpFrontFaceImage.get());
        m_rpCullFaceFrontFaceGeode = new CullFace(CullFace::FRONT);
        m_rpStateSetFrontFaceGeode->setAttributeAndModes(m_rpCullFaceFrontFaceGeode,
                                                         osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        m_rpStateSetFrontFaceGeode->setAttributeAndModes(m_rpMaterialSurface, osg::StateAttribute::ON);
        m_rpStateSetFrontFaceGeode->setTextureAttributeAndModes(0, m_rpFrontFaceTexture,
                                                                osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        m_rpStateSetFrontFaceGeode->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    }
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::setAuxiliarGeometrys()
//---------------------------------------------------
void HfT_osg_MobiusStrip::setAuxiliarGeometrys()
{
    m_rpGeode->addDrawable(m_rpSphereOriginDrawable);
    m_rpGeode->addDrawable(m_rpLineRadianDrawable);
    m_rpFrontFaceGeode->addDrawable(m_rpSphereOriginDrawable);
    m_rpFrontFaceGeode->addDrawable(m_rpLineRadianDrawable);
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::setCallbacks(bool iIsVisible)
//---------------------------------------------------
void HfT_osg_MobiusStrip::setCallbacks(bool iIsVisible)
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
//Implements HfT_osg_MobiusStrip::digitalize()
//---------------------------------------------------
void HfT_osg_MobiusStrip::digitalize()
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
            // parametric representation of a mobius strip
            // compute supporting points
            double x = m_radian * cos(paramU) * (1 + (paramV / 2) * cos(paramU / 2));
            double y = m_radian * sin(paramU) * (1 + (paramV / 2) * cos(paramU / 2));
            double z = m_radian * (paramV / 2) * sin(paramU / 2);
            m_rpSupportingPoints->push_back(Vec3(x, y, z));

            // Derivation with respect to u and derivation with respect to v
            // The cross product of both results in the normal vector
            // compute normal for each supporting point
            double xn = (m_radian * m_radian) * sin(paramU / 2) * cos(paramU) * (1 + paramV * cos(paramU / 2)) - 0.5 * (m_radian * m_radian) * paramV * sin(paramU);
            double yn = (m_radian * m_radian) * sin(paramU / 2) * sin(paramU) * (1 + paramV * cos(paramU / 2)) - 0.5 * (m_radian * m_radian) * paramV * cos(paramU);
            double zn = -(m_radian * m_radian * cos(paramU / 2)) * (1 + paramV * cos(paramU / 2));
            length = xn * xn + yn * yn + zn * zn;

            m_rpNormals->push_back(Vec3(-xn / sqrt(length), -yn / sqrt(length), -zn / sqrt(length)));

            // Texture coordinates are in a interval from 0 to 1
            // compute texture coordinate for each supporting point
            double ut = double(counterU) / m_patchesU;
            double vt = double(counterV) / m_patchesV;
            m_rpTexCoords->push_back(Vec2(ut, vt));

            //fprintf(stderr,"vecArray Mobius:%f %f %i %i \t %f %f %f \n",paramU,paramV,counterU,counterV,x,y,z);
            //fprintf(stderr,"TexCoords %i %i %i %i \t %f %f  \n",counterU,counterV,m_patchesU, m_patchesV, ut,vt);
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
//Implements HfT_osg_MobiusStrip::computeDirectrixU(int iSlideNumberU)
//---------------------------------------------------
void HfT_osg_MobiusStrip::computeDirectrixU(int iSlideNumberU)
{
    int numSuppPointsU = m_patchesU + 1;
    int position = numSuppPointsU * iSlideNumberU;

    //Line is created between the following points
    for (int i = 0; i < numSuppPointsU; i++)
    {
        m_rpDirectrixUEdges->push_back(i + position);
    }
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::computeDirectrixV(int iSlideNumberV)
//---------------------------------------------------
void HfT_osg_MobiusStrip::computeDirectrixV(int iSlideNumberV)
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
//Implements HfT_osg_MobiusStrip::computeEquator()
//---------------------------------------------------
bool HfT_osg_MobiusStrip::computeEquator()
{
    int numSuppPointsU = m_patchesU + 1;
    int numSuppPointsV = m_patchesV + 1;
    int position = (int)(floor((float)(numSuppPointsV / 2)) * numSuppPointsU);

    for (int i = 0; i < numSuppPointsU; i++)
    {
        m_rpEquatorEdges->push_back(position + i);
    }
    return true;
}

//---------------------------------------------------
//Implements HfT_osg_MobiusStrip::createAuxiliarObjects()
//---------------------------------------------------
void HfT_osg_MobiusStrip::createAuxiliarObjects()
{
    m_rpSphereOriginPoint = new Sphere(Vec3(0.0, 0.0, 0.0), 0.1);
    m_rpLineRadian = new Cylinder(m_rpSphereOriginPoint->getCenter() - Vec3d(m_radian, 0.0, 0.0) * 0.5,
                                  (m_rpSphereOriginPoint->getRadius()) / 5, (float)m_radian);
    m_rpLineRadian->setRotation(Quat(PI / 2, Vec3d(0.0, 1.0, 0.0)));

    m_rpSphereOriginDrawable = new ShapeDrawable(m_rpSphereOriginPoint);
    m_rpLineRadianDrawable = new ShapeDrawable(m_rpLineRadian);
}
