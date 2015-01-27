/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Abstract base class, parametric surface                    **
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

#include "HfT_osg_Parametric_Surface.h"
#include "HfT_osg_Sphere.h"
#include "HfT_osg_MobiusStrip.h"
#include <cover/coVRPluginSupport.h>
#include <stdexcept>
using namespace osg;
using namespace std;
using namespace covise;
using namespace opencover;

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface()
//---------------------------------------------------
HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface()
    : m_patchesU(20)
    , m_patchesV(20)
    , m_creatingMode(5)
{
    initializeMembers();
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(int iMode, double iLowU,
//                                                                double iUpU, double iLowV,
//                                                                double iUpV)
//---------------------------------------------------
HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(int iMode, double iLowU,
                                                       double iUpU, double iLowV,
                                                       double iUpV)
    : m_patchesU(20)
    , m_patchesV(20)
{
    //test if the mode is in the right range
    //test id the parameter is a number
    if ((iMode >= 0) && (iMode <= 5) && (!isNaN((float)iMode)))
    {
        m_creatingMode = iMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. \n");
    }

    //test if the lower u bound is less than the upper u bound
    //test if the parameters are numbers
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number. \n");
    }

    //test if the lower v bound is less than the upper v bound
    //test if the parameters are numbers
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number. \n");
    }
    initializeMembers();
}
//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(double iLowU, double iUpU,
//                                                                double iLowV,double iUpV)
//---------------------------------------------------
HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(double iLowU, double iUpU,
                                                       double iLowV, double iUpV)
    : m_patchesU(20)
    , m_patchesV(20)
    , m_creatingMode(5)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }

    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    initializeMembers();
}
//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(int iPatU, double iPatV,
//                                                                double iLowU, double iUpU,
//                                                                double iLowV, double iUpV,
//                                                                int iMode)
//---------------------------------------------------
HfT_osg_Parametric_Surface::HfT_osg_Parametric_Surface(int iPatU, int iPatV,
                                                       double iLowU, double iUpU,
                                                       double iLowV, double iUpV,
                                                       int iMode)
{
    if ((iPatU > 1) && (!isNaN((float)iPatU)))
    {
        m_patchesU = iPatU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iPatV > 1) && (!isNaN((float)iPatV)))
    {
        m_patchesV = iPatV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iMode >= 0) && (iMode <= 5) && (!isNaN((float)iMode)))
    {
        m_creatingMode = iMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. \n");
    }

    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }

    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    initializeMembers();
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::~HfT_osg_Parametric_Surface()
//---------------------------------------------------
HfT_osg_Parametric_Surface::~HfT_osg_Parametric_Surface()
{
    m_patchesU = 0;
    m_patchesV = 0;
    m_creatingMode = 0;
    m_lowerBoundU = 0.0;
    m_upperBoundU = 0.0;
    m_lowerBoundV = 0.0;
    m_upperBoundV = 0.0;
    m_isSet = false;
    m_areQuadsComputed = false;
    m_isFirstSet = false;
    m_isTransparent = false;

    //Delete all pointers
    if (m_pColorDirectU != NULL)
    {
        //Cleanup the pointer to the directrix u color
        //Calls destructor of the vec4
        delete m_pColorDirectU;
    }

    if (m_pColorDirectV != NULL)
    {
        //Cleanup the pointer to the directrix v color
        //Calls destructor of the vec4
        delete m_pColorDirectV;
    }

    if (m_pColorEquat != NULL)
    {
        //Cleanup the pointer to the equator color
        //Calls destructor of the vec4
        delete m_pColorEquat;
    }

    if (m_pColorSurface != NULL)
    {
        //Cleanup the pointer to the surface color
        //Calls destructor of the vec4
        delete m_pColorSurface;
    }

    //Delete from subtree
    m_rpTrafoNode->getParent(0)->removeChild(m_rpTrafoNode.get());
    m_rpTrafoNode->removeChild(m_rpGeode.get());
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::initializeMembers()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::initializeMembers()
{
    m_epsilon = 0.000001;
    m_isSet = false;
    m_areQuadsComputed = false;
    m_isFirstSet = false;
    m_isTransparent = false;
    m_rpTrafoNode = new MatrixTransform();
    m_rpGeode = new Geode();

    m_rpGeom = new Geometry();
    m_rpDirectrixUGeom = new Geometry();
    m_rpDirectrixVGeom = new Geometry();
    m_rpEquatorGeom = new Geometry();
    m_rpStateSet = m_rpGeom->getOrCreateStateSet();
    m_rpStateSetDirectrixU = m_rpDirectrixUGeom->getOrCreateStateSet();
    m_rpStateSetDirectrixV = m_rpDirectrixVGeom->getOrCreateStateSet();
    m_rpStateSetEquator = m_rpEquatorGeom->getOrCreateStateSet();
    m_rpSupportingPoints = new Vec3Array;
    m_rpNormals = new Vec3Array;
    m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
    m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
    m_rpDirectrixUEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    m_rpDirectrixVEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    m_rpEquatorEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    m_rpPointCloud = new DrawArrays(PrimitiveSet::POINTS);
    m_rpPolyMode = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    m_rpTexture = new Texture2D();
    m_rpTexCoords = new Vec2Array;
    //Matrials for the colors of the objects
    m_rpMaterialSurface = new Material();
    m_rpMaterialDirectU = new Material();
    m_rpMaterialDirectV = new Material();
    m_rpMaterialEquat = new Material();
    //red color
    m_pColorDirectU = new Vec4(1.0, 0.0, 0.0, 1.0);
    //yellow color
    m_pColorDirectV = new Vec4(1.0, 1.0, 0.0, 1.0);
    //blue color
    m_pColorEquat = new Vec4(0.0, 0.0, 1.0, 1.0);
    //thickness of a line is 5.0f
    m_rpLineWidth = new LineWidth(5.0f);
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::PatchesU()
//---------------------------------------------------
int HfT_osg_Parametric_Surface::PatchesU() const
{
    return m_patchesU;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::PatchesV()
//---------------------------------------------------
int HfT_osg_Parametric_Surface::PatchesV() const
{
    return m_patchesV;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::LowerBoundU()
//---------------------------------------------------
double HfT_osg_Parametric_Surface::LowerBoundU() const
{
    return m_lowerBoundU;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::UpperBoundU()
//---------------------------------------------------
double HfT_osg_Parametric_Surface::UpperBoundU() const
{
    return m_upperBoundU;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::LowerBoundV()
//---------------------------------------------------
double HfT_osg_Parametric_Surface::LowerBoundV() const
{
    return m_lowerBoundV;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::UpperBoundV()
//---------------------------------------------------
double HfT_osg_Parametric_Surface::UpperBoundV() const
{
    return m_upperBoundV;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::Mode()
//---------------------------------------------------
int HfT_osg_Parametric_Surface::Mode() const
{
    return m_creatingMode;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::SurfaceColor()
//---------------------------------------------------
Vec4 *HfT_osg_Parametric_Surface::SurfaceColor() const
{
    return m_pColorSurface;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::SupportingPoints()
//---------------------------------------------------
Vec3Array *HfT_osg_Parametric_Surface::SupportingPoints() const
{
    return m_rpSupportingPoints;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::Normals()
//---------------------------------------------------
Vec3Array *HfT_osg_Parametric_Surface::Normals() const
{
    return m_rpNormals;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setBoundries(const double& iLowU, const double& iUpU,
//                                                   const double& iLowV, const double& iUpV)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setBoundries(const double &iLowU, const double &iUpU,
                                              const double &iLowV, const double &iUpV)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    //recalculation of the surface with its new boundries
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setPatches(int iPatchesU, int iPatchesV)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setPatches(int iPatchesU, int iPatchesV)
{
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    //recalculation of the surface with its new patch values
    recomputeSurface('P');
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setBoundriesAndPatches(const double& iLowU, const double& iUpU,
//                                                             const double& iLowV, const double& iUpV,
//                                                             int iPatchesU, int iPatchesV)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setBoundriesAndPatches(const double &iLowU, const double &iUpU,
                                                        const double &iLowV, const double &iUpV,
                                                        int iPatchesU, int iPatchesV)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    //recalculation of the surface with its new boundries and patch values
    recomputeSurface('P');
}
//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setPatchesAndMode(int iPatchesU, int iPatchesV,
//                                                        int iNewMode)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setPatchesAndMode(int iPatchesU, int iPatchesV,
                                                   int iNewMode)
{
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        this->m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many U patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        this->m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many V patches. Choose a number greater than 1. \n");
    }
    //Storage of the old mode for the later comparison
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 5) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. ");
    }
    //recalculation of the surface with its new patch values and its new mode
    recomputeSurface(oldMode, 'P');
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setBoundriesAndMode(const double& iLowU, const double& iUpU,
//                                                          const double& iLowV, const double& iUpV,
//                                                          int iNewMode)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setBoundriesAndMode(const double &iLowU, const double &iUpU,
                                                     const double &iLowV, const double &iUpV,
                                                     int iNewMode)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 5) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. ");
    }
    //Patches are the same so the quads have not to be recomputed.
    //Setting the m_areQuadsComputed variable true if the new mode also
    //needs the quad mesh.
    if (oldMode == 2 || oldMode == 3 || oldMode == 4 || oldMode == 5)
    {
        if (m_creatingMode == 2 || m_creatingMode == 3 || m_creatingMode == 4 || m_creatingMode == 5)
        {
            m_areQuadsComputed = true;
        }
    }
    recomputeSurface(oldMode, 'B');
    m_areQuadsComputed = false;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setBoundriesPatchesAndMode(const double& iLowU, const double& iUpU,
//                                                                const double& iLowV, const double& iUpV,
//                                                                int iPatchesU, int iPatchesV,
//                                                                int iNewMode)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setBoundriesPatchesAndMode(const double &iLowU, const double &iUpU,
                                                            const double &iLowV, const double &iUpV,
                                                            int iPatchesU, int iPatchesV,
                                                            int iNewMode)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many U patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many V patches. Choose a number greater than 1. \n");
    }
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 5) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. ");
    }

    //Recalculation of the surface because all parameters have been changed
    recomputeSurface(oldMode, 'P');
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setMode( int iNewMode)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setMode(int iNewMode)
{
    int oldMode = m_creatingMode;

    if ((iNewMode >= 0) && (iNewMode <= 5) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. \n");
    }

    //Patches are the same so the quads have not to be recomputed.
    //Setting the m_areQuadsComputed variable true if the new mode also
    //needs the quad mesh.
    if (oldMode == 2 || oldMode == 3 || oldMode == 4 || oldMode == 5)
    {
        if (m_creatingMode == 2 || m_creatingMode == 3 || m_creatingMode == 4 || m_creatingMode == 5)
        {
            m_areQuadsComputed = true;
        }
    }
    recomputeSurface(oldMode, 'M');
    m_areQuadsComputed = false;
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setImage(std::string iImagePath)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setImage(std::string iImagePath)
{
    m_imagePath = iImagePath;
    m_rpImage = osgDB::readImageFile(m_imagePath);
    m_rpTexture->setImage(m_rpImage.get());
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::recomputeSurface(char iChar)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::recomputeSurface(char iChar)
{
    switch (iChar)
    {
    //Only boundries have been changed
    case 'B':
        m_rpSupportingPoints->clear();
        m_rpNormals->clear();
        m_rpTexCoords->clear();
        this->digitalize();
        break;
    //Boundries and patches or only patches have been changed
    case 'P':
        m_rpSupportingPoints->clear();
        m_rpNormals->clear();
        m_rpTexCoords->clear();
        this->digitalize();

        //mode is the same but edges have to be recalculated in a new DrawElementsUInt
        //and the primitive set has to be reseted
        switch (m_creatingMode)
        {
        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);
            break;
        //Triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:
        //Quad mesh with a texture on it
        case 5:
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            break;

        //Quad mesh with directrices in u and v direction
        case 3:
            m_rpGeom->removePrimitiveSet(0);
            m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            m_rpDirectrixUEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            m_rpDirectrixVEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            break;

        //Quad mesh with the equator
        case 4:
            m_rpGeom->removePrimitiveSet(0);
            m_rpEquatorGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            m_rpEquatorEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            break;

            m_isSet = true;
            createObjectInMode();
            m_isSet = false;
        }
        break;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::recomputeSurface(int iOldMode, char iChar)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::recomputeSurface(int iOldMode, char iChar)
{
    HfT_osg_Parametric_Surface *dynamicCastObject;

    //Recalculation of the supporting points, normals and texture coordinates
    if (iChar == 'B' || iChar == 'P')
    {
        recomputeSurface('B');
    }
    //Cleanup from the old presentation mode
    if (m_creatingMode != iOldMode)
    {
        switch (iOldMode)
        {

        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);
            m_rpPointCloud = new DrawArrays(PrimitiveSet::POINTS);
            break;

        //Triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:
            break;

        //Quad mesh with directrices in u and v direction
        //Set new DrawElementsUInt for another presentation step.
        //Make the geometries invisible until they were needed again.
        //This works with a drawCallback which is member of a geometry.
        case 3:
            m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_rpDirectrixUEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            m_rpDirectrixVEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            //Set geometries invisible
            m_rpDirectrixUGeom->setDrawCallback(new Drawable::DrawCallback());
            m_rpDirectrixVGeom->setDrawCallback(new Drawable::DrawCallback());
            m_rpDirectrixUGeom->dirtyDisplayList();
            m_rpDirectrixVGeom->dirtyDisplayList();
            this->setCallbacks(false);
            break;

        //Quad mesh with the equator
        case 4:
            m_rpEquatorGeom->removePrimitiveSet(0);
            m_rpEquatorEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            //Set geometries invisible
            m_rpEquatorGeom->setDrawCallback(new Drawable::DrawCallback());
            m_rpEquatorGeom->dirtyDisplayList();
            break;

        //Quad mesh with a texture on it
        case 5:
            m_rpPolyMode->setMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            //Set texture off
            m_rpStateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OFF);
            m_rpMaterialSurface->setColorMode(Material::OFF);
            if (m_isTransparent)
            {
                m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.0);
            }
            this->setCallbacks(false);
            //If the surface is a mobius strip set the texture of the second geode off
            dynamicCastObject = dynamic_cast<HfT_osg_MobiusStrip *>(this);
            if (dynamicCastObject != NULL)
            {
                ((HfT_osg_MobiusStrip *)this)->m_rpStateSetFrontFaceGeode->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OFF);
            }
            break;
        }
        //if during the old presentation step the quad mesh was needed
        // and for next presenation step the quad mesh is not needed
        if ((iOldMode == 2 || iOldMode == 3 || iOldMode == 4 || iOldMode == 5) && !m_areQuadsComputed)
        {
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
        }
        createObjectInMode();
    }
    else
    {
        switch (m_creatingMode)
        {
        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);
        //triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:
        //Quad mesh with a texture on it
        case 5:
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            break;

        //Quad mesh with directrices in u and v direction
        case 3:
            m_rpGeom->removePrimitiveSet(0);
            m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            m_rpDirectrixUEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            m_rpDirectrixVEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            break;

        //Quad mesh with the equator
        case 4:
            m_rpGeom->removePrimitiveSet(0);
            m_rpEquatorGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            m_rpEquatorEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            break;
        }
        m_isSet = true;
        createObjectInMode();
        m_isSet = false;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::createObjectInMode()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::createObjectInMode()
{
    HfT_osg_Parametric_Surface *dynamicCastObject;

    if (!m_isFirstSet)
    {
        this->setAuxiliarGeometrys();
        this->setCallbacks(false);
    }

    switch (m_creatingMode)
    {
    //pointcloud
    case 0:
        m_rpPointCloud->setFirst(0);
        m_rpPointCloud->setCount(m_rpSupportingPoints->size());
        m_rpGeom->addPrimitiveSet(m_rpPointCloud);
        m_rpGeom->setNormalBinding(Geometry::BIND_OFF);
        break;

    //triangles
    case 1:
        computeTriangleEdges();
        m_rpGeom->addPrimitiveSet(m_rpTriangEdges);

        if (!m_isSet)
        {
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        }
        m_rpGeom->setNormalBinding(Geometry::BIND_OFF);
        break;

    //quads
    case 2:
    //quads with directrices
    case 3:
    //quads with equator
    case 4:
    //filled polygon mode
    case 5:
        if (m_areQuadsComputed == false)
        {
            computeQuadEdges();
            m_rpGeom->addPrimitiveSet(m_rpQuadEdges);
        }
        if (!m_isSet)
        {
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        }
        break;
    }
    switch (m_creatingMode)
    {
    // Compute the directrices, add them to a geometry and add this to a geode
    // Set for each one a different colored material to differ between them.
    // Identification of the current Parametric Surface object with a dynamic cast.
    // If the object is a sphere, the directrix in u direction is shown at
    // the equator of the sphere.
    case 3:
        dynamicCastObject = dynamic_cast<HfT_osg_Sphere *>(this);
        if (dynamicCastObject == NULL)
        {
            this->computeDirectrixU(0);
        }
        else
        {
            this->computeDirectrixU((int)floor((float)(((this->m_patchesV) + 1) / 2)));
        }
        this->computeDirectrixV(0);
        m_rpDirectrixUGeom->addPrimitiveSet(m_rpDirectrixUEdges);
        m_rpDirectrixVGeom->addPrimitiveSet(m_rpDirectrixVEdges);

        if (!m_isSet)
        {
            // Red u parameter line
            m_rpGeode->addDrawable(m_rpDirectrixUGeom);
            setStateSetAttributes(m_rpStateSetDirectrixU, m_rpMaterialDirectU);
            setColorAndMaterial(*m_pColorDirectU, *m_rpMaterialDirectU);

            //Yellow v parameter line
            m_rpGeode->addDrawable(m_rpDirectrixVGeom);
            setStateSetAttributes(m_rpStateSetDirectrixV, m_rpMaterialDirectV);
            setColorAndMaterial(*m_pColorDirectV, *m_rpMaterialDirectV);

            //if the object is a mobius strip, do not show the auxiliar geometry
            dynamicCastObject = dynamic_cast<HfT_osg_MobiusStrip *>(this);
            if (dynamicCastObject == NULL)
            {
                //display the created auxiliar geometry
                this->setCallbacks(true);
            }

            if (m_isFirstSet)
            {
                //fprintf(stderr, "Zum zweiten Mal besetzt \n");
                m_rpDirectrixUGeom->setDrawCallback(NULL);
                m_rpDirectrixVGeom->setDrawCallback(NULL);
            }
            m_isFirstSet = true;
            m_rpGeom->setNormalBinding(Geometry::BIND_OFF);
        }
        break;

    //Compute the equator add them to a geometry and add this to a geode
    //Set a colored material.
    case 4:
        this->computeEquator();
        m_rpEquatorGeom->addPrimitiveSet(m_rpEquatorEdges);

        if (!m_isSet)
        {
            // Blue equator line
            m_rpGeode->addDrawable(m_rpEquatorGeom);
            setStateSetAttributes(m_rpStateSetEquator, m_rpMaterialEquat);
            setColorAndMaterial(*m_pColorEquat, *m_rpMaterialEquat);
            m_rpGeom->setNormalBinding(Geometry::BIND_OFF);

            if (m_isFirstSet)
            {
                //fprintf(stderr, "Zum zweiten Mal besetzt \n");
                m_rpEquatorGeom->setDrawCallback(NULL);
            }
            m_isFirstSet = true;
        }
        break;

    case 5:
        if (!m_isSet)
        {
            m_rpPolyMode->setMode(PolygonMode::FRONT_AND_BACK, PolygonMode::FILL);
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            if (m_rpImage.get())
            {
                m_rpTexture->setUnRefImageDataAfterApply(true);
                m_rpStateSet->setAttributeAndModes(m_rpMaterialSurface, osg::StateAttribute::ON);
                m_rpStateSet->setTextureAttributeAndModes(0, m_rpTexture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                m_rpStateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }
            dynamicCastObject = dynamic_cast<HfT_osg_MobiusStrip *>(this);
            if (dynamicCastObject != NULL)
            {
                ((HfT_osg_MobiusStrip *)this)->m_rpStateSetFrontFaceGeode->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::ON);
            }

            //falls mobius band (dynamic cast) texture attribute auch auf on setzen
            if (m_isTransparent)
            {
                m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.8);
            }
            m_rpGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);

            //display the created auxiliar geometry
            this->setCallbacks(true);

            dynamicCastObject = dynamic_cast<HfT_osg_Sphere *>(this);
            if (dynamicCastObject != NULL)
            {
                ((HfT_osg_Sphere *)this)->setCallbacks(false);
            }

            m_isFirstSet = true;
        }
        break;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::createSubtree()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::createSubtree()
{
    m_rpGeode->addDrawable(m_rpGeom);
    m_rpGeode->setNodeMask(m_rpGeode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_rpTrafoNode->addChild(m_rpGeode);
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::createSurface()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::createSurface()
{
    createSubtree();
    this->digitalize();
    m_rpGeom->setVertexArray(m_rpSupportingPoints);
    m_rpGeom->setNormalArray(m_rpNormals);
    m_rpGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);
    m_rpGeom->setTexCoordArray(0, m_rpTexCoords);

    //Geometry for the directrices in u direction
    m_rpDirectrixUGeom->setVertexArray(m_rpSupportingPoints);
    m_rpDirectrixUGeom->setNormalArray(m_rpNormals);
    m_rpDirectrixUGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);

    //Geometry for the directrices in v direction
    m_rpDirectrixVGeom->setVertexArray(m_rpSupportingPoints);
    m_rpDirectrixVGeom->setNormalArray(m_rpNormals);
    m_rpDirectrixVGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);

    //Geometry for the equator
    m_rpEquatorGeom->setVertexArray(m_rpSupportingPoints);
    m_rpEquatorGeom->setNormalArray(m_rpNormals);
    m_rpEquatorGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);
    createObjectInMode();
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::computeTriangleEdges()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::computeTriangleEdges()
{
    //Number of supporting points
    int numSupportingPointsU = m_patchesU + 1;
    int i = 0;

    //Don´t triagulate the last row because there exists no upper row
    //The beginning is at bottom left side, clockwise
    while ((i + numSupportingPointsU) < (m_rpSupportingPoints->size()))
    {
        if (((i + 1) % numSupportingPointsU) != 0)
        {
            //counterclockwise
            m_rpTriangEdges->push_back(i);
            m_rpTriangEdges->push_back(i + numSupportingPointsU);
            m_rpTriangEdges->push_back(i + numSupportingPointsU + 1);

            m_rpTriangEdges->push_back(i + numSupportingPointsU + 1);
            m_rpTriangEdges->push_back(i + 1);
            m_rpTriangEdges->push_back(i);
        }
        i++;
    }
    //Test loop
    //for(int i = 0; i < m_rpTriangEdges->size();i++){
    //   fprintf(stderr,"Edge: %i %i \n", i, m_rpTriangEdges->at(i));
    //}
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::computeQuadEdges()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::computeQuadEdges()
{
    //Number of supporting points
    int numSupportingPointsU = m_patchesU + 1;
    int i = 0;

    //Don´t triagulate the last row because there exists no upper row
    //The beginning is at bottom left side, clockwise
    while ((i + numSupportingPointsU) < (m_rpSupportingPoints->size()))
    {
        if (((i + 1) % numSupportingPointsU) != 0)
        {
            //counterclockwise
            m_rpQuadEdges->push_back(i);
            m_rpQuadEdges->push_back(i + numSupportingPointsU);
            m_rpQuadEdges->push_back(i + numSupportingPointsU + 1);
            m_rpQuadEdges->push_back(i + 1);
        }
        i++;
    }
    //   for(int i = 0; i < m_rpQuadEdges -> size(); i++){
    //      fprintf(stderr, "Verbindungspunkt %d \n",m_rpQuadEdges ->at(i));
    //   }
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setStateSetAttributes(StateSet *ioStateSet, Material *ioMaterial)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setStateSetAttributes(StateSet *ioStateSet, Material *ioMaterial)
{
    ioStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ioStateSet->setAttributeAndModes(ioMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
    ioStateSet->setAttributeAndModes(m_rpLineWidth, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setColorAndMaterial(Vec4 &ioColor, Material &ioMat)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setColorAndMaterial(Vec4 &ioColor, Material &ioMat)
{
    ioMat.setColorMode(Material::AMBIENT_AND_DIFFUSE);
    ioMat.setDiffuse(Material::FRONT_AND_BACK, ioColor);
    ioMat.setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    ioMat.setAmbient(Material::FRONT_AND_BACK, Vec4(ioColor.x() * 0.3,
                                                    ioColor.y() * 0.3,
                                                    ioColor.z() * 0.3,
                                                    ioColor.w()));
    ioMat.setShininess(Material::FRONT_AND_BACK, 100.0);
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::setSurface(Vec4 *iColorArray)
//---------------------------------------------------
void HfT_osg_Parametric_Surface::setSurface(Vec4 iColorArray)
{
    m_pColorSurface = new Vec4(iColorArray.x(),
                               iColorArray.y(),
                               iColorArray.z(),
                               iColorArray.w());
    this->setColorAndMaterial(*m_pColorSurface, *m_rpMaterialSurface);
    this->setImageAndTexture();
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::computeDirectrixNumber(char iUV, double iSliderValue)
//---------------------------------------------------
int HfT_osg_Parametric_Surface::computeDirectrixNumber(char iUV, double iSliderValue)
{
    double lowerBoundSurface;
    double upperBoundSurface;
    //definition interval of the parameter
    double domain;
    int patches;
    double newValue;
    double directrixNumber;

    //Translation of the interval so that the lower bound begins with 0.0
    switch (iUV)
    {
    case 'U':
        if (fabs(this->m_lowerBoundU) <= 0.00001)
        {
            lowerBoundSurface = 0.0;
            upperBoundSurface = this->m_upperBoundU;
        }
        else
        {
            lowerBoundSurface = 0.0;
            upperBoundSurface = (this->m_upperBoundU) - (this->m_lowerBoundU);
        }
        patches = this->m_patchesU;
        //Offset about the lower bound of the surface
        //Is necessary because the interval is also offsetted
        newValue = iSliderValue - this->m_lowerBoundU;
        break;

    case 'V':
        if (fabs(this->m_lowerBoundV) <= 0.001)
        {
            lowerBoundSurface = 0.0;
            upperBoundSurface = this->m_upperBoundV;
        }
        else
        {
            lowerBoundSurface = 0.0;
            upperBoundSurface = (this->m_upperBoundV) - (this->m_lowerBoundV);
        }
        patches = this->m_patchesV;
        //Offset about the lower bound of the surface
        //Is necessary because the interval is also offsetted
        newValue = iSliderValue - (this->m_lowerBoundV);
        break;
    }
    //definition interval of the parameter
    domain = upperBoundSurface - lowerBoundSurface;

    //How often does the distance fit in the slider value?
    directrixNumber = newValue / (domain / patches);

    if (directrixNumber - floor(directrixNumber) < 0.5)
    {
        return (int)floor(directrixNumber);
    }
    else
    {
        return (int)ceil(directrixNumber);
    }
}

//---------------------------------------------------
//Implements HfT_osg_Parametric_Surface::surfacePreframe()
//---------------------------------------------------
void HfT_osg_Parametric_Surface::surfacePreframe()
{
}
