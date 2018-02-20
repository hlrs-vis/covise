/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/coInteractor.h>
#include <cover/coVRCommunication.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
#include <osg/Plane>

#include "PointCloud.h"
#include "PointCloudInteractor.h"

using namespace covise;
using namespace opencover;
using namespace osg;

PointCloudInteractor::PointCloudInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority = Medium)
    : coTrackerButtonInteraction(type, name, priority)
    , m_selectedWithBox(false)
{
    fprintf(stderr, "\nPointCloudInteractor\n");
    selectedPointsGeode = new osg::Geode();
    selectedPointsGeode->setName("selectedPoints");
    previewPointsGeode = new osg::Geode();
    previewPointsGeode->setName("previewPoints");
    
    cover->getObjectsRoot()->addChild(selectedPointsGeode.get());
    cover->getObjectsRoot()->addChild(previewPointsGeode.get());
}


PointCloudInteractor::~PointCloudInteractor()
{
destroy();
}

bool PointCloudInteractor::destroy()
{
    cover->getObjectsRoot()->removeChild(selectedPointsGeode.get());
    cover->getObjectsRoot()->removeChild(previewPointsGeode.get());
    return true;
}

void
PointCloudInteractor::startInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPointCloudInteractor::startMove\n");

    // store hand mat
    Matrix initHandMat = cover->getPointerMat();
    // get postion and direction of pointer
    m_initHandPos = initHandMat.preMult(Vec3(0.0, 0.0, 0.0));
    m_initHandDirection = initHandMat.preMult(Vec3(0.0, 1.0, 0.0));
}


void
PointCloudInteractor::stopInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPointCloudInteractor::stopMove\n");

    //if (coVRMSController::instance()->isMaster())
    //{
        while (previewPointsGeode->getNumDrawables() > 0)
            previewPointsGeode->removeDrawables(0);

        Vec3 bestPoint;
        bool hitPointSuccess = hitPoint(bestPoint);
        if (hitPointSuccess)
        {
            addSelectedPoint(bestPoint);
            highlightPoint(bestPoint);
        }
    //}
}


bool PointCloudInteractor::hitPoint(Vec3& bestPoint)
{
    bool hitPointSuccess = false;
    if (m_files)
    {
        Matrix currHandMat = cover->getPointerMat();
        Matrix invBase = cover->getInvBaseMat();

        currHandMat = currHandMat * invBase;

        Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
        Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
        Vec3 currHandDirection = currHandEnd - currHandBegin;

        double smallestDistance = FLT_MAX;
        for (std::list<fileInfo>::const_iterator fit = m_files->begin(); fit != m_files->end(); fit++)
        {
            if (fit->pointSet)
            {
                for (int i=0; i< fit->pointSetSize; i++)
                {
                    Vec3 center = Vec3( (fit->pointSet[i].xmin+fit->pointSet[i].xmax)/2, (fit->pointSet[i].ymin+fit->pointSet[i].ymax)/2, (fit->pointSet[i].zmin+fit->pointSet[i].zmax)/2);
                    Vec3 corner = Vec3( fit->pointSet[i].xmin, fit->pointSet[i].ymin, fit->pointSet[i].zmin);
                    Vec3 radiusVec = center-corner;
                    float radius = 1.5 * radiusVec.length();

                    //if (hitPointSet(currHandDirection, currHandBegin, &fit->pointSet[i]))
                    if (hitPointSetBoundingSphere(currHandDirection, currHandBegin, center, radius))
                    {
                        for (int j=0; j<fit->pointSet[i].size; j++)
                        {
                            Vec3 currentPoint = Vec3(fit->pointSet[i].points[j].x,fit->pointSet[i].points[j].y,fit->pointSet[i].points[j].z);
                            double distance = LinePointDistance(currentPoint, currHandBegin, currHandDirection);
                            //double distance = LinePointMeasure(currentPoint, currHandBegin, currHandDirection);
                            if (distance<smallestDistance)
                            {
                                smallestDistance=distance;
                                bestPoint=currentPoint;   
                                hitPointSuccess = true;
                            }                
                        }            
                    }
                }
            }
        }
        if (cover->debugLevel(3))
            fprintf(stderr, "\nPointCloudInteractor::foundPoint %f %f %f \n", bestPoint.x(), bestPoint.y(), bestPoint.z());
    }
    return hitPointSuccess;
}

void
PointCloudInteractor::doInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPointCloudInteractor::stopMove\n");

    Vec3 bestPoint;
    bool hitPointSuccess= hitPoint(bestPoint);
    if (hitPointSuccess)
    {
        highlightPoint(bestPoint,true);
    }
}

void
PointCloudInteractor::highlightPoint(Vec3 selectedPoint, bool preview)
{
    osg::Sphere *selectedSphere = new osg::Sphere(selectedPoint,0.01);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::StateSet* stateSet = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    osg::ShapeDrawable *selectedSphereDrawable = new osg::ShapeDrawable(selectedSphere, hint);
    osg::Material *selMaterial = new osg::Material();
    if (preview)
    {
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        if (previewPointsGeode->getNumDrawables() == 0)
            previewPointsGeode->addDrawable(selectedSphereDrawable);
        else
            previewPointsGeode->setDrawable(0,selectedSphereDrawable);
    }
    else
    {
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.0, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.0, 0.0, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);

        selectedPointsGeode->addDrawable(selectedSphereDrawable);
    }
    stateSet->setAttribute(selMaterial);
    selectedSphereDrawable->setStateSet(stateSet);
}


void
PointCloudInteractor::addSelectedPoint(Vec3 selectedPoint)
{
//send message to NurbsSurfacePlugin
cover->sendMessage(NULL, "NurbsSurface", PluginMessageTypes::NurbsSurfacePointMsg, sizeof(selectedPoint), &selectedPoint);
}

double
PointCloudInteractor::LinePointMeasure(Vec3 center, Vec3 handPos, Vec3 handDirection)
{
    handDirection.normalize();
    Vec3 pMinusC = handPos - center;
    double b = handDirection * pMinusC;
    double c = sqrt(pMinusC * pMinusC - b * b);
    //double d = pMinusC.length();
    if (pMinusC.length()==0.)
        return 0.;
    
    double d = c / pMinusC.length();
    return d;
}

double
PointCloudInteractor::LinePointDistance(Vec3 center, Vec3 handPos, Vec3 handDirection)
{
    handDirection.normalize();
    Vec3 pMinusC = handPos - center;
    double b = handDirection * pMinusC;
    double c = sqrt(pMinusC * pMinusC - b * b);
    //double d = pMinusC.length();
    return c;

}

bool PointCloudInteractor::hitPointSetBoundingSphere(osg::Vec3 handDir, osg::Vec3 handPos, Vec3 center, float radius)
{
    float distance = LinePointDistance(center,handPos, handDir);
    if (distance<=radius)
    {
        return true;
    }
    return false;
}

bool PointCloudInteractor::hitPointSet(osg::Vec3 handDir, osg::Vec3 handPos, PointSet *pointset)
{
    osg::Vec3 contact;
    handDir.normalize();
    // build Plane
    //osg::Plane plane = osg::Plane(0.,0.,1.,-pointset.xmin);
    osg::Vec3 coord[6]= { 
        Vec3(pointset->xmin,0.,0.),
        Vec3(pointset->xmax,0.,0.),
        Vec3(0.,pointset->ymin,0.),
        Vec3(0.,pointset->ymax,0.),
        Vec3(0.,0.,pointset->zmin), 
        Vec3(0.,0.,pointset->zmax)
    };

    osg::Vec3 normal[6] = {
        Vec3(1.,0.,0.),
        Vec3(1.,0.,0.),
        Vec3(0.,1.,0.),
        Vec3(0.,1.,0.),
        Vec3(0.,0.,1.),
        Vec3(0.,0.,1.)
    };

    for (int i=0; i<6;i++)
    {
        if ( abs(normal[i] * handDir ) >= 0.0001)
        {
            float d = normal[i] * coord[i];
            float t = (d - normal[i] * handPos)/(normal[i] * handDir);
            osg::Vec3 newRay = handDir * t;
            contact = handPos + newRay;

            if (((i==0 || i==1) && (contact.z()<pointset->zmax && contact.z()>pointset->zmin)
                        && (contact.y()<pointset->ymax && contact.y()>pointset->ymin)) ||
                    ((i==2 || i==3) && (contact.x()<pointset->xmax && contact.x()>pointset->xmin)
                     && (contact.z()<pointset->zmax && contact.z()>pointset->zmin)) ||
                    ((i==4 || i==5) && (contact.x()<pointset->xmax && contact.x()>pointset->xmin)
                     && (contact.y()<pointset->ymax && contact.y()>pointset->ymin)) )
            {
                fprintf(stderr, "\nhitPointSet !\n");
                return true;
            }
        }
    }
    return false;
}
