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

PointCloudInteractor::PointCloudInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority, PointCloudPlugin *p)
    : coTrackerButtonInteraction(type, name, priority)
    , m_selectedWithBox(false)
{
	plugin = p;
    fprintf(stderr, "\nPointCloudInteractor\n");
    selectedPointsGroup = new osg::Group();
    previewPointsGroup = new osg::Group();
	axisGroup = new osg::Group();
    cover->getObjectsRoot()->addChild(selectedPointsGroup.get());
    cover->getObjectsRoot()->addChild(previewPointsGroup.get());
	cover->getObjectsRoot()->addChild(axisGroup.get());
}


PointCloudInteractor::~PointCloudInteractor()
{
destroy();
}

bool PointCloudInteractor::destroy()
{
    cover->getObjectsRoot()->removeChild(selectedPointsGroup.get());
    cover->getObjectsRoot()->removeChild(previewPointsGroup.get());
    return true;
}

void
PointCloudInteractor::startInteraction()
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nPointCloudInteractor::startMove\n");
    }
	actionsuccess = true;
	currFI = FileInfo();
	for (auto &i :plugin->files)
	{
		i.prevMat = i.tranformMat->getMatrix();
		moveMat.makeIdentity();
		startHandMat.makeIdentity();
		traMat.makeIdentity();
		rotMat.makeIdentity();
		pointToMove = Vec3();
		firstPt = Vec3();
		centerPoint = Vec3();
		radius = 0;
		SaveMat = true;
		if (i.filename == fileToMove)
		{
			currFI = i;
			currFI.prevMat = currFI.tranformMat->getMatrix();
			// higlight currFI needs to be implemented
			//highlightActiveCloud();
		}
	}

    // store hand mat
    const Matrix &initHandMat = cover->getPointerMat();
    // get postion and direction of pointer
    m_initHandPos = initHandMat.preMult(Vec3(0.0, 0.0, 0.0));
    m_initHandDirection = initHandMat.preMult(Vec3(0.0, 1.0, 0.0));
}

void
PointCloudInteractor::doInteraction()
{
	if (cover->debugLevel(3))
	{
		fprintf(stderr, "\nPointCloudInteractor::stopMove\n");
	}
	if (type != ButtonC)
	{
		pointSelection bestPoint;
		bool hitPointSuccess = hitPoint(bestPoint);
		if (hitPointSuccess)
		{
			highlightPoint(bestPoint, true);
			if (m_rotpts && selectedPoints.size() == 1 && previewPoints.size() == 1)
			{
				axisStart = selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].coordinates;
				Vec3 axisEnd = previewPoints[0].file->pointSet[previewPoints[0].pointSetIndex].points[previewPoints[0].pointIndex].coordinates;
				rotAxis = axisEnd - axisStart;
				Vec3 startPoint = axisStart - rotAxis / rotAxis.length();
				Vec3 endPoint = axisEnd + rotAxis / rotAxis.length();
				showAxis(startPoint, endPoint);
			}
		}
	}
}

void
PointCloudInteractor::stopInteraction()
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nPointCloudInteractor::stopMove\n");
    }
    while (previewPointsGroup->getNumChildren() > 0)
    {
        previewPointsGroup->removeChild(0,1);
    }
    previewPoints.clear();
    if (!m_deselection)
    {
		actionsuccess = true;
		if (type == ButtonC)
		{	
			if (selectedPoints.size() != 0)
			{
				selectedPointsGroup->removeChild(selectedPointsGroup->getNumChildren() - 1);
				selectedPoints.pop_back();
			}
			actionsuccess = true;
			if (m_rotation)
			{
				axisGroup->removeChild(0, 1);
				rotAxis = Vec3();
				m_rotation = false;
			}
		}
		else
		{
			pointSelection bestPoint;
			bool hitPointSuccess = hitPoint(bestPoint);
			if (m_freemove)
			{
				MovePoints(moveMat);
				hitPointSuccess = false;
			}
			if ( m_translation)
			{
				MovePoints(traMat);
				hitPointSuccess = false;
			}
			if (m_rotation)
			{
				MovePoints(moveMat);
				rotMat.makeIdentity();
				traMat.makeIdentity();
				hitPointSuccess = false;
			}
			if (m_rotaxis && rotAxis.length() != 0)
			{
				Vec3 endPoint = axisStart + rotAxis * (5 / rotAxis.length());
				showAxis(axisStart, endPoint);
				m_rotation = true;
				hitPointSuccess = false;
			}
			if (hitPointSuccess)
			{
				highlightPoint(bestPoint);
				if (m_rotpts && selectedPoints.size() == 2)
				{
					m_rotation = true;
				}
			}
		}
    }
    else
    {
        deselectPoint();
    }
}

bool PointCloudInteractor::hitPoint(pointSelection& bestPoint)
{
    bool hitPointSuccess = false;
    if (plugin->files.size() != 0)
    {
        Matrix currHandMat = cover->getPointerMat();
        const Matrix &invBase = cover->getInvBaseMat();

        currHandMat = currHandMat * invBase;
        Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
        Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
		osg::MatrixTransform *MoTra = new osg::MatrixTransform();
        Vec3 currHandDirection = currHandEnd - currHandBegin;
		if (fileToMove != "")
		{
			if (currFI.filename == "")
			{
				for (auto &i : plugin->files)
				{
					if (i.filename == fileToMove)
					{
						currFI = i;
						currFI.prevMat = currFI.tranformMat->getMatrix();
						currFI.fileButton->setState(true);
						// higlight currFI needs to be implemented
					}
				}
			}
			if (SaveMat)
			{
				startHandMat.invert(currHandMat);
				SaveMat = false;
			}
			moveMat = startHandMat * currHandMat;
			if (m_rotaxis && !m_rotation)
			{
				rotAxis = currHandDirection;
				axisStart = currHandBegin;
			}
		}
		if (m_freemove && currFI.filename != "")
		{
			currFI.tranformMat->setMatrix(currFI.prevMat * moveMat);
		}
		else if (m_translation && (firstPt.length() != 0 || pointToMove.length() != 0))
		{
			if (pointToMove.length() == 0)
			{
				pointToMove = firstPt;
				previewPointsGroup->removeChild(0, 1);
				previewPoints.clear();
			}
			Vec3 newVec;
			newVec = moveMat.preMult(pointToMove);
			traMat.makeTranslate(newVec - pointToMove);
			currFI.tranformMat->setMatrix(currFI.prevMat * traMat);
		}
		else if (m_rotation && (firstPt.length() != 0 || pointToMove.length() != 0))
		{
			if (pointToMove.length() == 0)
			{
				pointToMove = firstPt;
				centerPoint = axisStart + (((rotAxis / rotAxis.length()) * ((pointToMove - axisStart)* rotAxis / rotAxis.length())));
				radius = (pointToMove - centerPoint).length();
				startHandMat.invert(currHandMat);
			}
			moveMat = startHandMat * currHandMat;
			moveMat.orthoNormalize(moveMat);
			Vec3 newVec = moveMat.preMult(pointToMove);
			Vec3 centerPoint2 = axisStart;
			if (rotAxis * (newVec - axisStart) != 0)
			{
				centerPoint2 = axisStart + (((rotAxis / rotAxis.length()) * ((newVec - axisStart)* rotAxis / rotAxis.length())));
			}
			newVec = (newVec - centerPoint2) * (radius / (newVec - centerPoint2).length()) + centerPoint;
			rotMat.makeRotate(pointToMove - centerPoint, newVec - centerPoint);
			traMat.makeTranslate(-centerPoint);
			moveMat = traMat * rotMat * traMat.inverse(traMat);
			currFI.tranformMat->setMatrix(currFI.prevMat * moveMat);
		}			
		else
		{
			double smallestDistance = FLT_MAX;
			//double distToHand = FLT_MAX;
			if (fileToMove != "")
			{
				if (currFI.pointSet)
				{
					//std::string fileName = currFIfilename;
					//fprintf(stderr, "Testing Set %s \n", fileName.c_str());
					for (int i = 0; i < currFI.pointSetSize; i++)
					{
						Vec3 center = Vec3((currFI.pointSet[i].xmin + currFI.pointSet[i].xmax) / 2, (currFI.pointSet[i].ymin + currFI.pointSet[i].ymax) / 2, (currFI.pointSet[i].zmin + currFI.pointSet[i].zmax) / 2);
						Vec3 corner = Vec3(currFI.pointSet[i].xmin, currFI.pointSet[i].ymin, currFI.pointSet[i].zmin);
						Vec3 radiusVec = center - corner;
						float radius = 1.5 * radiusVec.length();
						if (hitPointSetBoundingSphere(currHandDirection, currHandBegin, center, radius))
						{
							for (int j = 0; j < currFI.pointSet[i].size; j++)
							{
								Vec3 currentPoint = currFI.pointSet[i].points[j].coordinates;
								double distance = LinePointMeasure(currentPoint, currHandBegin, currHandDirection);
									if (distance < smallestDistance)
									{
										smallestDistance = distance;
										bestPoint.pointSetIndex = i;
										bestPoint.pointIndex = j;
										bestPoint.file = &currFI;
										hitPointSuccess = true;
										bestPoint.selectionIndex = selectionSetIndex;
										bestPoint.isBoundaryPoint = getSelectionIsBoundary();
									}
								//}
							}
						}
					}
				}
			}
			else
			{
				for (auto &fit : plugin->files)
				{
					if (fit.pointSet)
					{
						//std::string fileName = fit->filename;
						//fprintf(stderr, "Testing Set %s \n", fileName.c_str());
						for (int i = 0; i < fit.pointSetSize; i++)
						{
							Vec3 center = Vec3((fit.pointSet[i].xmin + fit.pointSet[i].xmax) / 2, (fit.pointSet[i].ymin + fit.pointSet[i].ymax) / 2, (fit.pointSet[i].zmin + fit.pointSet[i].zmax) / 2);
							Vec3 corner = Vec3(fit.pointSet[i].xmin, fit.pointSet[i].ymin, fit.pointSet[i].zmin);
							Vec3 radiusVec = center - corner;
							float radius = 1.5 * radiusVec.length();
							if (hitPointSetBoundingSphere(currHandDirection, currHandBegin, center, radius))
							{
								for (int j = 0; j < fit.pointSet[i].size; j++)
								{
									Vec3 currentPoint = fit.pointSet[i].points[j].coordinates;
									double distance = LinePointMeasure(currentPoint, currHandBegin, currHandDirection);
									if (distance < smallestDistance)
									{
										smallestDistance = distance;
										bestPoint.pointSetIndex = i;
										bestPoint.pointIndex = j;
										bestPoint.file = &fit;
										hitPointSuccess = true;
										bestPoint.selectionIndex = selectionSetIndex;
										bestPoint.isBoundaryPoint = getSelectionIsBoundary();
									}
								}
							}
						}
					}
				}
			}
			if (hitPointSuccess)
			{
				//dont select the same point twice
				for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter != selectedPoints.end(); iter++)
				{
					if (iter->pointSetIndex == bestPoint.pointSetIndex && iter->pointIndex == bestPoint.pointIndex)
					{
						hitPointSuccess = false;
					}
				}
			}
			if (m_freemove || m_translation || m_rotaxis || m_rotpts || m_rotation)
			{
				firstPt = bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].coordinates;
				fileToMove = bestPoint.file->filename;
				if (m_freemove || m_translation || m_rotation)
				{
					hitPointSuccess = false;
				}
				else
				{
					hitPointSuccess = true;
				}
			}
		}
    }
    return hitPointSuccess;
}

void  
PointCloudInteractor::MovePoints(osg::Matrixd MoveMat)
{
	for (int i = 0; i < currFI.pointSet->size; i++)
	{
		currFI.pointSet->points[i].coordinates = MoveMat.preMult(currFI.pointSet->points[i].coordinates);
	}
}

void
PointCloudInteractor::highlightPoint(pointSelection& selectedPoint, bool preview)
{
	Vec3 newSelectedPoint = selectedPoint.file->pointSet[selectedPoint.pointSetIndex].points[selectedPoint.pointIndex].coordinates;
	//fprintf(stderr,"Selected point has ID %d", selectedPoint.file->pointSet[selectedPoint.pointSetIndex].IDs[selectedPoint.pointIndex]);

    osg::Matrix *sphereTransformationMatrix = new osg::Matrix;
    sphereTransformationMatrix->makeTranslate(newSelectedPoint);

    osg::MatrixTransform *sphereTransformation = new osg::MatrixTransform;
    sphereTransformation->setMatrix(*sphereTransformationMatrix);
	sphereTransformation->setName("Sphere Transformation");
    selectedPoint.transformationMatrix = sphereTransformation;

    osg::Geode *sphereGeode = new osg::Geode;
    sphereTransformation->addChild(sphereGeode);
    osg::Sphere *selectedSphere = new osg::Sphere(newSelectedPoint,1.0f);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::StateSet* stateSet = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    osg::ShapeDrawable *selectedSphereDrawable = new osg::ShapeDrawable(selectedSphere, hint);
    osg::Material *selMaterial = new osg::Material();
    sphereGeode->addDrawable(selectedSphereDrawable);

    if (preview)
    {
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        if (previewPointsGroup->getNumChildren() == 0)
        {
            previewPointsGroup->addChild(sphereTransformation);
        }
        else
        {
            previewPointsGroup->setChild(0,sphereTransformation);
        }
        previewPoints.clear();
        previewPoints.push_back(selectedPoint);
    }
    else
    {
        selectedPointsGroup->addChild(sphereTransformation);
        if (!selectedPoint.isBoundaryPoint)
        {
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.0, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.0, 0.0, 1.0f));
        }
        else
        {
            selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
            selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        }
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        //addSelectedPoint(newSelectedPoint);
        selectedPoints.push_back(selectedPoint);
        updateMessage(selectedPoints);
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

void PointCloudInteractor::updateMessage(vector<pointSelection> points)
{
    //send message to NurbsSurfacePlugin
    cover->sendMessage(NULL, "NurbsSurface", PluginMessageTypes::NurbsSurfacePointMsg, sizeof(points), &points);
    for (auto iter=points.begin(); iter !=points.end(); iter++)
    {
        if (iter->isBoundaryPoint)
        {
            fprintf(stderr, "PointCloudInteractor::updateMessage sending boundary point!\n");
        }
    }
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
    {
        return 0.;
    }
    
    double d = c * pMinusC.length();
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

    return (distance <= radius);
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

void PointCloudInteractor::resize()
{
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 1);
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();

    //controls the sphere size
    double scaleFactor = sphereSize * distObj / distWld;
    //scaleFactor = 1.1f;

    // scale all selected points
	if (selectedPoints.size() != 0)
	{
		for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter != selectedPoints.end(); iter++)
		{
			osg::Matrix sphereMatrix = iter->transformationMatrix->getMatrix();
			Vec3 translation = sphereMatrix.getTrans();
			sphereMatrix.makeScale(scaleFactor, scaleFactor, scaleFactor);
			sphereMatrix.setTrans(translation);
			iter->transformationMatrix->setMatrix(sphereMatrix);
		}
	}
    
    //scale the preview point(s)
    for (std::vector<pointSelection>::iterator iter = previewPoints.begin(); iter !=previewPoints.end(); iter++)
    {
        osg::Matrix sphereMatrix = iter->transformationMatrix->getMatrix();
        Vec3 translation = sphereMatrix.getTrans();
        sphereMatrix.makeScale(scaleFactor, scaleFactor, scaleFactor);
        sphereMatrix.setTrans(translation);
        iter->transformationMatrix->setMatrix(sphereMatrix);
    }
}

bool PointCloudInteractor::deselectPoint()
{
    bool hitPointSuccess = false;
    if (plugin->files.size() != 0)
    {
		if (type == ButtonC)
		{
			selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			selectedPoints.clear();
		}
		else
		{
			Matrix currHandMat = cover->getPointerMat();
			Matrix invBase = cover->getInvBaseMat();

			currHandMat = currHandMat * invBase;

			Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
			Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
			Vec3 currHandDirection = currHandEnd - currHandBegin;

			double smallestDistance = FLT_MAX;
			std::vector<pointSelection>::iterator deselectionPoint;

			for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter != selectedPoints.end(); iter++)
			{
				int i = iter->pointSetIndex;
				int j = iter->pointIndex;
				Vec3 currentPoint = iter->file->pointSet[i].points[j].coordinates;
				double distance = LinePointDistance(currentPoint, currHandBegin, currHandDirection);
				if (distance < smallestDistance)
				{
					smallestDistance = distance;
					deselectionPoint = iter;
					hitPointSuccess = true;
				}
			}
			if (hitPointSuccess)
			{
				selectedPointsGroup->removeChild(deselectionPoint->transformationMatrix);
				selectedPoints.erase(deselectionPoint);
			}
		}
    }
    updateMessage(selectedPoints);
	actionsuccess = true;
    return hitPointSuccess;
}

void PointCloudInteractor::setTranslation(bool translation)
{
	if (translation)
	{
		m_translation = true;
		SaveMat = true;
	}
	else
	{
		m_translation = false;
		SaveMat = false;
		if (selectedPoints.size() > 0)
		{
			selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			selectedPoints.clear();
		}
	}
}

void PointCloudInteractor::setRotPts(bool rotation)
{
	if (rotation)
	{
		m_rotpts = true;
		SaveMat = true;
	}
	else
	{
		m_rotpts = false;
		m_rotation = false;
		SaveMat = false;
		if (selectedPoints.size() > 0)
		{
			selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			selectedPoints.clear();
		}
		axisGroup->removeChild(0, 1);
		pointToMove = Vec3();
		axisStart = Vec3();
		rotAxis = Vec3();
	}
}

void PointCloudInteractor::setRotAxis(bool rotaxis)
{
	if (rotaxis)
	{
		m_rotaxis = true;
		SaveMat = true;
	}
	else
	{
		m_rotaxis = false;
		m_rotation = false;
		SaveMat = false;
		if (selectedPoints.size() > 0)
		{
			selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			selectedPoints.clear();
		}
		axisGroup->removeChild(0, 1);
		pointToMove = Vec3();
		axisStart = Vec3();
		rotAxis = Vec3();
	}
}

void PointCloudInteractor::setFreeMove(bool freemove)
{
	if (freemove)
	{
		m_freemove = true;
		SaveMat = true;
	}
	else
	{
		m_freemove = false;
		SaveMat = false;
		if (selectedPoints.size() > 0)
		{
			selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			selectedPoints.clear();
		}
	}
}

void PointCloudInteractor::setDeselection(bool deselection)
{
    if (deselection)
    {
        m_deselection = true;
    }
    else
    {
        m_deselection = false;
    }
}

void PointCloudInteractor::setSelectionSetIndex(int selectionSet)
{
    selectionSetIndex=selectionSet;
}

void PointCloudInteractor::setSelectionIsBoundary(bool selectionIsBoundary)
{
    m_selectionIsBoundary=selectionIsBoundary;
}

bool PointCloudInteractor::getSelectionIsBoundary()
{
    return m_selectionIsBoundary;
}

void PointCloudInteractor::getData(PointCloudInteractor *PCI)
{
	fileToMove = PCI->fileToMove;
	rotMat = PCI->rotMat;
	saveHandMat = PCI->saveHandMat;
	selectedPoints = PCI->selectedPoints;
	selectedPointsGroup = PCI->selectedPointsGroup;
	startHandMat = PCI->startHandMat;
	traMat = PCI->traMat;
	axisGroup = PCI->axisGroup;
	rotAxis = PCI->rotAxis;
	m_rotation = PCI->m_rotation;
}

void PointCloudInteractor::setFile(string filename)
{
	fileToMove = filename;
	selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
	selectedPoints.clear();
	actionsuccess = true;
}

osg::StateSet* PointCloudInteractor::highlightActiveCloud()
{
	osg::Material *selMaterial = new osg::Material();
	selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
	selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
	selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.4f, 0.4f, 0.4f, 1.0f));
	selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
	selMaterial->setColorMode(osg::Material::OFF);
	osg::StateSet *stateset = new osg::StateSet();
	stateset->setAttribute(selMaterial, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
	unsigned int mode = osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF;
	for (unsigned int ii = 0; ii < 4; ii++)
	{
		stateset->setTextureMode(ii, GL_TEXTURE_1D, mode);
		stateset->setTextureMode(ii, GL_TEXTURE_2D, mode);
		stateset->setTextureMode(ii, GL_TEXTURE_3D, mode);
		stateset->setTextureMode(ii, GL_TEXTURE_RECTANGLE, mode);
		stateset->setTextureMode(ii, GL_TEXTURE_CUBE_MAP, mode);
	}
	return stateset;
}

void PointCloudInteractor::showAxis(Vec3 startPoint, Vec3 endPoint)
{
	if (axisGroup->getNumChildren() != 0)
	{
		axisGroup->removeChildren(0, axisGroup->getNumChildren());
	}
	osg::Geometry *axisBeam = new osg::Geometry();
	osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;
	points->push_back(startPoint);
	points->push_back(endPoint);
	osg::LineWidth *linewidth = new osg::LineWidth();
	linewidth->setWidth(5);
	osg::Material *beamColor = new osg::Material();
	beamColor->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
	beamColor->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
	beamColor->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
	beamColor->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
	beamColor->setColorMode(osg::Material::OFF);
	axisBeam->setVertexArray(points.get());
	axisBeam->setColorBinding(osg::Geometry::BIND_OVERALL);
	axisBeam->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));
	axisBeam->getOrCreateStateSet()->setAttribute(beamColor, osg::StateAttribute::ON);
	axisBeam->getOrCreateStateSet()->setAttribute(linewidth, osg::StateAttribute::ON);
	axisBeam->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
	axisBeam->setName("Rotation Axis");
	axisGroup->addChild(axisBeam);
}