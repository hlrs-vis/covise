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
    selectedPointsGroup = new osg::Group();
    previewPointsGroup = new osg::Group();
	MovedPTSGroup = new osg::Group();
	cover->getObjectsRoot()->addChild(MovedPTSGroup.get());
    cover->getObjectsRoot()->addChild(selectedPointsGroup.get());
    cover->getObjectsRoot()->addChild(previewPointsGroup.get());
}


PointCloudInteractor::~PointCloudInteractor()
{
destroy();
}

bool PointCloudInteractor::destroy()
{
    cover->getObjectsRoot()->removeChild(selectedPointsGroup.get());
    cover->getObjectsRoot()->removeChild(previewPointsGroup.get());
	cover->getObjectsRoot()->removeChild(MovedPTSGroup.get());
    return true;
}

void
PointCloudInteractor::startInteraction()
{
	cout << "Start Interaction" << endl;
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

        while (previewPointsGroup->getNumChildren() > 0)
            previewPointsGroup->removeChild(0,1);

        previewPoints.clear();
        if (!m_deselection)
        {
            pointSelection bestPoint;
            bool hitPointSuccess = hitPoint(bestPoint);
            if (hitPointSuccess)
            {
                highlightPoint(bestPoint);
				if (m_translation || m_rotation)
				{
					MovedPTS.push_back(selectedPoints[selectedPoints.size() - 1]);
					int neededPts;
					if (m_translation)
					{
						neededPts = 2;
					}
					if (m_rotation)
					{
						neededPts = 3;
					}
					if (MovedPTS.size() == neededPts)
					{
						//Vec3 CoorBestPoint = Vec3(bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].x,
						//	bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].y,
						//	bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].z);
						std::vector<Vec3> CoordPts;
						for (std::vector<pointSelection>::iterator iter = MovedPTS.begin(); iter != MovedPTS.end(); iter++)
						{
							int i = iter->pointSetIndex;
							int j = iter->pointIndex;
							CoordPts.push_back(Vec3(iter->file->pointSet[i].points[j].x, iter->file->pointSet[i].points[j].y, iter->file->pointSet[i].points[j].z));
						}
						Vec3 CoorBestPoint = CoordPts[1] - CoordPts[0];
						string filename = bestPoint.file->filename;
						cout << "1 Mpts: " << MovedPTS.size() << endl;
						cout << "1 selPts: " << selectedPoints.size() << endl;
						CalcMoveVec(CoorBestPoint, filename);
						//MovedPTS.clear();
						MovedPTS.clear();
						selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
						selectedPoints.clear();
						cout << "Mpts: " << MovedPTS.size() << endl;
						cout << "selPts: " << selectedPoints.size() << endl;
						cout << "selectedPointsGroup: " << selectedPointsGroup->getNumChildren() << endl;
					}
					cout << "selectedPointsGroup: " << selectedPointsGroup->getNumChildren();
					//if (MovedPTS.size() == 2 || MovedPTS.size() == 3)
					//{
					//	MovedPTS.clear();
					//}
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
    if (m_files)
    {
        Matrix currHandMat = cover->getPointerMat();
        Matrix invBase = cover->getInvBaseMat();

        currHandMat = currHandMat * invBase;
        Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
        Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
        Vec3 currHandDirection = currHandEnd - currHandBegin;

        double smallestDistance = FLT_MAX;
        for (std::vector<FileInfo>::const_iterator fit = m_files->begin(); fit != m_files->end(); fit++)
        {
			if (fit->pointSet)
            {
                //std::string fileName = fit->filename;
                //fprintf(stderr, "Testing Set %s \n", fileName.c_str());
                for (int i=0; i< fit->pointSetSize; i++)
                {
                    Vec3 center = Vec3( (fit->pointSet[i].xmin+fit->pointSet[i].xmax)/2, (fit->pointSet[i].ymin+fit->pointSet[i].ymax)/2, (fit->pointSet[i].zmin+fit->pointSet[i].zmax)/2);
                    Vec3 corner = Vec3( fit->pointSet[i].xmin, fit->pointSet[i].ymin, fit->pointSet[i].zmin);
                    Vec3 radiusVec = center-corner;
                    float radius = 1.5 * radiusVec.length();

                    if (hitPointSetBoundingSphere(currHandDirection, currHandBegin, center, radius))
                    {
                        for (int j=0; j<fit->pointSet[i].size; j++)
                        {
                            Vec3 currentPoint = Vec3(fit->pointSet[i].points[j].x,fit->pointSet[i].points[j].y,fit->pointSet[i].points[j].z);
                            double distance = LinePointMeasure(currentPoint, currHandBegin, currHandDirection);
                            if (distance<smallestDistance)
                            {
                                smallestDistance=distance;
                                bestPoint.pointSetIndex = i;
                                bestPoint.pointIndex = j;
                                bestPoint.file = &(*fit);
                                hitPointSuccess = true;
                                bestPoint.selectionIndex = selectionSetIndex;
                                bestPoint.isBoundaryPoint = getSelectionIsBoundary();
                            }
                        }
                    }
                }
            }
        }
        if (hitPointSuccess)
        {
            //dont select the same point twice
            for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter !=selectedPoints.end(); iter++)
            {
                if (iter->pointSetIndex==bestPoint.pointSetIndex && iter->pointIndex==bestPoint.pointIndex)
                {
                    hitPointSuccess=false;
                }
            }
        }
    }
    return hitPointSuccess;
}

void PointCloudInteractor::doInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPointCloudInteractor::stopMove\n");
    pointSelection bestPoint;
    bool hitPointSuccess= hitPoint(bestPoint);
    if (hitPointSuccess)
    {
        highlightPoint(bestPoint,true);
    }
}

void PointCloudInteractor::CalcMoveVec(Vec3 SelPoint, string filename)
{
	PointSet *PTSet;
	PTSet = MovedPTS.begin()->file->pointSet;
	osg::Matrixf *Move = new osg::Matrixf();
	if (m_translation)
	{
		Move = MakeTranslate(SelPoint, Move, PTSet);
	}
	if (m_rotation)
	{
		Move = MakeRotate(SelPoint, Move, PTSet);
	}
	//Old but working
	//Move->makeTranslate(SelPoint);
	osg::MatrixTransform *MoTra = new osg::MatrixTransform();
	osg::Node *oldDrawing, *newDrawing, *nGroupNode = new osg::Node();
	for (unsigned int i = 0; i < cover->getObjectsRoot()->getNumChildren(); i++)
	{
		nGroupNode = cover->getObjectsRoot()->getChild(i);
		if (nGroupNode->getName() == filename)
		{
			cout << "neue geode zeichnen" << endl;
			oldDrawing = nGroupNode->asGroup()->getChild(0);
		}
	}
	MoTra->addChild(oldDrawing);
	MoTra->setMatrix(*Move);
	MoTra->setName("TraMa");
	newDrawing = MoTra;
	newDrawing->setName("Moved Geode");
	nGroupNode->asGroup()->replaceChild(oldDrawing, newDrawing);
	//Old but working
	//PTSet = selectedPoints.begin()->file->pointSet;
	//for (int i = 0; i < PTSet->size; i++)
	//{
	//	PTSet->points[i].x = PTSet->points[i].x + SelPoint.x();
	//	PTSet->points[i].y = PTSet->points[i].y + SelPoint.y();
	//	PTSet->points[i].z = PTSet->points[i].z + SelPoint.z();
	//}

	NodeInfo NI;
	FileInfo FI;
	FI = *MovedPTS.begin()->file;
	FI.nodes.clear();
	FI.pointSet = PTSet;
	PointCloudGeometry *drawable = new PointCloudGeometry(PTSet);
	drawable->changeLod(1.f);
	drawable->setPointSize(1.f);
	drawable->setName("Drawable Gedrehte Punkte");
	Geode *nGeode = new Geode();
	nGeode->addDrawable(drawable);
	nGeode->setName("Gedrehte Punkte");
	NI.node = nGeode;
	FI.nodes.push_back(NI);
	UpdatedFIVec.clear();
	UpdatedFIVec.push_back(FI);
	updatePoints(&UpdatedFIVec);
	if (MovedPTSGroup->getNumChildren() > 0)
	{
		MovedPTSGroup->replaceChild(MovedPTSGroup->getChild(0), nGeode);
	}

	/*
	altes Zeug läuft nicht
	FI = *selectedPoints.begin()->file;
	PTSet = selectedPoints.begin()->file->pointSet;
	std::vector<Vec3> CoordPts;
	for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter != selectedPoints.end(); iter++)
	{
		int i = iter->pointSetIndex;
		int j = iter->pointIndex;
		CoordPts.push_back(Vec3(iter->file->pointSet[i].points[j].x, iter->file->pointSet[i].points[j].y, iter->file->pointSet[i].points[j].z));
	}

	Vec3f VecAB, VecBC, MoveAreaVec;
	VecAB = CoordPts[1] - CoordPts[0];
	if (CoordPts.size() == 3)
	{
		cout << "rotation" << endl;
		VecBC = CoordPts[2] - CoordPts[1];
		MoveAreaVec = Vec3f(VecAB.operator^(VecBC));
		PointCloudInteractor::MakeRotate(MoveAreaVec);
	}
	if (m_translation)
	{
		cout << "translation" << endl;
		PointCloudInteractor::MakeTranslate(VecAB);
	}

	for (unsigned int i = 0; i < cover->getObjectsRoot()->getNumChildren(); i++)
	{
		nGroupNode = cover->getObjectsRoot()->getChild(i);
		if (cover->getObjectsRoot()->getChild(i)->getName() == filename)
		{
			cout << "neue geode zeichnen" << endl;
			oldGeode = nGroupNode->asGroup()->getChild(0);
			oldGeode->setName("Moved Geode!");
			MoveTransformation->addChild(oldGeode);
			MoveTransformation->setMatrix(*MoveMat);
			drawNode = MoveTransformation;
		}
	}
	nGroupNode->asGroup()->replaceChild(oldGeode, drawNode);
	NodeInfo NI;
	FI.nodes.clear();
	FI.pointSet = PTSet;
	PointCloudGeometry *drawable = new PointCloudGeometry(PTSet);
	//PointCloudGeometry *drawable = dynamic_cast<PointCloudGeometry*>(drawNode->asGeode()->getDrawable()->asGeometry());
	drawable->changeLod(1.f);
	drawable->setPointSize(4.f);
	drawable->setName("Drawable Gedrehte Punkte");
	Geode *nGeode = new Geode();
	nGeode->addDrawable(drawable);
	nGeode->setName("Gedrehte Punkte");
	//if (cover->getObjectsRoot()->getChild(cover->getObjectsRoot()->getNumChildren() - 1)->getName() == "Gedrehte Punkte")
	//{
	//	cover->getObjectsRoot()->removeChild(cover->getObjectsRoot()->getNumChildren() - 1);
	//}
	//cover->getObjectsRoot()->addChild(nGeode);
	NI.node = nGeode;
	FI.nodes.push_back(NI);
	UpdatedFIVec.clear();
	UpdatedFIVec.push_back(FI);
	updatePoints(&UpdatedFIVec);
	*/
}

osg::Matrixf*
PointCloudInteractor::MakeRotate(Vec3 PTVec, osg::Matrixf *RotMat, PointSet *PTSet)
{
	RotMat->makeRotate(PTVec, Vec3(PTVec));
	for (int i = 0; i < PTSet->size; i++)
	{
		Vec3f PtsCoord = Vec3f(PTSet->points[i].x, PTSet->points[i].y, PTSet->points[i].z);
		PtsCoord = RotMat->getRotate().operator*(PtsCoord);
		PTSet->points[i].x = PtsCoord.x();
		PTSet->points[i].y = PtsCoord.y();
		PTSet->points[i].z = PtsCoord.z();
	}
	return RotMat;
}

osg::Matrixf*
PointCloudInteractor::MakeTranslate(Vec3 PTVec, osg::Matrixf *TraMat, PointSet *PTSet)
{
	TraMat->makeTranslate(PTVec);
	for (int i = 0; i < PTSet->size; i++)
	{
		PTSet->points[i].x = PTSet->points[i].x + PTVec.x();
		PTSet->points[i].y = PTSet->points[i].y + PTVec.y();
		PTSet->points[i].z = PTSet->points[i].z + PTVec.z();
	}
	return TraMat;
}

void
PointCloudInteractor::highlightPoint(pointSelection& selectedPoint, bool preview)
{
	Vec3 newSelectedPoint = Vec3(selectedPoint.file->pointSet[selectedPoint.pointSetIndex].points[selectedPoint.pointIndex].x,
		selectedPoint.file->pointSet[selectedPoint.pointSetIndex].points[selectedPoint.pointIndex].y,
		selectedPoint.file->pointSet[selectedPoint.pointSetIndex].points[selectedPoint.pointIndex].z);
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
            previewPointsGroup->addChild(sphereTransformation);
        else
            previewPointsGroup->setChild(0,sphereTransformation);
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
            fprintf(stderr, "PointCloudInteractor::updateMessage sending boundary point!\n");
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
        return 0.;
    
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
    for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter !=selectedPoints.end(); iter++)
    {
        osg::Matrix sphereMatrix = iter->transformationMatrix->getMatrix();
        Vec3 translation = sphereMatrix.getTrans();
        sphereMatrix.makeScale(scaleFactor, scaleFactor, scaleFactor);
        sphereMatrix.setTrans(translation);
        iter->transformationMatrix->setMatrix(sphereMatrix);
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
    if (m_files)
    {
        Matrix currHandMat = cover->getPointerMat();
        Matrix invBase = cover->getInvBaseMat();
		
        currHandMat = currHandMat * invBase;

        Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
        Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
        Vec3 currHandDirection = currHandEnd - currHandBegin;

        double smallestDistance = FLT_MAX;
        std::vector<pointSelection>::iterator deselectionPoint;

        for (std::vector<pointSelection>::iterator iter = selectedPoints.begin(); iter !=selectedPoints.end(); iter++)
        {
            int i = iter->pointSetIndex;
            int j = iter->pointIndex;
            Vec3 currentPoint = Vec3(iter->file->pointSet[i].points[j].x,iter->file->pointSet[i].points[j].y,iter->file->pointSet[i].points[j].z);
            double distance = LinePointDistance(currentPoint, currHandBegin, currHandDirection);
            if (distance<smallestDistance)
            {
                smallestDistance=distance;
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
    updateMessage(selectedPoints);
    return hitPointSuccess;
}

void PointCloudInteractor::setTranslation(bool translation)
{
	if (translation)
		m_translation = true;
	else
		m_translation = false;
}

void PointCloudInteractor::setRotation(bool rotation)
{
	if (rotation)
		m_rotation = true;
	else
		m_rotation = false;
}

void PointCloudInteractor::setDeselection(bool deselection)
{
    if (deselection)
        m_deselection = true;
    else
        m_deselection = false;
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
