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
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nPointCloudInteractor::startMove\n");
    }
	actionsuccess = true;

    // store hand mat
    const Matrix &initHandMat = cover->getPointerMat();
    // get postion and direction of pointer
    m_initHandPos = initHandMat.preMult(Vec3(0.0, 0.0, 0.0));
    m_initHandDirection = initHandMat.preMult(Vec3(0.0, 1.0, 0.0));
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
		if (selectedPoints.size() !=  0)
		{
			filename = selectedPoints[0].file->filename;
		}
		actionsuccess = true;
		if (type == ButtonD)
		{	
			selectedPointsGroup->removeChild(selectedPointsGroup->getNumChildren() - 1);
			selectedPoints.pop_back();
			actionsuccess = true;
			//if (snapOn)
			//{
			//	snapOn = false;
			//}
			//else
			//{
			//	snapOn = true;
			//}
			//if (m_files->size() >= 2)
			//{
			//	Matrixd TraSnap;
			//	if (m_translation&& selectedPoints.size() >= 2)
			//	{
			//		std::vector<Vec3> PtSVec;
			//		for (std::vector<pointSelection>::iterator i = selectedPoints.begin(); i < selectedPoints.end(); i++)
			//		{
			//			PtSVec.push_back(Vec3(i->file->pointSet[i->pointSetIndex].points[i->pointIndex].x, i->file->pointSet[i->pointSetIndex].points[i->pointIndex].y, i->file->pointSet[i->pointSetIndex].points[i->pointIndex].z));
			//		}
			//		TraSnap.makeTranslate(PtSVec[1] - PtSVec[0]);
			//		MovePoints(TraSnap);
			//		osg::MatrixTransform *SnapCloud = new osg::MatrixTransform();
			//		SnapCloud->setMatrix(TraSnap);
			//		MoveCloud(SnapCloud, TRUE);
			//		selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			//		selectedPoints.clear();
			//	}
			//	if (m_rotation && selectedPoints.size() >= 6)
			//	{
			//		std::vector<Vec3> PtVecs1;
			//		std::vector<Vec3> PtVecs2;
			//		for (std::vector<pointSelection>::iterator i = selectedPoints.begin(); i < selectedPoints.end(); i++)
			//		{
			//			Vec3 Coord = Vec3(i->file->pointSet[i->pointSetIndex].points[i->pointIndex].x, i->file->pointSet[i->pointSetIndex].points[i->pointIndex].y, i->file->pointSet[i->pointSetIndex].points[i->pointIndex].z);
			//			if (i->file->filename == filename)
			//			{
			//				PtVecs1.push_back(Coord);
			//			}
			//			else
			//			{
			//				PtVecs2.push_back(Coord);
			//			}
			//		}
			//		Vec3 Axis1, Axis2, RotPt1, RotPt2, OrthoVec1, OrthoVec2;
			//		double ang;
			//		Axis1 = PtVecs1[1] - PtVecs1[0];
			//		Axis2 = PtVecs2[1] - PtVecs2[0];
			//		RotPt1 = PtVecs1[2];
			//		RotPt2 = PtVecs2[2];
			//		OrthoVec1 = Axis1 ^ (RotPt1 - PtVecs1[0]);
			//		OrthoVec2 = Axis2 ^ (RotPt2 - PtVecs2[0]);
			//		ang = acos((OrthoVec1 * OrthoVec2) / (OrthoVec1.length()* OrthoVec2.length()));
			//		Matrixd RotSnap;
			//		TraSnap.makeTranslate(-PtVecs1[0]);
			//		RotSnap.makeRotate(-ang, Axis2);
			//		RotSnap = TraSnap * RotSnap;
			//		TraSnap.makeTranslate(PtVecs2[0]);
			//		RotSnap = RotSnap * TraSnap;
			//		MovePoints(RotSnap);
			//		MatrixTransform *SnapCloud = new MatrixTransform();
			//		SnapCloud->setMatrix(RotSnap);
			//		MoveCloud(SnapCloud, TRUE);
			//		selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
			//		selectedPoints.clear();
			//	}
			//}
		}
		else
		{
			pointSelection bestPoint;
			bool hitPointSuccess = hitPoint(bestPoint);
			if (hitPointSuccess)
			{
				highlightPoint(bestPoint);
				if (!snapOn)
				{
					if (m_rotation)
					{
						if (SaveMat && selectedPoints.size() == 3)
						{
							// store start mat for moving points
							StartHandMat = SaveHandMat * cover->getInvBaseMat();
							StartHandMat.inverse(StartHandMat);
							SaveMat = false;
						}
						if (selectedPoints.size() >= 4)
						{
							MovePoints(RotMat);
							for (int i = 0; i < MatNames.size(); i++)
							{
								if (MatNames[i] == selectedPoints[0].file->filename)
								{
									PrevMats[i] = PrevMats[i] * RotMat;
									break;
								}
							}
							RotMat.makeIdentity();
							TraMat.makeIdentity();
							OrgDirect = Vec3();
							SaveMat = true;
							selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
							selectedPoints.clear();
						}
					}
					if (m_translation && selectedPoints.size() >= 2)
					{
						std::vector<Vec3> TraVec;
						for (std::vector<pointSelection>::iterator Pts = selectedPoints.begin(); Pts < selectedPoints.end(); Pts++)
						{
							Vec3 PointCoord = Vec3(Pts->file->pointSet[Pts->pointSetIndex].points[Pts->pointIndex].x, Pts->file->pointSet[Pts->pointSetIndex].points[Pts->pointIndex].y, Pts->file->pointSet[Pts->pointSetIndex].points[Pts->pointIndex].z);
							TraVec.push_back(PointCoord);
						}
						TraMat.makeTranslate(TraVec[1] - TraVec[0]);
						MovePoints(TraMat);
						for (int i = 0; i < MatNames.size(); i++)
						{
							if (MatNames[i] == selectedPoints[0].file->filename)
							{
								PrevMats[i] = PrevMats[i] * TraMat;
								break;
							}
						}
						TraMat.makeIdentity();
						OrgDirect = Vec3();
						selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
						selectedPoints.clear();
						SaveMat = true;
					}					
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
    if (m_files != nullptr)
    {
        Matrix currHandMat = cover->getPointerMat();
        const Matrix &invBase = cover->getInvBaseMat();

        currHandMat = currHandMat * invBase;
		SaveHandMat = currHandMat;
        Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
        Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
        Vec3 currHandDirection = currHandEnd - currHandBegin;
		if (((m_rotation && selectedPoints.size() >= 3)||(m_translation && selectedPoints.size() >= 1)) && !snapOn)
		{
			Vec3 newVec, PtToMove;
			osg::MatrixTransform *MoTra = new osg::MatrixTransform();
			
			bool success = false;
			if (m_rotation)
			{
				Vec3 AxisStart, AxisEnd;
				AxisStart.x() = selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].x;  
				AxisStart.y() = selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].y;
				AxisStart.z() = selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].z;

				AxisEnd.x() = selectedPoints[1].file->pointSet[selectedPoints[1].pointSetIndex].points[selectedPoints[1].pointIndex].x;
				AxisEnd.y() = selectedPoints[1].file->pointSet[selectedPoints[1].pointSetIndex].points[selectedPoints[1].pointIndex].y;
				AxisEnd.z() = selectedPoints[1].file->pointSet[selectedPoints[1].pointSetIndex].points[selectedPoints[1].pointIndex].z;

				PtToMove.x() = selectedPoints[2].file->pointSet[selectedPoints[2].pointSetIndex].points[selectedPoints[2].pointIndex].x;
				PtToMove.y() = selectedPoints[2].file->pointSet[selectedPoints[2].pointSetIndex].points[selectedPoints[2].pointIndex].y;
				PtToMove.z() = selectedPoints[2].file->pointSet[selectedPoints[2].pointSetIndex].points[selectedPoints[2].pointIndex].z;
				if (OrgDirect.length() == 0)
				{
					OrgDirect = PtToMove - currHandBegin;
					RotAxis = AxisEnd - AxisStart;
					CenterPoint = AxisStart + (((RotAxis / RotAxis.length()) * ((PtToMove - AxisStart)* RotAxis / RotAxis.length())));
					RotRadius = (PtToMove - CenterPoint).length();
				}
				double MovPointer = (currHandBegin - CenterPoint) * RotAxis / RotAxis.length();
				double a = -(MovPointer / (currHandDirection * RotAxis / RotAxis.length()));
				Vec3 Schnittpunkt = currHandBegin + currHandDirection * a;
				Schnittpunkt = CenterPoint + ((Schnittpunkt - CenterPoint) * (RotRadius / (Schnittpunkt - CenterPoint).length()));
				Vec3 TransformedPT = (StartHandMat * currHandMat).preMult(PtToMove);
				RotMat = StartHandMat * currHandMat;
				newVec = RotMat.preMult(PtToMove) + (Schnittpunkt - TransformedPT);
				RotMat.makeRotate(PtToMove - CenterPoint, Schnittpunkt - CenterPoint);
				TraMat.makeTranslate(-CenterPoint);
				RotMat = TraMat * RotMat * TraMat.inverse(TraMat);
				MoTra->setMatrix(RotMat);
				success = true;
			}
			if (m_translation)
			{				
				PtToMove = Vec3(selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].x, selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].y, selectedPoints[0].file->pointSet[selectedPoints[0].pointSetIndex].points[selectedPoints[0].pointIndex].z);
				if (OrgDirect.length() == 0)
				{
					OrgDirect = PtToMove - currHandBegin;
				}
				Vec3 Direct = currHandDirection * (OrgDirect.length() / currHandDirection.length());
				newVec = Direct - OrgDirect;
				TraMat.makeTranslate(newVec);
				MoTra->setMatrix(TraMat);
				success = true;
				newVec = TraMat.preMult(PtToMove);
			}
			if (success)
			{
				MoveCloud(MoTra, false);
			}
			bestPoint.pointSetIndex = selectedPoints[selectedPoints.size() - 1].pointSetIndex;
			bestPoint.pointIndex = selectedPoints[selectedPoints.size() - 1].file->pointSet->size + 1;
			bestPoint.file = selectedPoints[selectedPoints.size() - 1].file;
			hitPointSuccess = true;
			bestPoint.selectionIndex = selectionSetIndex;
			bestPoint.isBoundaryPoint = getSelectionIsBoundary();
			bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].x = newVec.x();
			bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].y = newVec.y();
			bestPoint.file->pointSet[bestPoint.pointSetIndex].points[bestPoint.pointIndex].z = newVec.z();
		}
		else
		{
			double smallestDistance = FLT_MAX;
			for (std::vector<FileInfo>::const_iterator fit = m_files->begin(); fit != m_files->end(); fit++)
			{
				//Control if snapping is activated
				if (fit->filename == FileToMove || FileToMove == "")
				{
					if (snapOn)
					{
						//Set Selection Criteria if Rotation Mode is set active
						if (m_rotation)
						{
							if (selectedPoints.size() >= 3)
							{
								if (selectedPoints.size() >= 4)
								{
									if (selectedPoints[3].file->filename != fit->filename)
									{
										continue;
									}
								}
								if (selectedPoints[2].file->filename == fit->filename)
								{
									continue;
								}
							}
							else
							{
								if (selectedPoints.size() >= 1)
								{
									if (selectedPoints[0].file->filename != fit->filename)
									{
										continue;
									}
								}
							}
						}
						//set selection criteria if translation mode is set active
						if (m_translation)
						{
							if (selectedPoints.size() >= 1)
							{
								if (selectedPoints[0].file->filename == fit->filename)
								{
									continue;
								}
							}
						}

					}
					if (fit->pointSet)
					{
						//std::string fileName = fit->filename;
						//fprintf(stderr, "Testing Set %s \n", fileName.c_str());
						for (int i = 0; i < fit->pointSetSize; i++)
						{
							Vec3 center = Vec3((fit->pointSet[i].xmin + fit->pointSet[i].xmax) / 2, (fit->pointSet[i].ymin + fit->pointSet[i].ymax) / 2, (fit->pointSet[i].zmin + fit->pointSet[i].zmax) / 2);
							Vec3 corner = Vec3(fit->pointSet[i].xmin, fit->pointSet[i].ymin, fit->pointSet[i].zmin);
							Vec3 radiusVec = center - corner;
							float radius = 1.5 * radiusVec.length();

							if (hitPointSetBoundingSphere(currHandDirection, currHandBegin, center, radius))
							{
								for (int j = 0; j < fit->pointSet[i].size; j++)
								{
									Vec3 currentPoint = Vec3(fit->pointSet[i].points[j].x, fit->pointSet[i].points[j].y, fit->pointSet[i].points[j].z);
									double distance = LinePointMeasure(currentPoint, currHandBegin, currHandDirection);
									if (distance < smallestDistance)
									{
										smallestDistance = distance;
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
		}
    }
    return hitPointSuccess;
}

void
PointCloudInteractor::CloudMatrix()
{
	for (vector<FileInfo>::const_iterator i = m_files->begin(); i < m_files->end(); i++)
	{
		bool Matexist = false;
		for (int k = 0; k < MatNames.size(); k++)
		{
			if (MatNames[k] == i->filename)
			{
				Matexist = true;
				break;
			}
		}		
		if (!Matexist)
		{
			Matrixd Mat;
			PrevMats.push_back(Mat);
			MatNames.push_back(i->filename);
		}
	}
}

void 
PointCloudInteractor::MoveCloud(osg::MatrixTransform *MoveTra, bool Snap)
{
	osg::Node *oldDrawing = new osg::Node();
	osg::Node *newDrawing = new osg::Node();
	osg::Node *nGroupNode = new osg::Node();
	for (unsigned int i = 0; i < cover->getObjectsRoot()->getNumChildren(); i++)
	{
		nGroupNode = cover->getObjectsRoot()->getChild(i);
		if (nGroupNode->getName() == selectedPoints[0].file->filename)
		{
			oldDrawing = nGroupNode->asGroup()->getChild(0);
			break;
		}
	}
	if (oldDrawing->asTransform() != 0)
	{
		for (int i = 0; i < MatNames.size(); i++)
		{
			if (MatNames[i] == selectedPoints[0].file->filename)
			{
				if (Snap)
				{
					oldDrawing->asTransform()->asMatrixTransform()->setMatrix(oldDrawing->asTransform()->asMatrixTransform()->getMatrix() * MoveTra->getMatrix());
					PrevMats[i] = oldDrawing->asTransform()->asMatrixTransform()->getMatrix();
				}
				else
				{
					oldDrawing->asTransform()->asMatrixTransform()->setMatrix(PrevMats[i] * MoveTra->getMatrix());
				}
				break;
			}
		}
	}
	else
	{
		MoveTra->setName("MovedGeode");
		MoveTra->addChild(oldDrawing);
		newDrawing = MoveTra;
		newDrawing->setName(nGroupNode->getName() + " Transformed");
		nGroupNode->asGroup()->replaceChild(oldDrawing, newDrawing);
	}
}

void 
PointCloudInteractor::doInteraction()
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nPointCloudInteractor::stopMove\n");
    }
	if (type != ButtonD)
	{
		if (m_files != nullptr)
		{
			if (m_files->size() != PrevMats.size())
			{
				CloudMatrix();
			}
		}
		pointSelection bestPoint;
		bool hitPointSuccess = hitPoint(bestPoint);
		if (hitPointSuccess)
		{
			highlightPoint(bestPoint, true);
		}
	}
}

void  
PointCloudInteractor::MovePoints(osg::Matrixd MoveMat)
{
	std::vector<FileInfo> Files;
	Files = *m_files;
	MovedPTSGroup->removeChildren(0, MovedPTSGroup->getNumChildren());
	UpdatedFIVec.clear();
	for (std::vector<FileInfo>::iterator File = Files.begin(); File != Files.end(); File++)
	{
		FileInfo FI;
		FI = *File;
		if (filename == File->filename)
		{
			PointSet *PTSet;
			PTSet = File->pointSet;
			NodeInfo NI;
			for (int i = 0; i < PTSet->size; i++)
			{
				Vec3 PtCoord = Vec3(PTSet->points[i].x, PTSet->points[i].y, PTSet->points[i].z);
				PtCoord = MoveMat.preMult(PtCoord);
				PTSet->points[i].x = PtCoord.x();
				PTSet->points[i].y = PtCoord.y();
				PTSet->points[i].z = PtCoord.z();
			}
			FI.nodes.clear();
			FI.pointSet = PTSet;
			PointCloudGeometry *drawable = new PointCloudGeometry(FI.pointSet);
			drawable->changeLod(1.f);
			drawable->setPointSize(1.f);
			drawable->setName("Drawable Gedrehte Punkte");
			Geode *nGeode = new Geode();
			nGeode->addDrawable(drawable);
			nGeode->setName(FI.filename);
			NI.node = nGeode;
			FI.nodes.push_back(NI);
			FI.filename = File->filename;
			MovedPTSGroup->setName("Moved Group");
			MovedPTSGroup->addChild(FI.nodes[0].node);
		}
		UpdatedFIVec.push_back(FI);
	}
	updatePoints(&UpdatedFIVec);
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
    if (m_files != nullptr)
    {
		if (type == ButtonD)
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
				Vec3 currentPoint = Vec3(iter->file->pointSet[i].points[j].x, iter->file->pointSet[i].points[j].y, iter->file->pointSet[i].points[j].z);
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
		m_translation = true;
	else
		m_translation = false;
}

void PointCloudInteractor::setRotation(bool rotation)
{
	if (rotation)
	{
		m_rotation = true;
	}
	else
		m_rotation = false;
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
	FileToMove = PCI->FileToMove;
	m_files = PCI->m_files;
	MatNames = PCI->MatNames;
	MovedPTSGroup = PCI->MovedPTSGroup;
	PrevMats = PCI->PrevMats;
	RotMat = PCI->RotMat;
	SaveHandMat = PCI->SaveHandMat;
	selectedPoints = PCI->selectedPoints;
	selectedPointsGroup = PCI->selectedPointsGroup;
	snapOn = PCI->snapOn;
	StartHandMat = PCI->StartHandMat;
	TraMat = PCI->TraMat;
}

void PointCloudInteractor::setFile(string filename)
{
	FileToMove = filename;
	selectedPointsGroup->removeChildren(0, selectedPointsGroup->getNumChildren());
	selectedPoints.clear();
	actionsuccess = true;
}
