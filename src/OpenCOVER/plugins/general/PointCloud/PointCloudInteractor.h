/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINTCLOUD_INTERACTOR_H
#define POINTCLOUD_INTERACTOR_H

#include "Points.h"
#include "PointCloud.h"
#include "FileInfo.h"
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/Vec3>
#include <osg/ShapeDrawable>

#include <util/coRestraint.h>

#include <OpenVRUI/coTrackerButtonInteraction.h>
class FileInfo;
class NodeInfo;

class PointCloudInteractor : public vrui::coTrackerButtonInteraction
{
public:
    void boxSelect(osg::Vec3, osg::Vec3);

    // Interactor to pick spheres
    PointCloudInteractor(coInteraction::InteractionType type, const char *name, vrui::coInteraction::InteractionPriority priority);

    ~PointCloudInteractor();
    virtual bool destroy();

    // ongoing interaction
    void doInteraction();

    // stop the interaction
    void stopInteraction();

    // start the interaction
    void startInteraction();

	//calculate moving vector
	void CalcMoveVec(std::vector<osg::Vec3>, std::vector<osg::Vec3>, string);

    void addSelectedPoint(osg::Vec3);
    void updateMessage(vector<pointSelection> points);

    // updates the temporary copy of the spheres
    // on them the intersection test takes place
    void updatePoints(const std::vector<FileInfo> *allFiles)
    {
        this->m_files = allFiles;
    };
	void getData(PointCloudInteractor *PCI);
	void setTranslation(bool);
	void setRotation(bool);

    void setDeselection(bool);

    // measure angle between hand position-point and hand-direction
    // returns -1 on pointing away from point
    // returns the cosine of angle
    double LinePointMeasure(osg::Vec3 center, osg::Vec3 handPos, osg::Vec3 handDirection);
    
    void resize();
    bool deselectPoint();
    void setSelectionSetIndex(int selectionSet);
    void setSelectionIsBoundary(bool selectionIsBoundary);
    bool getSelectionIsBoundary();
	bool actionsuccess =0;

private:

    // needed for interaction
    osg::Vec3 m_initHandPos, m_initHandDirection, OrgVec, RotAxis, PrevHandPos;
	string filename;
	void MoveCloud(osg::MatrixTransform *MoveMat, bool Snap);
	bool snapOn = false;

	osg::Matrixd RotMat = osg::Matrixd();
	osg::Matrixd TraMat = osg::Matrixd();
	std::vector<osg::Matrixd> PrevMats;
	std::vector<std::string> MatNames;
	bool SaveMat = true;
	osg::Matrix StartHandMat;
	osg::Vec3 OldVec;

	double OldAngle = 0;
	double PreAngle = 0;

	void MovePoints(osg::Matrixd MoveMat);
	void CloudMatrix();

    const std::vector<FileInfo> *m_files;
    bool m_selectedWithBox;

    void swap(float &m, float &n);

    bool hitPoint(pointSelection &bestPoint);
    void highlightPoint(pointSelection&, bool preview= false);
    bool hitPointSet(osg::Vec3 handDir, osg::Vec3 handPos, PointSet *pointset);
    double LinePointDistance(osg::Vec3 center, osg::Vec3 handPos, osg::Vec3 handDirection);

	std::vector<FileInfo> UpdatedFIVec;

	bool m_rotation = false;

	bool m_translation = false;

    bool hitPointSetBoundingSphere(osg::Vec3 handDir, osg::Vec3 handPos, osg::Vec3 center, float radius);

    osg::MatrixTransform *sc; 
    
    double sphereSize = 10.0;

    bool m_deselection = false;

    std::vector<pointSelection> selectedPoints;
    std::vector<pointSelection> previewPoints;
	std::vector<pointSelection> MovedPTS;
    //pointSelection previewPoint;

    osg::ref_ptr<osg::MatrixTransform> highlight;

    osg::ref_ptr<osg::Group> selectedPointsGroup;
    osg::ref_ptr<osg::Group> previewPointsGroup;
	osg::ref_ptr<osg::Group> MovedPTSGroup;

    int selectionSetIndex = 0;
    bool m_selectionIsBoundary = false;

};
#endif //POINTCLOUD_INTERACTOR_H
