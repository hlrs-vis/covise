/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef CROSSFALLEDITOR_HPP
#define CROSSFALLEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;

class TopviewGraph;
class ProfileGraph;

class RSystemElementRoad;

class SectionHandle;
class CrossfallMoveHandle;

class RoadSystemItem;

class CrossfallRoadSystemItem;
class CrossfallRoadItem;
class CrossfallSectionItem;
class CrossfallRoadPolynomialItem;

#include <QPointF>
#include <QMap>
#include <QMultiMap>

class CrossfallEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph);
    virtual ~CrossfallEditor();

    // TODO look for better solution
    SectionHandle *getInsertSectionHandle();

    // ProfileGraph //
    //
    ProfileGraph *getProfileGraph() const
    {
        return profileGraph_;
    }

    // Smooth Radius //
    //
    double getSmoothRadius() const
    {
        return smoothRadius_;
    }

    // Selected Roads //
    //
    void addSelectedRoad(RSystemElementRoad *road);
    int delSelectedRoad(RSystemElementRoad *road);

    // MoveHandles //
    //
    void registerMoveHandle(CrossfallMoveHandle *handle);
    int unregisterMoveHandle(CrossfallMoveHandle *handle);
    bool translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    //	virtual void			mouseAction(MouseAction * mouseAction);
    //	virtual void			keyAction(KeyAction * keyAction);

protected:
    virtual void init();
    virtual void kill();

private:
    CrossfallEditor(); /* not allowed */
    CrossfallEditor(const CrossfallEditor &); /* not allowed */
    CrossfallEditor &operator=(const CrossfallEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // Graph //
    //
    CrossfallRoadSystemItem *roadSystemItem_;

    // ProfileGraph //
    //
    ProfileGraph *profileGraph_;
    RoadSystemItem *roadSystemItemPolyGraph_;

    // TODO look for better solution
    SectionHandle *insertSectionHandle_;

    // Smooth Radius //
    //
    double smoothRadius_;

    // ProfileGraph: Selected Items //
    //
    QMap<RSystemElementRoad *, CrossfallRoadPolynomialItem *> selectedCrossfallRoadItems_;
    QMultiMap<int, CrossfallMoveHandle *> selectedMoveHandles_;
};

#endif // CROSSFALLEDITOR_HPP
