/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   06.04.2010
**
**************************************************************************/

#ifndef TRACKEDITORTOOL_HPP
#define TRACKEDITORTOOL_HPP

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"

#include "src/data/prototypemanager.hpp"

// Qt //
//
#include <QMap>
class QGroupBox;
//class QAction;
//class QMenu;
//class QToolButton;

class TrackEditorTool : public Tool
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief Ids of the TrackEditor tools.
	*
	* This enum defines the Id of each tool.
	*/
    //	enum TrackEditorToolId
    //	{
    //		TTE_UNKNOWN,
    //		TTE_SELECT,
    //		TTE_INSERT,
    //		TTE_DELETE
    //	};

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackEditorTool(PrototypeManager *prototypeManager, ToolManager *toolManager);
    virtual ~TrackEditorTool()
    { /* does nothing */
    }

protected:
private:
    TrackEditorTool(); /* not allowed */
    TrackEditorTool(const TrackEditorTool &); /* not allowed */
    TrackEditorTool &operator=(const TrackEditorTool &); /* not allowed */

    void initToolBar();
    void initToolWidget();

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);

    //################//
    // SLOTS          //
    //################//

public slots:
    void sendToolAction();

    void activateEditor();
    void handleToolClick(int);
    void handleRoadTypeSelection(int);
    void handleTrackSelection(int);
    void handleElevationSelection(int);
    void handleSuperelevationSelection(int);
    void handleCrossfallSelection(int);
    void handleLaneSectionSelection(int);
    void handleRoadSystemSelection(int);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    PrototypeManager *prototypeManager_;

    ODD::ToolId toolId_;

    QMap<PrototypeManager::PrototypeType, RSystemElementRoad *> currentPrototypes_;
    RoadSystem *currentRoadSystemPrototype_;

    // GUI Elements //
    //
    QGroupBox *sectionPrototypesGroupBox_;
    QGroupBox *trackPrototypesGroupBox_;
    QGroupBox *roadSystemPrototypesGroupBox_;
};

class TrackEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackEditorToolAction(ODD::ToolId toolId, QMap<PrototypeManager::PrototypeType, RSystemElementRoad *> prototypes, RoadSystem *roadSystemPrototype);
    virtual ~TrackEditorToolAction()
    { /* does nothing */
    }

    // RoadPrototype //
    //
    QMap<PrototypeManager::PrototypeType, RSystemElementRoad *> getPrototypes() const
    {
        return prototypes_;
    }

    // RoadSystemPrototype //
    //
    RoadSystem *getRoadSystemPrototype() const
    {
        return roadSystemPrototype_;
    }

protected:
private:
    TrackEditorToolAction(); /* not allowed */
    TrackEditorToolAction(const TrackEditorToolAction &); /* not allowed */
    TrackEditorToolAction &operator=(const TrackEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // RoadPrototype //
    //
    QMap<PrototypeManager::PrototypeType, RSystemElementRoad *> prototypes_;

    // RoadSystemPrototype //
    //
    RoadSystem *roadSystemPrototype_;
};

#endif // TRACKEDITORTOOL_HPP
