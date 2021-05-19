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

#include "editortool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"

 // Qt //
 //
#include <QMap>
class QGroupBox;

class TrackEditorTool : public EditorTool
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
    // enum TrackEditorToolId
    // {
    //  TTE_UNKNOWN,
    //  TTE_SELECT,
    //  TTE_INSERT,
    //  TTE_DELETE
    // };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackEditorTool(ToolManager *toolManager);
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

    void activateRibbonEditor();
    void handleToolClick(int);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:

    ODD::ToolId toolId_;

    QButtonGroup *ribbonToolGroup_;
};

class TrackEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackEditorToolAction(ODD::ToolId toolId);
    virtual ~TrackEditorToolAction()
    { /* does nothing */
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
};

#endif // TRACKEDITORTOOL_HPP
