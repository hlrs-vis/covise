/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10/18/2010
 **
 **************************************************************************/

#ifndef LANEEDITORTOOL_HPP
#define LANEEDITORTOOL_HPP

#include "editortool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"
#include "ui_LaneRibbon.h"

class QDoubleSpinBox;

class LaneEditorTool : public EditorTool
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit LaneEditorTool(ToolManager *toolManager);
    virtual ~LaneEditorTool()
    { /* does nothing */
    }


private:
    LaneEditorTool(); /* not allowed */
    LaneEditorTool(const LaneEditorTool &); /* not allowed */
    LaneEditorTool &operator=(const LaneEditorTool &); /* not allowed */

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
    void activateEditor();
    void activateRibbonEditor();
    void setWidth();
    void setRibbonWidth();
    void handleToolClick(int);
    void handleRibbonToolClick(int);
    void onCheckBoxStateChanged(int);
    // void setEditMode();

        //################//
        // PROPERTIES     //
        //################//

private:
    Ui::LaneRibbon *ui;
    ODD::ToolId toolId_;
    QDoubleSpinBox *widthEdit_;

    QButtonGroup *ribbonToolGroup_;
};

class LaneEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneEditorToolAction(ODD::ToolId toolId, double width);
    virtual ~LaneEditorToolAction()
    { /* does nothing */
    }

    double getWidth() const
    {
        return width_;
    }

private:
    LaneEditorToolAction(); /* not allowed */
    LaneEditorToolAction(const LaneEditorToolAction &); /* not allowed */
    LaneEditorToolAction &operator=(const LaneEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    double width_;
};

#endif // LANEEDITORTOOL_HPP
