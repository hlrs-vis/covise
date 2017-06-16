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

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"
#include "ui_LaneRibbon.h"

class QDoubleSpinBox;

class LaneEditorTool : public Tool
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

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::LaneRibbon *ui;
    ODD::ToolId toolId_;
    QDoubleSpinBox *widthEdit_;
};

class LaneEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    enum ActionType
    {
        Width
    };
    explicit LaneEditorToolAction(ODD::ToolId toolId, ActionType at, double value);
    virtual ~LaneEditorToolAction()
    { /* does nothing */
    }

    double getWidth() const
    {
        return width;
    }
    void setWidth(double w);
    ActionType getType() const
    {
        return type;
    }

private:
    LaneEditorToolAction(); /* not allowed */
    LaneEditorToolAction(const LaneEditorToolAction &); /* not allowed */
    LaneEditorToolAction &operator=(const LaneEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    double width;
    ActionType type;
};

#endif // LANEEDITORTOOL_HPP
