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

#ifndef SHAPEEDITORTOOL_HPP
#define SHAPEEDITORTOOL_HPP

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"
#include "ui_ShapeRibbon.h"


class ShapeEditorTool : public Tool
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeEditorTool(ToolManager *toolManager);
    virtual ~ShapeEditorTool()
    { /* does nothing */
    }

private:
    ShapeEditorTool(); /* not allowed */
    ShapeEditorTool(const ShapeEditorTool &); /* not allowed */
    ShapeEditorTool &operator=(const ShapeEditorTool &); /* not allowed */

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
    void handleToolClick(int);
	void activateRibbonEditor();
	void handleRibbonToolClick(int);

    //################//
    // PROPERTIES     //
    //################//

private:
    ODD::ToolId toolId_;
	Ui::ShapeRibbon *ui_;
};

class ShapeEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeEditorToolAction(ODD::ToolId toolId);
    virtual ~ShapeEditorToolAction()
    { /* does nothing */
    }


private:
    ShapeEditorToolAction(); /* not allowed */
    ShapeEditorToolAction(const ShapeEditorToolAction &); /* not allowed */
    ShapeEditorToolAction &operator=(const ShapeEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:

};

#endif // SHAPEEDITORTOOL_HPP
