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

#ifndef CROSSFALLEDITORTOOL_HPP
#define CROSSFALLEDITORTOOL_HPP

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"
#include "ui_CrossfallRibbon.h"

class QDoubleSpinBox;

class CrossfallEditorTool : public Tool
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallEditorTool(ToolManager *toolManager);
    virtual ~CrossfallEditorTool()
    { /* does nothing */
    }

private:
    CrossfallEditorTool(); /* not allowed */
    CrossfallEditorTool(const CrossfallEditorTool &); /* not allowed */
    CrossfallEditorTool &operator=(const CrossfallEditorTool &); /* not allowed */

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
    void setRadius();
	void activateRibbonEditor();
	void handleRibbonToolClick(int);
	void setRibbonRadius();

    //################//
    // PROPERTIES     //
    //################//

private:
    ODD::ToolId toolId_;
	Ui::CrossfallRibbon *ui_;

    QDoubleSpinBox *radiusEdit_;
};

class CrossfallEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallEditorToolAction(ODD::ToolId toolId, double radius);
    virtual ~CrossfallEditorToolAction()
    { /* does nothing */
    }

    double getRadius() const
    {
        return radius_;
    }
    void setRadius(double radius);

private:
    CrossfallEditorToolAction(); /* not allowed */
    CrossfallEditorToolAction(const CrossfallEditorToolAction &); /* not allowed */
    CrossfallEditorToolAction &operator=(const CrossfallEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:

    double radius_;
};

#endif // CROSSFALLEDITORTOOL_HPP
