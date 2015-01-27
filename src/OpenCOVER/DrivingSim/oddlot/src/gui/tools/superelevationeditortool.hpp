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

#ifndef SUPERELEVATIONEDITORTOOL_HPP
#define SUPERELEVATIONEDITORTOOL_HPP

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"

class QDoubleSpinBox;

class SuperelevationEditorTool : public Tool
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationEditorTool(ToolManager *toolManager);
    virtual ~SuperelevationEditorTool()
    { /* does nothing */
    }

private:
    SuperelevationEditorTool(); /* not allowed */
    SuperelevationEditorTool(const SuperelevationEditorTool &); /* not allowed */
    SuperelevationEditorTool &operator=(const SuperelevationEditorTool &); /* not allowed */

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

    //################//
    // PROPERTIES     //
    //################//

private:
    ODD::ToolId toolId_;

    QDoubleSpinBox *radiusEdit_;
};

class SuperelevationEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationEditorToolAction(ODD::ToolId toolId, double radius);
    virtual ~SuperelevationEditorToolAction()
    { /* does nothing */
    }

    double getRadius() const
    {
        return radius_;
    }
    void setRadius(double radius);

private:
    SuperelevationEditorToolAction(); /* not allowed */
    SuperelevationEditorToolAction(const SuperelevationEditorToolAction &); /* not allowed */
    SuperelevationEditorToolAction &operator=(const SuperelevationEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    double radius_;
};

#endif // SUPERELEVATIONEDITORTOOL_HPP
