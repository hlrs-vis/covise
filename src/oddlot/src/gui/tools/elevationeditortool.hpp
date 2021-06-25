/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   21.06.2010
 **
 **************************************************************************/

#ifndef ELEVATIONEDITORTOOL_HPP
#define ELEVATIONEDITORTOOL_HPP

#include "editortool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"
#include "ui_ElevationRibbon.h"

class QDoubleSpinBox;

class ElevationEditorTool : public EditorTool
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit ElevationEditorTool(ToolManager *toolManager);
    virtual ~ElevationEditorTool()
    { /* does nothing */
    }

private:
    ElevationEditorTool(); /* not allowed */
    ElevationEditorTool(const ElevationEditorTool &); /* not allowed */
    ElevationEditorTool &operator=(const ElevationEditorTool &); /* not allowed */

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
    void handleToolClick(int);
    void handleRibbonToolClick(int);
    void setRadius();
    void setHeight();
    void setIHeight();
    void setRHeight();
    void setRIHeight();
    void setRRadius();
    void setSectionStart();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ElevationRibbon *ui;
    ODD::ToolId toolId_;
    QButtonGroup *ribbonToolGroup_;

    QDoubleSpinBox *radiusEdit_;
    QDoubleSpinBox *heightEdit_;
    QDoubleSpinBox *iHeightEdit_;
};

class ElevationEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId, double radius, double height, double iHeight, double sectionStart = 0.0);
    virtual ~ElevationEditorToolAction()
    { /* does nothing */
    }

    double getRadius() const
    {
        return radius_;
    }
    void setRadius(double radius);
    double getHeight() const
    {
        return height_;
    }
    void setHeight(double height);
    double getIHeight()
    {
        return iHeight_;
    }
    void setIHeight(double iHeight);
    double getSectionStart() const
    {
        return start_;
    }
    void setSectionStart(double start);


private:
    ElevationEditorToolAction(); /* not allowed */
    ElevationEditorToolAction(const ElevationEditorToolAction &); /* not allowed */
    ElevationEditorToolAction &operator=(const ElevationEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    double radius_;
    double height_;
    double iHeight_;
    double start_;
};

#endif // ELEVATIONEDITORTOOL_HPP
