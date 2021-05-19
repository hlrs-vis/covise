/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   31.03.2010
 **
 **************************************************************************/

#ifndef TYPEEDITORTOOL_HPP
#define TYPEEDITORTOOL_HPP

#include "editortool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

#include <QMap>

class QGroupBox;
class QAction;
class QMenu;
class QToolButton;
class RoadTypeComboBox;

class TypeEditorTool : public EditorTool
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit TypeEditorTool(ToolManager *toolManager);
    virtual ~TypeEditorTool()
    { /* does nothing */
    }

private:
    TypeEditorTool(); /* not allowed */
    TypeEditorTool(const TypeEditorTool &); /* not allowed */
    TypeEditorTool &operator=(const TypeEditorTool &); /* not allowed */

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
    void activateProject(bool hasActive);
    void activateEditor();
    void activateRibbonEditor();
    void handleToolClick(int);
    void handleRoadTypeSelection(int);

    //################//
    // PROPERTIES     //
    //################//

private:
    ODD::ToolId toolId_;
    TypeSection::RoadType roadType_;

    // GUI Elements //
    //
    QMap<QString, TypeSection::RoadType> roadTypes_;
    QGroupBox *selectGroupBox_;
    QMenu *roadTypeToolButtonMenu_;
    QToolButton *roadTypeToolButton_;
    QButtonGroup *ribbonToolGroup_;

    bool active_;
};

class TypeEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TypeEditorToolAction(ODD::ToolId toolId);
    virtual ~TypeEditorToolAction()
    { /* does nothing */
    }

private:
    TypeEditorToolAction(); /* not allowed */
    TypeEditorToolAction(const TypeEditorToolAction &); /* not allowed */
    TypeEditorToolAction &operator=(const TypeEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // TYPEEDITORTOOL_HPP
