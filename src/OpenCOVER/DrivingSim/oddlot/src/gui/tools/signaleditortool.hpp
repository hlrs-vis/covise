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

#ifndef SIGNALEDITORTOOL_HPP
#define SIGNALEDITORTOOL_HPP

#include "tool.hpp"

#include "toolaction.hpp"
#include "src/util/odd.hpp"

#include "ui_SignalRibbon.h"

class QGroupBox;

class SignalEditorTool : public Tool
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalEditorTool(ToolManager *toolManager);
    virtual ~SignalEditorTool()
    { /* does nothing */
    }

	void signalSelection(bool);

private:
    SignalEditorTool(); /* not allowed */
    SignalEditorTool(const SignalEditorTool &); /* not allowed */
    SignalEditorTool &operator=(const SignalEditorTool &); /* not allowed */

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
    void handleToolClick(int);

    //################//
    // PROPERTIES     //
    //################//

private:
    ODD::ToolId toolId_;
	Ui::SignalRibbon *ui;

    bool active_;
};

class SignalEditorToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalEditorToolAction(ODD::ToolId toolId);
    virtual ~SignalEditorToolAction()
    { /* does nothing */
    }

private:
    SignalEditorToolAction(); /* not allowed */
    SignalEditorToolAction(const SignalEditorToolAction &); /* not allowed */
    SignalEditorToolAction &operator=(const SignalEditorToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // TYPEEDITORTOOL_HPP
