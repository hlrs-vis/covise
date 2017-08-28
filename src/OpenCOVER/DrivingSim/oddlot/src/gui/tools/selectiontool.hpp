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

#ifndef SELECTIONTOOL_HPP
#define SELECTIONTOOL_HPP

#include "tool.hpp"
#include "toolaction.hpp"
#include "src/gui/keyaction.hpp"

#include <QPushButton>
#include <QAction>

class SelectionTool : public Tool
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief Ids of the zoom tools.
	*
	* This enum defines the Id of each tool.
	*/
    enum SelectionToolId
    {
        TSL_UNKNOWN,
        TSL_BOUNDINGBOX
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SelectionTool(ToolManager *toolManager);
    virtual ~SelectionTool()
    { /* does nothing */
    }

protected:
private:
    SelectionTool(); /* not allowed */
    SelectionTool(const SelectionTool &); /* not allowed */
    SelectionTool &operator=(const SelectionTool &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);

    //################//
    // SLOTS          //
    //################//

private slots:
    void activateProject(bool);
    void handleToolClick(int);

public slots:

    virtual void keyAction(KeyAction *keyAction);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    SelectionTool::SelectionToolId selectionToolId_;

    // Actions //
    //
    QPushButton *selectionBox_;
};

class SelectionToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SelectionToolAction(SelectionTool::SelectionToolId selectionToolId);
    virtual ~SelectionToolAction()
    { /* does nothing */
    }

    SelectionTool::SelectionToolId getSelectionToolId() const
    {
        return selectionToolId_;
    }

	bool SelectionToolAction::getBoundingBoxActive()
	{
		return boundingBoxActive_;
	}


protected:
private:
    SelectionToolAction(); /* not allowed */
    SelectionToolAction(const SelectionToolAction &); /* not allowed */
    SelectionToolAction &operator=(const SelectionToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    SelectionTool::SelectionToolId selectionToolId_;

	bool boundingBoxActive_;
};

#endif // SELECTIONTOOL_HPP
