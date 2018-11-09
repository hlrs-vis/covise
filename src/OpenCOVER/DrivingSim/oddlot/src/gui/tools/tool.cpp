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

#include "tool.hpp"

#include "toolmanager.hpp"

#include <QAbstractButton>

//################//
// CONSTRUCTOR    //
//################//

Tool::Tool(ToolManager *toolManager)
    : QObject(toolManager)
    , toolManager_(toolManager)
{
    // does nothing //
}

ToolButtonGroup::ToolButtonGroup(ToolManager *toolManager)
	: QButtonGroup() 
	, toolManager_(toolManager) 
{
	connect(toolManager_, SIGNAL(pressButton(int)), this, SLOT(setButtonPressed(int)));
};

void
ToolButtonGroup::setButtonPressed(int i)
{
	QAbstractButton *button = QButtonGroup::button(i);
	if (button)
	{
		button->setChecked(true);
	}
}
