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

#include "toolwidget.hpp"

ToolWidget::ToolWidget(QWidget *parent)
    : QWidget(parent)
    , index_(-2)
{
}

void
ToolWidget::setToolBoxIndex(int index)
{
    index_ = index;
}

void
ToolWidget::activateWidget(int index)
{
    if (index == index_)
    {
        emit activated();
    }
}
