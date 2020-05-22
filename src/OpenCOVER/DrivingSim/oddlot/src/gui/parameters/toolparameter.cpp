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

#include "toolparameter.hpp"

ToolParameter::ToolParameter(ODD::ToolId toolId, ODD::ToolId paramToolId, char list, ParameterTypes type, const QString &text, const QString &labelText)
	: toolId_(toolId),
	paramToolId_(paramToolId),
	list_(list),
	valid_(false),
	type_(type),
	text_(text),
	labelText_(labelText)
{
}

ToolParameter::~ToolParameter()
{

}

void
ToolParameter::setText(const QString &text)
{
	text_ = text;
}