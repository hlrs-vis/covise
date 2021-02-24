/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef TOOLPARAMETER_HPP
#define TOOLPARAMETER_HPP

#include "src/util/odd.hpp"

#include <QString>

class ToolParameter 
{
public:
	enum ParameterTypes
	{
		UINT = 0,
		INT = 1,
		USHORT = 2,
		SHORT = 3,
		STRING = 4,
		DOUBLE = 5,
		OBJECT = 6,
		OBJECT_LIST = 7,
		DATE_TIME = 8,
		ENUM = 9,
		BOOL = 10,
		FLOAT = 11,
	};


	explicit ToolParameter(ODD::ToolId toolId, ODD::ToolId paramToolId, char list, ParameterTypes type, const QString &text, bool active = false, const QString &labelText = "", const QString &valueDisplayed = "");
	virtual ~ToolParameter();

	ODD::ToolId getToolId()
	{
		return toolId_;
	}

	ODD::ToolId getParamToolId()
	{
		return paramToolId_;
	}

	virtual void delParamValue() = 0;

	void setValueDisplayed(const QString &text)
	{
		valueDisplayed_ = text;
	}

	QString getValueDisplayed()
	{
		return valueDisplayed_;
	} 

	ParameterTypes getType()
	{
		return type_;
	}

	QString getText()
	{
		return text_;
	}

	void setText(const QString &text);

	QString getLabelText()
	{
		return labelText_;
	}

	bool isList()
	{
		if (list_ > 0)
		{
			return true;
		}
		return false;
	}

	unsigned char getListIndex()
	{
		return list_;
	}

	bool isActive()
	{
		return active_;
	}

	void setActive(bool active)
	{
		active_ = active;
	}

	bool isValid()
	{
		return valid_;
	}

	void setValid(bool valid)
	{
		valid_ = valid;
	}

private:
	ToolParameter(); /* not allowed */
	ToolParameter(const ToolParameter &); /* not allowed */
	ToolParameter &operator=(const ToolParameter &); /* not allowed */

private:
	ODD::ToolId toolId_;
	ODD::ToolId paramToolId_;
	unsigned char list_;
	bool valid_;
	ParameterTypes type_;
	QString text_;
	bool active_;
	QString valueDisplayed_;
	QString labelText_;
};

#endif // TOOLPARAMETER_HPP
