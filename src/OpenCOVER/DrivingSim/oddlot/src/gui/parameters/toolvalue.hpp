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
#ifndef TOOLVALUE_HPP
#define TOOLVALUE_HPP

#include "toolparameter.hpp"

template<class T>
class ToolValue : public ToolParameter
{
public:
	explicit ToolValue<T>(ODD::ToolId toolId, ODD::ToolId paramToolId, char list, ParameterTypes type, const QString &text, const QString &labelText = "", T* value = NULL) :
		ToolParameter(toolId, paramToolId, list, type, text, labelText)
		, value_(value)
	{

	}

	virtual ~ToolValue()
	{
//		delete value_;
		value_ = NULL;
	};

	virtual void delParamValue()
	{
		value_ = NULL;
	}

	ToolValue<T> *clone()
	{
		ToolValue<T> *value = new ToolValue<T>(getToolId(), getParamToolId(), getListIndex(), getType(), getText(), value_);

		return value;
	}

	ToolValue<T> *parameterClone()
	{
		ToolValue<T> *value = new ToolValue<T>(getToolId(), getParamToolId(), getListIndex(), getType(), getText());

		return value;
	}

	T *getValue()
	{
		return value_;
	}

	void setValue(T value)
	{
		if (!value_)
		{
			value_ = new T;
		}
		*value_ = value;
		setValid(true);
	}

	void setValue(T *value)
	{
		value_ = value;
		if (value)
		{
			setValid(true);
		}
	}

	virtual bool verify()
	{
		if (!value_)
		{
			return false;
		}

		T *v = static_cast<T *>(value_);
		if (!v)
		{
			return false;
		}

		return true;
	}


private:
	ToolValue(); /* not allowed */
	ToolValue(const ToolValue &); /* not allowed */
	ToolValue &operator=(const ToolValue &); /* not allowed */

private:
	T *value_;
};

#endif // TOOLVALUE_HPP
