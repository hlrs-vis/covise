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

#ifndef TOOL_HPP
#define TOOL_HPP


#include <QMap>

#include "src/util/odd.hpp"
#include "toolparameter.hpp"
#include "toolvalue.hpp"

class Tool
{

public:
	explicit Tool(ODD::ToolId id, int listSize);
	virtual ~Tool();

	ODD::ToolId getToolId()
	{
		return id_;
	}
	void readParams(ToolParameter *s);

	template<class...Arg>
	void readParams(ToolParameter *start, Arg... arg);

	template<class T>
	int deleteValue(ToolValue<T> *v)
	{
		if (v->getType() == ToolParameter::OBJECT_LIST)
		{
			int k;
			QMap<unsigned int, ToolParameter *>::iterator paramIt = params_.begin();
			while (paramIt != params_.end())
			{
				if (paramIt.value() == v)
				{
					k = paramIt.key();
					params_.remove(k);
					return k;
				}
				paramIt++;
			}

			QList<ToolParameter *> list;
			QMap<unsigned int, QList<ToolParameter *>>::iterator it = paramList_.begin();
			while (it != paramList_.constEnd())
			{
				if (it.value().contains(v))
				{
					k = it.key();
					list = it.value();
					break;
				}
				it++;
			}

			if (list.size() == listSize_) // separate list in single parameters
			{ 
				ToolParameter *emptyParam = NULL;
				paramIt = params_.end();
				do
				{
					paramIt--;
					if (paramIt.value()->getToolId() == v->getToolId())
					{
						emptyParam = paramIt.value();
						break;
					}
				} while (paramIt != params_.begin());
				params_.remove(paramIt.key());

				foreach(ToolParameter *param, list)
				{
					it.value().removeOne(param);
					if (v == param)
					{
						delete v;
					}
					else
					{
						params_.insert(generateParamId(), param);
					}
				}
				params_.insert(generateParamId(), emptyParam);
				paramList_.remove(k);
				return k;
			}
			else
			{
				foreach(ToolParameter *param, list)
				{
					if (v == param)
					{
						it.value().removeOne(param);
						if (it.value().isEmpty())
						{
							paramList_.remove(k);
						}
						delete v;
						return k;
					}
				}
			}
		}
		return -1;
	}

	template<class T>
	QList<T*> deleteValue(int id)
	{
		QList<T*> objectList;

		if (paramList_.contains(id))
		{
			QList<ToolParameter *> list = paramList_.value(id);
			for (int i = 0; i < list.size();)
			{
				ToolParameter *p = list.takeAt(i);

				ToolValue<T> *v = dynamic_cast<ToolValue<T> *>(p);
				objectList.append(v->getValue());
				delete v;
			}
			paramList_.remove(id);
		}
		else
		{
			ToolParameter *p = params_.value(id);
			ToolValue<T> *v = dynamic_cast<ToolValue<T> *>(p);
			objectList.append(v->getValue());
			delete v;

			params_.remove(id);
		}

		return objectList;
	}

	template<class T>
	QList<T *> removeToolParameters(int id)
	{
		ToolParameter *p = getLastParam(id);
		QList<T *> objectList = deleteValue<T>(id);

		return objectList;
	}

	template<class T>
	ToolValue<T> *getValue(T *value)
	{
		foreach(QList<ToolParameter *> params, paramList_.values())
		{
			foreach(ToolParameter * p, params)
			{
				ToolValue<T> *toolValue = dynamic_cast<ToolValue<T> *>(p);
				if (toolValue)
				{
					if (toolValue->getValue() == value)
					{
						return toolValue;
					}
				}
			}
		}

		foreach(ToolParameter *p, params_.values())
		{
			ToolValue<T> *toolValue = dynamic_cast<ToolValue<T> *>(p);
			if (toolValue)
			{
				if (toolValue->getValue() == value)
				{
					return toolValue;
				}
			}
		}
		return NULL;
	}

	template<class T>
	T *getValue(int id)
	{
		ToolParameter *p = getLastParam(id);
		ToolValue<T> *v = dynamic_cast<ToolValue<T> *>(p);

		return v->getValue();
	}

	QMap<unsigned int, QList<ToolParameter *>> *getParamList()
	{
		return &paramList_;
	}

	QMap<unsigned int, ToolParameter *> *getParams()
	{
		return &params_;
	}

	unsigned int generateParamId();
	int getParamId(ToolParameter *s);
/*	ToolParameter *getParam(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId); */

	QList<ToolParameter *> getParamList(unsigned char listId);
	ToolParameter * getLastParam(unsigned char listId);
	ToolParameter *getParam(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId);

	int getListSize()
	{
		return listSize_;
	}

	int getObjectCount(const ODD::ToolId &id);
	int getObjectCount(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId);

	bool verify();

private:
	Tool(); /* not allowed */
	Tool(const Tool &); /* not allowed */
	Tool &operator=(const Tool &); /* not allowed */

private:
	ODD::ToolId id_;
	int listSize_;

	QMap<unsigned int, QList<ToolParameter *>> paramList_;
	QMap<unsigned int, ToolParameter *> params_;
};


#endif // TOOL_HPP
