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


#include "src/mainwindow.hpp"

//  GUI //
//
#include "toolparameter.hpp"
#include "toolparametersettings.hpp"

Tool::Tool(ODD::ToolId id, int listSize)
	: id_(id),
	listSize_(listSize)
{
}

Tool::~Tool()
{
	QMutableMapIterator<unsigned int, QList<ToolParameter *>> it(paramList_);
	while (it.hasNext())
	{
		it.next();
		QList<ToolParameter *> params = it.value();
		for (int i = 0; i < params.size();)
		{
			ToolParameter *param = params.takeAt(i);
			param->delParamValue();
			delete param;
		}

		paramList_.remove(it.key());
	}

	QMutableMapIterator<unsigned int, ToolParameter *> paramIt(params_);
	while (paramIt.hasNext())
	{
		paramIt.next();
		ToolParameter *param = params_.take(paramIt.key());
		param->delParamValue();
		delete param;
	}
}


void
Tool::readParams(ToolParameter *s)
{
	static int lastParamID = -1;

	if (s->getType() == ToolParameter::OBJECT_LIST)
	{
		ODD::ToolId toolId = s->getToolId();
		ODD::ToolId paramToolId = s->getParamToolId();
		
		int objectCount = getObjectCount(toolId, paramToolId);
		if (objectCount == listSize_)
		{
			QList<ToolParameter *>objectParameterList;
			QMutableMapIterator<unsigned int, ToolParameter*> it(params_);
			while (it.hasNext())
			{
				it.next();
				ToolParameter *param = it.value();
				if ((param->getType() == ToolParameter::OBJECT_LIST) && (param->getToolId() == toolId) && (param->getParamToolId() == paramToolId))
				{
					objectParameterList.append(param);
					if (lastParamID < 0)
					{
						lastParamID = it.key();
					}
					params_.remove(it.key());
				}
			}
			paramList_.insert(generateParamId(), objectParameterList);
			objectParameterList.clear();
			params_.insert(lastParamID, s);
		}
		else if (objectCount > listSize_)
		{
			QMutableMapIterator<unsigned int, QList<ToolParameter *>> it(paramList_);
			while (it.hasNext())
			{
				it.next();
				ToolParameter * p = it.value().first();
				if ((p->getToolId() == toolId) && (p->getParamToolId() == paramToolId))
				{
					it.value().append(params_.value(lastParamID));
					break;
				}
			}
			params_.insert(lastParamID, s);
		}
		else
		{
			params_.insert(generateParamId(), s);
		}

	}
	else
	{
		params_.insert(generateParamId(), s);
	}
	
}


/*template<class... Arg>
void
Tool::readParams(ToolParameter *s, Arg... arg)
{
	int i = sizeof ...(arg);
	readParams(arg...);
	paramList_.insert(generateParamId(), s);
} */

unsigned int
Tool::generateParamId()
{
	static unsigned int id = 0;

	return ++id;
}

int
Tool::getParamId(ToolParameter *s)
{
	QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_.constBegin();
	while (it != paramList_.constEnd())
	{
		QList<ToolParameter *>params = it.value();
		if (params.contains(s))
		{
			return it.key();
		}

		it++;
	}

	if (it == paramList_.constEnd())
	{
		return params_.key(s);
	}

	return -1;
}

/*ToolParameter *
Tool::getParam(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId)
{
	foreach(ToolParameter *param, paramList_.values())
	{
		if (param->getToolId() == toolId)
		{
			if (param->getParamToolId() == paramToolId)
			{
				return param;
			}
		}
	}

	return NULL; 
} */

bool
Tool::verify()
{
	QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_.constBegin();
	while (it != paramList_.constEnd())
	{
		foreach(ToolParameter *p, it.value())
		{
			if (!p->isValid())
			{
				if (!p->isList())
				{
					return false;
				}
				else if (getParamList(p->getListIndex()).size() == 0)
				{

					return false;
				}
			}
		}
		it++;
	}

	foreach(ToolParameter *p, params_.values())
	{
		if (!p->isValid())
		{
			return false;
		}
	}

	return true;
}


QList<ToolParameter *> 
Tool::getParamList(unsigned char listId)
{
	QList<ToolParameter *> list;
	QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_.constBegin();
	while (it != paramList_.constEnd())
	{
		QList<ToolParameter *>params = it.value();
		foreach(ToolParameter *param, params)
		{
			if (param->getListIndex() == listId)
			{
				if (param->isValid())
				{
					list.push_back(param);
				}
			}
		}
		it++;
	}

	return list; 
}

ToolParameter * 
Tool::getLastParam(unsigned char listId)
{
	if (params_.contains(listId))
	{
		return params_.value(listId);
	}
	QList<ToolParameter *> list = paramList_.value(listId);
	if (!list.empty())
	{
		return list.last();
	}

	return NULL;
}

ToolParameter *
Tool::getParam(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId)
{
	foreach(ToolParameter *param, params_)
	{
		if ((param->getToolId() == toolId) && (param->getParamToolId() == paramToolId))
		{
			return param;
		}
	}

	foreach(QList<ToolParameter *> list, paramList_)
	{
		ToolParameter *param = list.last();
		if ((param->getToolId() == toolId) && (param->getParamToolId() == paramToolId))
		{
			return param;
		}
	}

	return NULL;
}

int 
Tool::getObjectCount(const ODD::ToolId &toolId, const ODD::ToolId &paramToolId)
{
	int count = 0;
	foreach(ToolParameter *p, params_)
	{
		if ((p->getToolId() == toolId) && (p->getParamToolId() == paramToolId))
		{
			if (p->isValid())
			{
				count++;
			}
		}
	}

	foreach(QList<ToolParameter *>list, paramList_.values())
	{
		ToolParameter *p = list.first();
		if ((p->getToolId() == toolId) && (p->getParamToolId() == paramToolId))
		{
			count += list.size();
		}
	}

	return count;
}

int
Tool::getObjectCount(const ODD::ToolId &id)
{
	if (paramList_.contains(id))
	{
		return paramList_.value(id).size();
	}

	return 0; 
} 




