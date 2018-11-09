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

#ifndef TOOL_HPP
#define TOOL_HPP

#include <QObject>
#include <QButtonGroup>

class ToolManager;

class Tool : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Tool(ToolManager *toolManager);
    virtual ~Tool()
    { /* does nothing */
    }

protected:
private:
    Tool(); /* not allowed */
    Tool(const Tool &); /* not allowed */
    Tool &operator=(const Tool &); /* not allowed */

    //################//
    // SIGNALS        //
    //################//

    //signals:
    //	void						toolAction(ToolAction *);

    //################//
    // PROPERTIES     //
    //################//

protected:
    // ToolManager //
    //
    ToolManager *toolManager_;

private:
};

class ToolButtonGroup : public QButtonGroup
{
	Q_OBJECT

public:
	explicit ToolButtonGroup(ToolManager *toolManager);
	virtual ~ToolButtonGroup()
	{ /* does nothing */
	}

protected:
private:
	ToolButtonGroup(); /* not allowed */
	ToolButtonGroup(const ToolButtonGroup &); /* not allowed */
	ToolButtonGroup &operator=(const ToolButtonGroup &); /* not allowed */

protected:
	// ToolManager //
	//
	ToolManager *toolManager_;

private slots:
	void setButtonPressed(int i);
};


#endif // TOOL_HPP
