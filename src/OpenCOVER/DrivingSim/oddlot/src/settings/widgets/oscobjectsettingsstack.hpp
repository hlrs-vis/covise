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

#ifndef OSCOBJECTSETTINGSSTACK_HPP
#define OSCOBJECTSETTINGSSTACK_HPP

#include "src/settings/settingselement.hpp"


class OSCElement;

class QStackedWidget;

class OSCObjectSettingsStack : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCObjectSettingsStack(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, OSCElement *element);
    virtual ~OSCObjectSettingsStack();

    // Observer Pattern //
    //
    virtual void updateObserver();

	// Initialize generic user interface //
	//
	void uiInit();

	int getStackSize();

	void addWidget(QWidget *widget);

private:

	//################//
	// SIGNALS        //
	//################//

signals:

    //################//
    // SLOTS          //
    //################//

private slots:
	void removeWidget();

    //################//
    // PROPERTIES     //
    //################//

private:
    bool init_;
	ProjectSettings *projectSettings_;
	OSCElement *element_;

	QStackedWidget *stack_;
	QVBoxLayout *objectBoxLayout_;
};

#endif // OSCOBJECTSETTINGSSTACK_HPP
