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

#ifndef OSCOBJECTSETTINGS_HPP
#define OSCOBJECTSETTINGS_HPP

#include "src/settings/settingselement.hpp"

#include "oscMemberValue.h"


namespace Ui
{
class OSCObjectSettings;
}

namespace OpenScenario
{
class oscObjectBase;
}
//class SignalManager;

class OSCElement;
class OSCBase;
class OpenScenarioEditorToolAction;
class OSCObjectSettingsStack;


class OSCObjectSettings: public QWidget
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCObjectSettings(ProjectSettings *projectSettings, OSCObjectSettingsStack *parent, OSCElement *element);
    virtual ~OSCObjectSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

	// Initialize generic user interface //
	//
	void uiInit();

private:
    void updateProperties();

	//################//
	// SIGNALS        //
	//################//

signals:

    //################//
    // SLOTS          //
    //################//

private slots:
    void onEditingFinished(QString name);
	void onPushButtonPressed(QString name);
	void onValueChanged();

    //################//
    // PROPERTIES     //
    //################//

private:
	Ui::OSCObjectSettings *ui;
	ProjectSettings *projectSettings_;
	OSCObjectSettingsStack *parentStack_;

    OpenScenario::oscObjectBase *object_;
	OSCElement *element_;
	OSCBase *base_;

    bool init_;

	QMap<QString, QWidget*> memberWidgets_;

    bool valueChanged_;

};

#endif // OSCOBJECTSETTINGS_HPP
