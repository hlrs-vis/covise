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

namespace Ui
{
class oscObjectSettings;
}

namespace OpenScenario
{
class oscObjectBase;
}
//class SignalManager;

class OSCElement;

class oscObjectSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit oscObjectSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, OSCElement *element);
    virtual ~oscObjectSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

	// Initialize generic user interface //
	//
	void uiInit();

private:
    void updateProperties();

    //################//
    // SLOTS          //
    //################//

private slots:
    void onEditingFinished(const QString &name);

    //################//
    // PROPERTIES     //
    //################//

private:
	Ui::oscObjectSettings *ui;
    OpenScenario::oscObjectBase *object_;

    bool init_;

	QMap<QString, QWidget*> memberWidgets_;

    bool valueChanged_;
};

#endif // OSCOBJECTSETTINGS_HPP
