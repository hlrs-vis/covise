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

#ifndef CONTROLLERSETTINGS_HPP
#define CONTROLLERSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class ControllerSettings;
}

class RSystemElementController;

class ControllerSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ControllerSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementController *controller);
    virtual ~ControllerSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateControlEntries();

    //################//
    // SLOTS          //
    //################//

private slots:
    void onEditingFinished();
    void onEditingFinished(int);
    void onEditingFinished(double);
    void onValueChanged();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ControllerSettings *ui;

    RSystemElementController *controller_;

    bool init_;

    bool valueChanged_;
};

#endif // CONTROLLERSETTINGS_HPP
