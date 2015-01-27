/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/11/2010
**
**************************************************************************/

#ifndef JUNCTIONSETTINGS_HPP
#define JUNCTIONSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class JunctionSettings;
}

class RSystemElementJunction;

class JunctionSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementJunction *junction);
    virtual ~JunctionSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateConnections();

    //################//
    // SLOTS          //
    //################//

private slots:

    void on_nameButton_released();
    void on_cleanConnectionsButton_released();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::JunctionSettings *ui;

    RSystemElementJunction *junction_;
};

#endif // JUNCTIONSETTINGS_HPP
