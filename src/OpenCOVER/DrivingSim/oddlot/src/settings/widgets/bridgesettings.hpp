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

#ifndef BRIDGESETTINGS_HPP
#define BRIDGESETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class BridgeSettings;
}

class Bridge;
class SignalManager;

class BridgeSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit BridgeSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Bridge *bridge);
    virtual ~BridgeSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();

    //################//
    // SLOTS          //
    //################//

private slots:
    void onEditingFinished();
    void on_sSpinBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::BridgeSettings *ui;

    Bridge *bridge_;

    bool init_;
};

#endif // BRIDGEETTINGS_HPP
