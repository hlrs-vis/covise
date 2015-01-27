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

#ifndef SIGNALSETTINGS_HPP
#define SIGNALSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class SignalSettings;
}

class Signal;
class SignalManager;
class SignalContainer;

class QDoubleSpinBox;

class SignalSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Signal *signal);
    virtual ~SignalSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateProperties(QString country, SignalContainer *signalProperties);
    void addSignals();
    double signalT(double s, double t, double roadDistance);
    void enableCrossingParams(bool value);

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_signalComboBox_activated(int);
    void onEditingFinished();
    void onEditingFinished(int);
    void on_sSpinBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::SignalSettings *ui;

    SignalManager *signalManager_;

    Signal *signal_;

    bool init_;
};

#endif // SIGNALSETTINGS_HPP
