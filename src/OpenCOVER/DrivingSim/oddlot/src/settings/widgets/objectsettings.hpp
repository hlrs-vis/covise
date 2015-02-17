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

#ifndef OBJECTSETTINGS_HPP
#define OBJECTSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class ObjectSettings;
}

class Object;
class SignalManager;
class ObjectContainer;

class ObjectSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Object *object);
    virtual ~ObjectSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateProperties(QString country, ObjectContainer *objectProperties);
    void addObjects();
    double objectT(double s, double t, double roadDistance);

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_objectComboBox_activated(int);
    void onEditingFinished();
    void onEditingFinished(int);
    void on_sSpinBox_editingFinished();
    void onValueChanged();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ObjectSettings *ui;

    SignalManager *objectManager_;

    Object *object_;

    bool init_;

    bool valueChanged_;
};

#endif // OBJECTSETTINGS_HPP
