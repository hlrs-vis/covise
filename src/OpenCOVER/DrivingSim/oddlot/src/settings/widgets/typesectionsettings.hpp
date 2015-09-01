/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TYPESECTIONSETTINGS_HPP
#define TYPESECTIONSETTINGS_HPP

#include "src/settings/settingselement.hpp"
#include "src/data/roadsystem/sections/typesection.hpp"

class TypeSectionSettingsUI;

class TypeSectionSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//
public:
    explicit TypeSectionSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, TypeSection *typesection);
    virtual ~TypeSectionSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_roadTypeBox_activated(int);
    void on_maxSpeed_valueChanged(double);

    //################//
    // PROPERTIES     //
    //################//

private:
    TypeSectionSettingsUI *ui;

    TypeSection *typeSection_;

    bool init_;

    int lastIndex_;
};

#endif // TYPESECTIONSETTINGS_HPP