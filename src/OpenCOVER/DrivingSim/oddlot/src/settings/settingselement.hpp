/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#ifndef SETTINGSELEMENT_HPP
#define SETTINGSELEMENT_HPP

#include <QWidget>
#include "src/data/observer.hpp"

// Data //
//
#include "src/data/projectdata.hpp"

class DataElement;

// Settings //
//
#include "projectsettings.hpp"

class SettingsElement : public QWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SettingsElement(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, DataElement *dataElement);
    virtual ~SettingsElement();

    // Garbage //
    //
    void registerForDeletion();
    virtual void notifyDeletion(); // to be implemented by subclasses
    bool isInGarbage() const
    {
        return isInGarbage_;
    }

    // Data //
    //
    ProjectData *getProjectData() const;

    // Settings //
    //
    ProjectSettings *getProjectSettings() const;

    // DataElement //
    //
    DataElement *getDataElement() const
    {
        return dataElement_;
    }

    // ParentSettingsElement //
    //
    SettingsElement *getParentSettingsElement() const
    {
        return parentSettingsElement_;
    }
    //	void						setParentSettingsElement(SettingsElement * parentSettingsElement);

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    SettingsElement(); /* not allowed */
    SettingsElement(const SettingsElement &); /* not allowed */
    SettingsElement &operator=(const SettingsElement &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    //	void						hideSettingsElement();

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

private:
    // ProjectSettings //
    //
    ProjectSettings *projectSettings_;

    // ParentSettingsElement //
    //
    SettingsElement *parentSettingsElement_;

    // DataElement //
    //
    DataElement *dataElement_;

    // Garbage //
    //
    bool isInGarbage_;
};

#endif // SETTINGSELEMENT_HPP
