/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   01.06.2010
**
**************************************************************************/

#ifndef PEDESTRIANSYSTEM_HPP
#define PEDESTRIANSYSTEM_HPP

#include "src/data/dataelement.hpp"

#include <QStringList>

class PedestrianGroup;

class PedestrianSystem : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum PedestrianSystemChange
    {
        CVS_ProjectDataChanged = 0x1,
        CVS_PedestrianGroupsChanged = 0x2,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PedestrianSystem();
    virtual ~PedestrianSystem();

    // PedestrianGroups //
    //
    void addPedestrianGroup(PedestrianGroup *pedestrianGroup);
    QList<PedestrianGroup *> getPedestrianGroups() const
    {
        return pedestrianGroups_;
    }

    // IDs //
    //
    const QString getUniqueId(const QString &suggestion);

    // ProjectData //
    //
    ProjectData *getParentProjectData() const
    {
        return parentProjectData_;
    }
    void setParentProjectData(ProjectData *projectData);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPedestrianSystemChanges() const
    {
        return pedestrianSystemChanges_;
    }
    void addPedestrianSystemChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForPedestrianGroups(Visitor *visitor);

private:
    PedestrianSystem(const PedestrianSystem &); /* not allowed */
    PedestrianSystem &operator=(const PedestrianSystem &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int pedestrianSystemChanges_;

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    // PedestrianGroups //
    //
    QList<PedestrianGroup *> pedestrianGroups_; // owned

    // IDs //
    //
    QStringList ids_;
    int idCount_;
};

#endif // PEDESTRIANSYSTEM_HPP
