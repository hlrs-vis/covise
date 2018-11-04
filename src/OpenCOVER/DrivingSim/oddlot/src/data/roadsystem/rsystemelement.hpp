/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef RSYSTEMELEMENT_HPP
#define RSYSTEMELEMENT_HPP

#include "../dataelement.hpp"
#include "odrID.hpp"

// Qt //
//
#include <QString>

class RSystemElement : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum DRoadSystemElementType
    {
        DRE_None,
        DRE_Road,
        DRE_Controller,
        DRE_Junction,
        DRE_Fiddleyard,
        DRE_PedFiddleyard,
        DRE_Signal,
        DRE_Object,
		DRE_JunctionGroup
    };

    enum RSystemElementChange
    {
        CRE_IdChange = 0x1,
        CRE_NameChange = 0x2,
        CRE_ParentRoadSystemChange = 0x4
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RSystemElement(const QString &name, const odrID &id, DRoadSystemElementType elementType);
    virtual ~RSystemElement();

    // RoadSystem //
    //
    RoadSystem *getRoadSystem() const
    {
        return parentRoadSystem_;
    }
    void setRoadSystem(RoadSystem *parentRoadSystem);

    // RSystemElement //
    //
    const QString &getName() const
    {
        return name_;
    }
    const odrID &getID() const
    {
        return id_;
    }
    QString getIdName() const;

    void setName(const QString &name);
    void setID(const odrID &id);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRSystemElementChanges() const
    {
        return rSystemElementChanges_;
    }
    void addRSystemElementChanges(int changes);

private:
    RSystemElement(); /* not allowed */
    RSystemElement(const RSystemElement &); /* not allowed */
    RSystemElement &operator=(const RSystemElement &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Type //
    //
    RSystemElement::DRoadSystemElementType elementType_;

    // RoadSystem //
    //
    RoadSystem *parentRoadSystem_; // linked

    // RSystemElement //
    //
    QString name_; // name of the element
    odrID id_; // unique ID within database

    // Observer Pattern //
    //
    int rSystemElementChanges_;
};

#endif // RSYSTEMELEMENT_HPP
