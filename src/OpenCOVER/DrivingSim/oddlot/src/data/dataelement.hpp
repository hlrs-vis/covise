/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#ifndef DATAELEMENT_HPP
#define DATAELEMENT_HPP

#include "acceptor.hpp"
#include "subject.hpp"

#include "../util/odd.hpp"

// Qt //
//
#include <QList>
#include <QDebug>
#include <QList>
#include <QMap>
#include <QMultiMap>
#include <QString>

class QUndoStack;
class ProjectData;

class DataElement : public Acceptor, public Subject
{

    //################//
    // STATIC         //
    //################//

public:
    enum DataElementChange
    {
        CDE_DataElementCreated = 0x1,
        CDE_DataElementDeleted = 0x2,
        CDE_DataElementAdded = 0x4,
        CDE_DataElementRemoved = 0x8,
        CDE_SelectionChange = 0x10, // 16
        CDE_ChildSelectionChange = 0x20,
        CDE_HidingChange = 0x40,
        CDE_ChildChange = 0x80,
        CDE_LinkedToProject = 0x100,
        CDE_UnlinkedFromProject = 0x200
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit DataElement();
    virtual ~DataElement();

    // ProjectData //
    //
    virtual ProjectData *getProjectData();
    virtual ChangeManager *getChangeManager();
    virtual QUndoStack *getUndoStack();

    // Parent //
    //
    DataElement *getParentElement() const
    {
        return parentElement_;
    }

    // Selection //
    //
    bool isElementSelected() const
    {
        return selected_;
    }
    void setElementSelected(bool selected);

    // Children //
    //
    bool isChildElementSelected() const
    {
        return childSelected_;
    }
    QList<DataElement *> getSelectedChildElements() const
    {
        return selectedChildElements_;
    }

    // Hiding //
    //
    bool isElementHidden() const
    {
        return hidden_;
    }
    void setElementHidden(bool hidden);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getDataElementChanges() const
    {
        return dataElementChanges_;
    }

protected:
    // Project //
    //
    void linkToProject(ProjectData *projectData);
    void unlinkFromProject();
    bool isLinkedToProject();

    // Parent //
    //
    void setParentElement(DataElement *parentElement);

private:
    //DataElement(); /* not allowed */
    DataElement(const DataElement &); /* not allowed */
    DataElement &operator=(const DataElement &); /* not allowed */

    // Children //
    //
    void addChild(DataElement *element);
    void delChild(DataElement *element);

    void addSelectedChild(DataElement *element);
    void delSelectedChild(DataElement *element);

    // Observer Pattern //
    //
    void addDataElementChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int dataElementChanges_;

    // Project //
    //
    ProjectData *projectData_;

    // Parent //
    //
    DataElement *parentElement_; // linked

    // Selection //
    //
    bool selected_;
    bool childSelected_;

    // Children //
    //
    QList<DataElement *> childElements_; // linked
    QList<DataElement *> selectedChildElements_; // linked

    // Hiding //
    //
    bool hidden_;
};

class UserData
{
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit UserData(const QString &code, const QString &value)
    {
        code_ = code;
        value_ = value;
    };
    virtual ~UserData()
    { /* does nothing */
    }

    QString getCode() const
    {
        return code_;
    }
    void setCode(const QString &code)
    {
        code_ = code;
    }

    QString getValue() const
    {
        return value_;
    }
    void setValue(const QString &value)
    {
        value_ = value;
    }

private:
    QString code_;
    QString value_;
};

#endif // DATAELEMENT_HPP
