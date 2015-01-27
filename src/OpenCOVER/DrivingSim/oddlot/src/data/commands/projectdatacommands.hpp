/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROJECTDATACOMMANDS_HPP
#define PROJECTDATACOMMANDS_HPP

// 1800

#include "datacommand.hpp"

class ProjectData;

//#########################//
// SetProjectDimensionsCommand //
//#########################//

class SetProjectDimensionsCommand : public DataCommand
{
public:
    explicit SetProjectDimensionsCommand(ProjectData *projectData, double north, double south, double east, double west, DataCommand *parent = NULL);
    virtual ~SetProjectDimensionsCommand();

    virtual int id() const
    {
        return 0x1801;
    }

    virtual void undo();
    virtual void redo();

private:
    SetProjectDimensionsCommand(); /* not allowed */
    SetProjectDimensionsCommand(const SetProjectDimensionsCommand &); /* not allowed */
    SetProjectDimensionsCommand &operator=(const SetProjectDimensionsCommand &); /* not allowed */

private:
    ProjectData *projectData_; // linked

    double newNorth_;
    double newSouth_;
    double newEast_;
    double newWest_;

    double oldNorth_;
    double oldSouth_;
    double oldEast_;
    double oldWest_;
};

//#########################//
// SetProjectNameCommand //
//#########################//

class SetProjectNameCommand : public DataCommand
{
public:
    explicit SetProjectNameCommand(ProjectData *projectData, const QString &name, DataCommand *parent = NULL);
    virtual ~SetProjectNameCommand();

    virtual int id() const
    {
        return 0x1802;
    }

    virtual void undo();
    virtual void redo();

private:
    SetProjectNameCommand(); /* not allowed */
    SetProjectNameCommand(const SetProjectNameCommand &); /* not allowed */
    SetProjectNameCommand &operator=(const SetProjectNameCommand &); /* not allowed */

private:
    ProjectData *projectData_; // linked

    QString newName_;
    QString oldName_;
};

//#########################//
// SetProjectVersionCommand //
//#########################//

class SetProjectVersionCommand : public DataCommand
{
public:
    explicit SetProjectVersionCommand(ProjectData *projectData, double version, DataCommand *parent = NULL);
    virtual ~SetProjectVersionCommand();

    virtual int id() const
    {
        return 0x1804;
    }

    virtual void undo();
    virtual void redo();

private:
    SetProjectVersionCommand(); /* not allowed */
    SetProjectVersionCommand(const SetProjectVersionCommand &); /* not allowed */
    SetProjectVersionCommand &operator=(const SetProjectVersionCommand &); /* not allowed */

private:
    ProjectData *projectData_; // linked

    double newVersion_;
    double oldVersion_;
};

//#########################//
// SetProjectDateCommand //
//#########################//

class SetProjectDateCommand : public DataCommand
{
public:
    explicit SetProjectDateCommand(ProjectData *projectData, const QString &date, DataCommand *parent = NULL);
    virtual ~SetProjectDateCommand();

    virtual int id() const
    {
        return 0x1808;
    }

    virtual void undo();
    virtual void redo();

private:
    SetProjectDateCommand(); /* not allowed */
    SetProjectDateCommand(const SetProjectDateCommand &); /* not allowed */
    SetProjectDateCommand &operator=(const SetProjectDateCommand &); /* not allowed */

private:
    ProjectData *projectData_; // linked

    QString newDate_;
    QString oldDate_;
};

#endif // PROJECTDATACOMMANDS_HPP
