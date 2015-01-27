/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#ifndef ROADTYPEEDITOR_HPP
#define ROADTYPEEDITOR_HPP

#include "projecteditor.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

class ProjectData;
class TopviewGraph;

class RoadTypeRoadSystemItem;

class SectionHandle;

class RoadTypeEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadTypeEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~RoadTypeEditor();

    // Handle //
    //
    SectionHandle *getInsertSectionHandle() const;

    // Tool //
    //
    virtual void toolAction(ToolAction *);

    // RoadType //
    //
    TypeSection::RoadType getCurrentRoadType() const
    {
        return currentRoadType_;
    }
    void setCurrentRoadType(TypeSection::RoadType roadType);

protected:
    virtual void init();
    virtual void kill();

private:
    RoadTypeEditor(); /* not allowed */
    RoadTypeEditor(const RoadTypeEditor &); /* not allowed */
    RoadTypeEditor &operator=(const RoadTypeEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadSystem //
    //
    RoadTypeRoadSystemItem *roadSystemItem_;

    // Handle //
    //
    SectionHandle *insertSectionHandle_;

    // RoadType //
    //
    TypeSection::RoadType currentRoadType_;
};

#endif // ROADTYPEEDITOR_HPP
