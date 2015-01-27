/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/18/2010
**
**************************************************************************/

#ifndef SCENERYTESSELATION_HPP
#define SCENERYTESSELATION_HPP

#include "src/data/dataelement.hpp"

class ScenerySystem;

class SceneryTesselation : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum SceneryTesselationChange
    {
        CST_ScenerySystemChanged = 0x1,
        CST_TesselateRoadsChanged = 0x2,
        CST_TesselatePathsChanged = 0x3
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SceneryTesselation();
    virtual ~SceneryTesselation();

    // SceneryTesselation //
    //
    bool getTesselateRoads() const
    {
        return tesselateRoads_;
    }
    void setTesselateRoads(bool tesselate);

    bool getTesselatePaths() const
    {
        return tesselatePaths_;
    }
    void setTesselatePaths(bool tesselate);

    // ScenerySystem //
    //
    ScenerySystem *getParentScenerySystem() const
    {
        return parentScenerySystem_;
    }
    void setParentScenerySystem(ScenerySystem *scenerySystem);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getSceneryTesselationChanges() const
    {
        return sceneryTesselationChanges_;
    }
    void addSceneryTesselationChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    //	SceneryTesselation(); /* not allowed */
    SceneryTesselation(const SceneryTesselation &); /* not allowed */
    SceneryTesselation &operator=(const SceneryTesselation &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // ScenerySystem //
    //
    ScenerySystem *parentScenerySystem_;

    // Change flags //
    //
    int sceneryTesselationChanges_;

    // SceneryTesselation //
    //
    bool tesselateRoads_;
    bool tesselatePaths_;
};

#endif // SCENERYTESSELATION_HPP
