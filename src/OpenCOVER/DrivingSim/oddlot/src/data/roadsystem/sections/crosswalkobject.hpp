/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#ifndef CROSSWALKOBJECT_HPP
#define CROSSWALKOBJECT_HPP

#include "roadsection.hpp"
#include "src/data/roadsystem/odrID.hpp"

class Crosswalk : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum CrosswalkChange
    {
        CEL_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Crosswalk(const odrID &id, const QString &name, double s, double length);
    virtual ~Crosswalk()
    { /* does nothing */
    }

    // Crosswalk //
    //
    const odrID &getId() const
    {
        return id_;
    }
    void setId(const odrID &id)
    {
        id_ = id;
    }

    QString getName() const
    {
        return name_;
    }
    void setName(const QString &name)
    {
        name_ = name;
    }

    double getS() const
    {
        return s_;
    }
    void setS(const double s)
    {
        s_ = s;
    }

    double getLength() const
    {
        return length_;
    }
    void setLength(const double length)
    {
        length_ = length;
    }

    double getCrossProb() const
    {
        return crossProb_;
    }
    void setCrossProb(const double crossProb)
    {
        crossProb_ = crossProb;
        crossProbSet_ = true;
    }
    bool hasCrossProb() const
    {
        return crossProbSet_;
    }

    double getResetTime() const
    {
        return resetTime_;
    }
    void setResetTime(const double resetTime)
    {
        resetTime_ = resetTime;
        resetTimeSet_ = true;
    }
    bool hasResetTime() const
    {
        return resetTimeSet_;
    }

    virtual double getSEnd() const;

    QString getType() const
    {
        return type_;
    }
    void setType(const QString &type)
    {
        type_ = type;
        typeSet_ = true;
    }
    bool hasType() const
    {
        return typeSet_;
    }

    int getDebugLvl() const
    {
        return debugLvl_;
    }
    void setDebugLvl(const int debugLvl)
    {
        debugLvl_ = debugLvl;
        debugLvlSet_ = true;
    }
    bool hasDebugLvl() const
    {
        return debugLvlSet_;
    }

    int getFromLane() const
    {
        return fromLane_;
    }
    void setFromLane(const int fromLane)
    {
        fromLane_ = fromLane;
        fromSet_ = true;
    }
    bool hasFromLane() const
    {
        return fromSet_;
    }
    int getToLane() const
    {
        return toLane_;
    }
    void setToLane(const int toLane)
    {
        toLane_ = toLane;
        toSet_ = true;
    }
    bool hasToLane() const
    {
        return toSet_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getCrosswalkChanges() const
    {
        return crosswalkChanges_;
    }
    void addCrosswalkChanges(int changes);

    // Prototype Pattern //
    //
    Crosswalk *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Crosswalk(); /* not allowed */
    Crosswalk(const Crosswalk &); /* not allowed */
    Crosswalk &operator=(const Crosswalk &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    RSystemElementRoad *parentRoad_;

    // Crosswalk //
    //
    // Mandatory
    odrID id_;
    QString name_;
    double s_;
    double length_;
    // Optional
    double crossProb_;
    bool crossProbSet_;
    double resetTime_;
    bool resetTimeSet_;
    QString type_;
    bool typeSet_;
    int debugLvl_;
    bool debugLvlSet_;

    // Validity //
    //
    int fromLane_;
    bool fromSet_;
    int toLane_;
    bool toSet_;

    // Change flags //
    //
    int crosswalkChanges_;
};

#endif // CROSSWALKOBJECT_HPP
