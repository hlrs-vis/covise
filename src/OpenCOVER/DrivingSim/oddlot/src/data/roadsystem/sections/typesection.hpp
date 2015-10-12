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

#ifndef TYPESECTION_HPP
#define TYPESECTION_HPP

#include "roadsection.hpp"

#include <QString>

class SpeedRecord
{
public:
    SpeedRecord();
    SpeedRecord(QString &max, QString &unit);
    float maxSpeed;
};

class TypeSection : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum TypeSectionChange
    {
        CTS_TypeChange = 0x1
    };

    enum RoadType
    {
        RTP_NONE,
        RTP_UNKNOWN,
        RTP_RURAL,
        RTP_MOTORWAY,
        RTP_TOWN,
        RTP_LOWSPEED,
        RTP_PEDESTRIAN
    };
    static RoadType parseRoadType(const QString &type);
    static QString parseRoadTypeBack(TypeSection::RoadType type);

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TypeSection(double s, TypeSection::RoadType type);
    virtual ~TypeSection();

    // TypeSection //
    //
    TypeSection::RoadType getRoadType()
    {
        return type_;
    }
    void setRoadType(TypeSection::RoadType roadType);

    void setSpeedRecord(SpeedRecord *sr);
    SpeedRecord *getSpeedRecord();

    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
    virtual double getLength() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTypeSectionChanges() const
    {
        return typeSectionChanges_;
    }
    void addTypeSectionChanges(int changes);

    // Prototype Pattern //
    //
    TypeSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    TypeSection(); /* not allowed */
    TypeSection(const TypeSection &); /* not allowed */
    TypeSection &operator=(const TypeSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int typeSectionChanges_;

    // Type Properties //
    //
    TypeSection::RoadType type_;

    SpeedRecord *speedRecord;
};

#endif // TYPESECTION_HPP
