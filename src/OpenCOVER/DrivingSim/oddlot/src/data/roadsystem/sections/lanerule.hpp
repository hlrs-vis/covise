/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.02.2010
**
**************************************************************************/

#ifndef LANERULE_HPP
#define LANERULE_HPP

#include "src/data/dataelement.hpp"

#include <QString>

class LaneRule : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneRuleChange
    {
        CLR_ParentLaneChanged = 0x1,
        CLR_OffsetChanged = 0x2,
        CLR_ValueChanged = 0x4
    };

	static const QList<QString> KNOWNVALUES;


    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRule(double sOffset, const QString &value);
    virtual ~LaneRule();

    // Lane Functions //
    //
    Lane *getParentLane() const
    {
        return parentLane_;
    }
    void setParentLane(Lane *parentLane);

    double getSSectionStart() const
    {
        return sOffset_;
    }
    double getSSectionEnd() const;
    double getLength() const;

    // RoadMark Parameters //
    //
    double getSOffset() const
    {
        return sOffset_;
    }
    void setSOffset(double sOffset);

    QString getValue() const
    {
        return value_;
    }
    void setValue(const QString &value);


    int getRuleChanges() const
    {
        return ruleChanges_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();

    // Prototype Pattern //
    //
    LaneRule *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneRule(); /* not allowed */
    LaneRule(const LaneRule &); /* not allowed */
    LaneRule &operator=(const LaneRule &); /* not allowed */

    // Observer Pattern //
    //
    void addRuleChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int ruleChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    // RoadMark Properties //
    //
    double sOffset_;
	QString value_;
};

#endif // LANERULE_HPP
