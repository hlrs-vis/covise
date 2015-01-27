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

#ifndef RSYSTEMELEMENTCONTROLLER_HPP
#define RSYSTEMELEMENTCONTROLLER_HPP

#include "rsystemelement.hpp"

class ControlEntry
{
public:
    explicit ControlEntry(const QString &signalId, const QString &type)
    {
        signalId_ = signalId;
        type_ = type;
    };
    virtual ~ControlEntry(){ /* does nothing */ };

    void setSignalId(const QString &signalId)
    {
        signalId_ = signalId;
    }
    QString getSignalId() const
    {
        return signalId_;
    }

    void setType(const QString &type)
    {
        type_ = type;
    }
    QString getType() const
    {
        return type_;
    }

private:
    QString signalId_;
    QString type_;
};

class RSystemElementController : public RSystemElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RSystemElementController(const QString &name, const QString &id, int sequence, const QString &script, double cycleTime, const QList<ControlEntry *> &controlEntries);
    virtual ~RSystemElementController();

    // Prototype Pattern //
    //
    RSystemElementController *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

    QString getId() const
    {
        return id_;
    }
    void setId(const QString &id)
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

    int getSequence() const
    {
        return sequence_;
    }
    void setSequence(const int sequence)
    {
        sequence_ = sequence;
    }

    QString getScript() const
    {
        return script_;
    }
    void setScript(const QString &script)
    {
        script_ = script;
    }

    double getCycleTime() const
    {
        return cycleTime_;
    }
    void setCycleTime(const double cycleTime)
    {
        cycleTime_ = cycleTime;
    }

    QList<ControlEntry *> getControlEntries() const
    {
        return controlEntries_;
    }
    void setControlEntries(const QList<ControlEntry *> &controlEntries)
    {
        controlEntries_ = controlEntries;
    }

private:
    RSystemElementController(); /* not allowed */
    RSystemElementController(const RSystemElementController &); /* not allowed */
    RSystemElementController &operator=(const RSystemElementController &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    QString name_;
    QString id_;
    int sequence_;
    QString script_;
    double cycleTime_;

    QList<ControlEntry *> controlEntries_;
};

#endif // RSYSTEMELEMENTCONTROLLER_HPP
