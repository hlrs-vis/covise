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
#include "src/data/observer.hpp"

class Signal;

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

class RSystemElementController : public RSystemElement, public Observer
{
public:

    enum ControllerChange
    {
        CRC_ParameterChange = 0x1,
        CRC_EntryChange = 0x2
    };

    struct ControllerUserData
    {
        QString script;
        double cycleTime;
    };

    //################//
    // FUNCTIONS      //
    //################//

    explicit RSystemElementController(const QString &name, const QString &id, int sequence, const QString &script, double cycleTime, const QList<ControlEntry *> &controlEntries);
    explicit RSystemElementController(const QString &name, const QString &id, int sequence, ControllerUserData &controllerUserData, const QList<ControlEntry *> &controlEntries);
    virtual ~RSystemElementController();

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getControllerChanges() const
    {
        return controllerChanges_;
    }
    void addControllerChanges(int changes);
    virtual void updateObserver();

    // Prototype Pattern //
    //
    RSystemElementController *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

    int getSequence() const
    {
        return sequence_;
    }
    void setSequence(const int sequence)
    {
        sequence_ = sequence;
    }

    ControllerUserData getControllerUserData() const
    {
        return controllerUserData_;
    }

    void setControllerUserData(const ControllerUserData &controllerUserData)
    {
        controllerUserData_ = controllerUserData;
    }

    QString getScript() const
    {
        return controllerUserData_.script;
    }
    void setScript(const QString &script)
    {
        controllerUserData_.script = script;
    }

    double getCycleTime() const
    {
        return controllerUserData_.cycleTime;
    }
    void setCycleTime(const double cycleTime)
    {
        controllerUserData_.cycleTime = cycleTime;
    }

    QList<ControlEntry *> getControlEntries() const
    {
        return controlEntries_;
    }
    void setControlEntries(const QList<ControlEntry *> &controlEntries)
    {
        controlEntries_ = controlEntries;
    }

    void addControlEntry(ControlEntry *controlEntry, Signal *signal);
    bool delControlEntry(ControlEntry *controlEntry, Signal *signal);

    QMap<QString, Signal *> getSignals()
    {
        return signals_;
    }

private:
    RSystemElementController(); /* not allowed */
    RSystemElementController(const RSystemElementController &); /* not allowed */
    RSystemElementController &operator=(const RSystemElementController &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    int sequence_;
    ControllerUserData controllerUserData_;


    QList<ControlEntry *> controlEntries_;
    QMap<QString, Signal *> signals_; // Signals listed in the control entries

    // Change flags //
    //
    int controllerChanges_;
};

#endif // RSYSTEMELEMENTCONTROLLER_HPP
