/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.06.2010
**
**************************************************************************/

#ifndef PEDESTRIAN_HPP
#define PEDESTRIAN_HPP

#include "src/data/dataelement.hpp"

class PedestrianGroup;

class Pedestrian : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum PedestrianChange
    {
        CVR_PedestrianGroupChanged = 0x1,
        CVR_IdChanged = 0x2,
        CVR_NameChanged = 0x4,
        CVR_GeometryChanged = 0x8,
        CVR_InitialStateChanged = 0x10,
        CVR_AnimationsChanged = 0x20
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Pedestrian(
        bool defaultPed,
        bool templatePed,
        const QString &id,
        const QString &name,
        const QString &templateId,
        const QString &rangeLOD,
        const QString &debugLvl,
        const QString &modelFile,
        const QString &scale,
        const QString &heading,
        const QString &startRoadId,
        const QString &startLane,
        const QString &startDir,
        const QString &startSOffset,
        const QString &startVOffset,
        const QString &startVel,
        const QString &startAcc,
        const QString &idleIdx,
        const QString &idleVel,
        const QString &slowIdx,
        const QString &slowVel,
        const QString &walkIdx,
        const QString &walkVel,
        const QString &jogIdx,
        const QString &jogVel,
        const QString &lookIdx,
        const QString &waveIdx);
    virtual ~Pedestrian();

    // Pedestrian //
    //
    bool isDefault()
    {
        return defaultPed_;
    }
    bool isTemplate()
    {
        return templatePed_;
    }
    QString getName() const
    {
        return name_;
    }
    QString getId() const
    {
        return id_;
    }
    QString getTemplateId() const
    {
        return templateId_;
    }
    QString getRangeLOD() const
    {
        return rangeLOD_;
    }
    QString getDebugLvl() const
    {
        return debugLvl_;
    }

    // Parameters //
    //
    QString getModelFile() const
    {
        return modelFile_;
    }
    QString getScale() const
    {
        return scale_;
    }
    QString getHeading() const
    {
        return heading_;
    }

    QString getStartRoadId() const
    {
        return startRoadId_;
    }
    QString getStartLane() const
    {
        return startLane_;
    }
    QString getStartDir() const
    {
        return startDir_;
    }
    QString getStartSOffset() const
    {
        return startSOffset_;
    }
    QString getStartVOffset() const
    {
        return startVOffset_;
    }
    QString getStartVel() const
    {
        return startVel_;
    }
    QString getStartAcc() const
    {
        return startAcc_;
    }

    QString getIdleIdx() const
    {
        return idleIdx_;
    }
    QString getIdleVel() const
    {
        return idleVel_;
    }
    QString getSlowIdx() const
    {
        return slowIdx_;
    }
    QString getSlowVel() const
    {
        return slowVel_;
    }
    QString getWalkIdx() const
    {
        return walkIdx_;
    }
    QString getWalkVel() const
    {
        return walkVel_;
    }
    QString getJogIdx() const
    {
        return jogIdx_;
    }
    QString getJogVel() const
    {
        return jogVel_;
    }
    QString getLookIdx() const
    {
        return lookIdx_;
    }
    QString getWaveIdx() const
    {
        return waveIdx_;
    }

    // Parent //
    //
    PedestrianGroup *getParentPedestrianGroup() const
    {
        return parentPedestrianGroup_;
    }
    void setParentPedestrianGroup(PedestrianGroup *pedestrianGroup);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPedestrianChanges() const
    {
        return pedestrianChanges_;
    }
    void addPedestrianChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    Pedestrian(); /* not allowed */
    Pedestrian(const Pedestrian &); /* not allowed */
    Pedestrian &operator=(const Pedestrian &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int pedestrianChanges_;

    // Parent //
    //
    PedestrianGroup *parentPedestrianGroup_;

    // Pedestrian //
    //
    bool defaultPed_;
    bool templatePed_;
    QString id_;
    QString name_;
    QString templateId_;
    QString rangeLOD_;
    QString debugLvl_;

    // Parameters //
    //
    QString modelFile_;
    QString scale_;
    QString heading_;

    QString startRoadId_;
    QString startLane_;
    QString startDir_;
    QString startSOffset_;
    QString startVOffset_;
    QString startVel_;
    QString startAcc_;

    QString idleIdx_;
    QString idleVel_;
    QString slowIdx_;
    QString slowVel_;
    QString walkIdx_;
    QString walkVel_;
    QString jogIdx_;
    QString jogVel_;
    QString lookIdx_;
    QString waveIdx_;
};

#endif // PEDESTRIAN_HPP
