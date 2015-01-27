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

#ifndef SURFACESECTION_HPP
#define SURFACESECTION_HPP

#include "roadsection.hpp"

#include <QString>
#include <QVector>

struct CRGElement
{
    CRGElement(QString file = "",
               QString sStart = "",
               QString sEnd = "",
               QString orientation = "",
               QString mode = "",
               QString sOffset = "",
               QString tOffset = "",
               QString zOffset = "",
               QString zScale = "",
               QString hOffset = "")
        : file_(file)
        , sStart_(sStart)
        , sEnd_(sEnd)
        , orientation_(orientation)
        , mode_(mode)
        , sOffset_(sOffset)
        , tOffset_(tOffset)
        , zOffset_(zOffset)
        , zScale_(zScale)
        , hOffset_(hOffset)
    {
    }

    QString file_;
    QString sStart_;
    QString sEnd_;
    QString orientation_;
    QString mode_;
    QString sOffset_;
    QString tOffset_;
    QString zOffset_;
    QString zScale_;
    QString hOffset_;
};

class SurfaceSection
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SurfaceSection();
    virtual ~SurfaceSection()
    { /* does nothing */
    }

    // SurfaceSection //
    //
    void addCRG(QString file,
                QString sStart,
                QString sEnd,
                QString orientation,
                QString mode,
                QString sOffset,
                QString tOffset,
                QString zOffset,
                QString zScale,
                QString hOffset)
    {
        crgs.append(CRGElement(file,
                               sStart,
                               sEnd,
                               orientation,
                               mode,
                               sOffset,
                               tOffset,
                               zOffset,
                               zScale,
                               hOffset));
    }

    int getNumCRG() const
    {
        return crgs.size();
    }

    QString getFile(int i) const
    {
        return crgs[i].file_;
    }
    QString getSStart(int i) const
    {
        return crgs[i].sStart_;
    }
    QString getSEnd(int i) const
    {
        return crgs[i].sEnd_;
    }
    QString getOrientation(int i) const
    {
        return crgs[i].orientation_;
    }
    QString getMode(int i) const
    {
        return crgs[i].mode_;
    }
    QString getSOffset(int i) const
    {
        return crgs[i].sOffset_;
    }
    QString getTOffset(int i) const
    {
        return crgs[i].tOffset_;
    }
    QString getZOffset(int i) const
    {
        return crgs[i].zOffset_;
    }
    QString getZScale(int i) const
    {
        return crgs[i].zScale_;
    }
    QString getHOffset(int i) const
    {
        return crgs[i].hOffset_;
    }

    SurfaceSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    SurfaceSection(const SurfaceSection &); /* not allowed */
    SurfaceSection &operator=(const SurfaceSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Type Properties //
    //
    QVector<CRGElement> crgs;
};

#endif // SURFACESECTION_HPP
