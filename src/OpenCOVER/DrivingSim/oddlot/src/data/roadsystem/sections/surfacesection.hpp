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

class SurfaceSection
{
	enum SurfaceOrientation
	{
		SSO_SAME = 0,
		SSO_OPPOSITE = 1
	};
	static SurfaceOrientation parseSurfaceOrientation(const QString &orientation);
	static QString parseSurfaceOrientationBack(SurfaceOrientation orientation);

	enum ApplicationMode
	{
		SSA_ATTACHED,
		SSA_ATTACHED0,
		SSA_GENUINE
	};
	static ApplicationMode parseApplicationMode(const QString &application);
	static QString parseApplicationModeBack(ApplicationMode application);

	enum SurfacePurpose
	{
		SSP_ELEVATION,
		SSP_FRICTION
	};
	static SurfacePurpose parseSurfacePurpose(const QString &purpose);
	static QString parseSurfacePurposeBack(SurfacePurpose purpose);

	struct CRGElement
	{
		CRGElement(QString file = "",
			QString sStart = "",
			QString sEnd = "",
			SurfaceOrientation orientation = SSO_SAME,
			ApplicationMode mode = SSA_GENUINE,
			SurfacePurpose purpose = SSP_ELEVATION,
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
			, purpose_(purpose)
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
		SurfaceOrientation orientation_;
		ApplicationMode mode_;
		SurfacePurpose purpose_;
		QString sOffset_;
		QString tOffset_;
		QString zOffset_;
		QString zScale_;
		QString hOffset_;
	};

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
				QString purpose,
                QString sOffset,
                QString tOffset,
                QString zOffset,
                QString zScale,
                QString hOffset)
    {
        crgs.append(CRGElement(file,
                               sStart,
                               sEnd,
                               parseSurfaceOrientation(orientation),
                               parseApplicationMode(mode),
							   parseSurfacePurpose(purpose),		
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
        return parseSurfaceOrientationBack(crgs[i].orientation_);
    }
    QString getMode(int i) const
    {
        return parseApplicationModeBack(crgs[i].mode_);
    }
	QString getPurpose(int i) const
	{
		return parseSurfacePurposeBack(crgs[i].purpose_);
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
