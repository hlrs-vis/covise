/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.04.2010
**
**************************************************************************/

#ifndef TRACKMOVEVALIDATOR_HPP
#define TRACKMOVEVALIDATOR_HPP

#include "src/data/acceptor.hpp"

#include <QPointF>

class TrackEditor;

/*! \brief This class checks if all TrackComponents can set a specific point/heading.
*/
class TrackMoveValidator : public Visitor
{

    //################//
    // STATIC         //
    //################//

public:
    enum State
    {
        STATE_STARTPOINT,
        STATE_STARTHEADING,
        STATE_ENDPOINT,
        STATE_ENDHEADING
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackMoveValidator();
    virtual ~TrackMoveValidator()
    { /* does nothing */
    }

    //  //
    //
    bool isValid() const
    {
        return isValid_;
    }
    void reset()
    {
        isValid_ = true;
    }

    // State //
    //
    void setState(TrackMoveValidator::State state);

    void setGlobalDeltaPos(const QPointF &dPos);
    void setGlobalHeading(const double heading);

    // Visitor Pattern //
    //
    virtual void visit(TrackSpiralArcSpiral *);
    virtual void visit(TrackElementLine *);
    virtual void visit(TrackElementArc *);
    virtual void visit(TrackElementSpiral *);
    virtual void visit(TrackElementPoly3 *);

private:
    //	TrackMoveValidator(); /* not allowed */
    TrackMoveValidator(const TrackMoveValidator &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    bool isValid_;

    // State //
    //
    TrackMoveValidator::State state_;

    QPointF dPos_;
    double heading_;
};

#endif // TRACKMOVEVALIDATOR_HPP
