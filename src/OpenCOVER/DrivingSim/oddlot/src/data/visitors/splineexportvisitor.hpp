/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.06.2010
**
**************************************************************************/

#ifndef SPLINEEXPORTVISITOR_HPP
#define SPLINEEXPORTVISITOR_HPP

#include "src/data/visitor.hpp"

#include <QString>

class SplineExportVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SplineExportVisitor(const QString &dirName);
    //	virtual ~SplineExportVisitor(){};

    virtual void visit(RoadSystem *roadSystem);
    virtual void visit(RSystemElementRoad *road);

private:
    //	SplineExportVisitor(); /* not allowed */
    SplineExportVisitor(const SplineExportVisitor &); /* not allowed */
    SplineExportVisitor &operator=(const SplineExportVisitor &); /* not allowed */

private:
    QString dirName_;
};

#endif // SPLINEEXPORTVISITOR_HPP
