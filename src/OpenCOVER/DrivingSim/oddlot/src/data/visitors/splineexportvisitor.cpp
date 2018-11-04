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

#include "splineexportvisitor.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Utils //
//
#include "math.h"

// Qt //
//
#include <QFile>
#include <QMessageBox>
#include <QApplication>

SplineExportVisitor::SplineExportVisitor(const QString &dirName)
    : dirName_(dirName)
{
}

void
SplineExportVisitor::visit(RoadSystem *roadSystem)
{
    roadSystem->acceptForRoads(this);
}

void
SplineExportVisitor::visit(RSystemElementRoad *road)
{
    // Open file //
    //
    //	fileName = QString("export/spline_");
    QString fileName = dirName_;
    fileName.append("/spline_");
    fileName.append(road->getID().speakingName());
    fileName.append(".txt");

    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text))
    {
        qDebug("Cannot write file %s:\n%s.",
                fileName.toUtf8().constData(),
                file.errorString().toUtf8().constData());
        //		QMessageBox::warning(this, tr("ODD"),
        //			tr("Cannot write file %1:\n%2.")
        //			.arg(fileName)
        //			.arg(file.errorString()));
        return;
    }

    // Export //
    //
    QTextStream out(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);

    // Initialization //
    //
    double length = road->getLength();
    double pointsPerMeter = 1.0; // BAD: hard coded!
    int pointCount = int(ceil(length * pointsPerMeter)); // TODO curvature...
    if (pointCount <= 1)
    {
        pointCount = 2; // should be at least 2 to get a quad
    }
    QVector<QPointF> points(pointCount);
    double segmentLength = length / (pointCount - 1);

    // Print //
    //
    //	qDebug("Punkt X Y Z <- X <- Y <- Z X -> Y -> Z ->");
    out << "Punkt\tX\tY\tZ\t<- X\t<- Y\t<- Z\tX ->\tY ->\tZ ->"
        << "\n";
    for (int i = 0; i < pointCount; ++i)
    {
        double s = i * segmentLength; // [sStart, sEnd]
        points[i] = road->getGlobalPoint(s, 0.0); /*road->getMinWidth(s)*/
        //		qDebug() << i << "\t" << points[i].x() << "\t" << points[i].y() << "\t0\t0\t0\t0\t0\t0\t0";
        out << i << "\t" << points[i].x() << "\t" << points[i].y() << "\t0\t0\t0\t0\t0\t0\t0"
            << "\n";
    }

    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
}
