/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.01.2010
**
**************************************************************************/

#include "projectwidget.hpp"

#include "src/globalDefines.hpp"

#include "src/mainwindow.hpp"

#include "src/util/odd.hpp"
// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/changemanager.hpp"
#include "src/data/prototypemanager.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/tilesystem/tilesystem.hpp"
#include "src/data/tilesystem/tile.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/visitors/splineexportvisitor.hpp"

#include "src/data/vehiclesystem/vehiclesystem.hpp"
#include "src/data/pedestriansystem/pedestriansystem.hpp"
#include "src/data/scenerysystem/scenerysystem.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/editors/projecteditor.hpp"

#include "src/graph/profilegraph.hpp"

// Editor //
//
#include "src/graph/editors/roadlinkeditor.hpp"
#include "src/graph/editors/roadtypeeditor.hpp"
#include "src/graph/editors/trackeditor.hpp"
#include "src/graph/editors/elevationeditor.hpp"
#include "src/graph/editors/superelevationeditor.hpp"
#include "src/graph/editors/crossfalleditor.hpp"
#include "src/graph/editors/laneeditor.hpp"
#include "src/graph/editors/junctioneditor.hpp"
#include "src/graph/editors/signaleditor.hpp"

// Tree //
//
#include "src/tree/projecttree.hpp"

// Settings //
//
#include "src/settings/projectsettings.hpp"

// Tools, Mouse & Key //
//
#include "tools/toolaction.hpp"
#include "tools/toolmanager.hpp"
#include "tools/selectiontool.hpp"
#include "mouseaction.hpp"
#include "keyaction.hpp"

// I/O //
//
#include "src/io/domparser.hpp"
#include "src/io/domwriter.hpp"

// Qt //
//
#include <QtGui>
#include <QUndoGroup>
#include <QFileDialog>
#include <QDomDocument>
#include <QGridLayout>
#include <QSplitter>

#include <QMessageBox>
#include <QAction>
#include <QApplication>

/** \brief Main Contructor. Use only this one.
*
* The constructor creates the Model, View and Controller. It flags
* the project as untitled. The QAction for the project menu is created.
*
* \todo Restructuring when table views are used.
*
*/
ProjectWidget::ProjectWidget(MainWindow *mainWindow)
    : QWidget(mainWindow)
    , mainWindow_(mainWindow)
    , fileName_("")
    , strippedFileName_("")
    , isUntitled_(true)
    , isModified_(true)
    , projectData_(NULL)
    , topviewGraph_(NULL)
    , projectEditor_(NULL)
    , changeManager_(NULL)
{
    // Layout //
    //
    QGridLayout *layout = new QGridLayout();
    setLayout(layout);
    QSplitter *splitter = new QSplitter(this);
    splitter->setOrientation(Qt::Vertical);
    layout->addWidget(splitter);

    // Project Handling //
    //
    projectMenuAction_ = new QAction(this); // text will be set later
    projectMenuAction_->setCheckable(true);

    // File Handling //
    //
    setAttribute(Qt::WA_DeleteOnClose); // free memory after closing

    // UndoStack //
    //
    QUndoStack *undoStack = new QUndoStack(mainWindow_->getUndoGroup());
    undoStack->setUndoLimit(100); // TODO: SETTINGS, not hardcoded

    // ChangeManager //
    //
    changeManager_ = new ChangeManager(this);
    connect(undoStack, SIGNAL(indexChanged(int)), changeManager_, SLOT(notifyObservers())); // Changing the index of the UndoStack triggers the notification of the observers!

    // MODEL //
    //
    projectData_ = new ProjectData(this, undoStack, changeManager_, 1, 2, "Untitled", 1.0, QDateTime::currentDateTime().toString("ddd MMM hh:mm:ss yyyy"), 10000.0, -10000.0, 10000.0, -10000.0);

    connect(projectData_->getUndoStack(), SIGNAL(cleanChanged(bool)), this, SLOT(setProjectClean(bool)));

    projectData_->setRoadSystem(new RoadSystem());
    projectData_->setTileSystem(new TileSystem());
    projectData_->setVehicleSystem(new VehicleSystem());
    projectData_->setPedestrianSystem(new PedestrianSystem());
    projectData_->setScenerySystem(new ScenerySystem());

    // VIEW: Graph //
    //
    topviewGraph_ = new TopviewGraph(this, projectData_);
    splitter->addWidget(topviewGraph_);
    splitter->setStretchFactor(0, 3);

    // The ChangeManager triggers the view's garbage disposal.
    connect(projectData_->getChangeManager(), SIGNAL(notificationDone()), topviewGraph_, SLOT(garbageDisposal()));

    // Routes the tool, mouse and key events to the project widget.
    connect(topviewGraph_, SIGNAL(toolActionSignal(ToolAction *)), this, SLOT(toolAction(ToolAction *)));
    connect(topviewGraph_, SIGNAL(mouseActionSignal(MouseAction *)), this, SLOT(mouseAction(MouseAction *)));
    connect(topviewGraph_, SIGNAL(keyActionSignal(KeyAction *)), this, SLOT(keyAction(KeyAction *)));

    // VIEW: ProfileGraph //
    //
    profileGraph_ = new ProfileGraph(this, projectData_);
    profileGraph_->hide();
    splitter->addWidget(profileGraph_);
    splitter->setStretchFactor(1, 1);

    // The ChangeManager triggers the view's garbage disposal.
    connect(projectData_->getChangeManager(), SIGNAL(notificationDone()), profileGraph_, SLOT(garbageDisposal()));

    // VIEW: HeightGraph //
    //
    heightGraph_ = new ProfileGraph(this, projectData_, 20.0);
    heightGraph_->hide();
    heightGraph_->getScene()->doDeselect(false);
    splitter->addWidget(heightGraph_);
    splitter->setStretchFactor(1, 1);

    // The ChangeManager triggers the view's garbage disposal.
    connect(projectData_->getChangeManager(), SIGNAL(notificationDone()), heightGraph_, SLOT(garbageDisposal()));

    // CONTROLLER //
    //
    editors_.insert(ODD::ERL, new RoadLinkEditor(this, projectData_, topviewGraph_));
    editors_.insert(ODD::ERT, new RoadTypeEditor(this, projectData_, topviewGraph_));
    editors_.insert(ODD::ETE, new TrackEditor(this, projectData_, topviewGraph_));
    editors_.insert(ODD::EEL, new ElevationEditor(this, projectData_, topviewGraph_, profileGraph_));
    editors_.insert(ODD::ESE, new SuperelevationEditor(this, projectData_, topviewGraph_, profileGraph_));
    editors_.insert(ODD::ECF, new CrossfallEditor(this, projectData_, topviewGraph_, profileGraph_));
    editors_.insert(ODD::ELN, new LaneEditor(this, projectData_, topviewGraph_, heightGraph_));
    editors_.insert(ODD::EJE, new JunctionEditor(this, projectData_, topviewGraph_));
    editors_.insert(ODD::ESG, new SignalEditor(this, projectData_, topviewGraph_));

    // VIEW: Tree //
    //
    projectTree_ = new ProjectTree(this, projectData_);

    // The ChangeManager triggers the view's garbage disposal.
    connect(projectData_->getChangeManager(), SIGNAL(notificationDone()), projectTree_, SLOT(garbageDisposal()));

    // VIEW: Settings //
    //
    projectSettings_ = new ProjectSettings(this, projectData_);

    connect(projectData_->getChangeManager(), SIGNAL(notificationDone()), projectSettings_, SLOT(garbageDisposal()));

    projectionSettings = ProjectionSettings::instance();
    lodSettings = LODSettings::instance();

    currentRoadPrototype_ = new RSystemElementRoad("prototype", "prototype", "-1");

    QList<PrototypeContainer<RSystemElementRoad *> *> roadTypePrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype);
    QList<PrototypeContainer<RSystemElementRoad *> *> laneSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype);
    QList<PrototypeContainer<RSystemElementRoad *> *> superelevationPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype);
    QList<PrototypeContainer<RSystemElementRoad *> *> crossfallPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype);
    currentRoadPrototype_->superposePrototype(roadTypePrototypes.first()->getPrototype());
    currentRoadPrototype_->superposePrototype(laneSectionPrototypes.first()->getPrototype());
    currentRoadPrototype_->superposePrototype(superelevationPrototypes.first()->getPrototype());
    currentRoadPrototype_->superposePrototype(crossfallPrototypes.first()->getPrototype());

    testRoadPrototype_ = new RSystemElementRoad("prototype", "prototype", "-1");

    testRoadPrototype_->superposePrototype(roadTypePrototypes.first()->getPrototype());
    testRoadPrototype_->superposePrototype(laneSectionPrototypes.at(3)->getPrototype());
    testRoadPrototype_->superposePrototype(superelevationPrototypes.first()->getPrototype());
    testRoadPrototype_->superposePrototype(crossfallPrototypes.first()->getPrototype());
}

/*!
* \todo check if up to date
*/
ProjectWidget::~ProjectWidget()
{
    delete projectSettings_;
    delete heightGraph_;
    foreach (ProjectEditor *editor, editors_)
    {
        delete editor;
    }
    delete topviewGraph_;
    delete profileGraph_;

    delete projectTree_;

    delete projectData_;
    delete changeManager_;
}

//################//
// METHODS        //
//################//

void
ProjectWidget::setEditor(ODD::EditorId id)
{
    QMap<ODD::EditorId, ProjectEditor *>::const_iterator it = editors_.find(id);
    if (it != editors_.end())
    {
        // No Change if not changing //
        //
        if (projectEditor_ == it.value())
        {
            projectEditor_->show();
            return;
        }

        topviewGraph_->preEditorChange();

        // Hide last one //
        //
        if (projectEditor_)
        {
            projectEditor_->hide();
            //projectData_->getChangeManager()->unregisterAll(); // clear Subject-Observer list
            // This should be unnecessary if every observer detaches itself. It is even wrong,
            // if there are objects that are independent of the editor (e.g. maps).
        }

        // Show new one //
        //
        projectEditor_ = it.value();
        projectEditor_->show();

        // ProfileGraph //
        //
        if ((id == ODD::EEL)
            || (id == ODD::ESE)
            || (id == ODD::ECF))
        {
            profileGraph_->show();
        }
        else
        {
            profileGraph_->hide();
        }

        // HeightGraph //
        //
        if ((id == ODD::ELN))
        {
            heightGraph_->show();
        }
        else
        {
            heightGraph_->hide();
        }

        topviewGraph_->postEditorChange();
    }
    else
    {
        // Hide last one //
        //
        if (projectEditor_)
        {
            projectEditor_->hide();
            //			projectData_->getChangeManager()->unregisterAll(); // clear Subject-Observer list
        }

        // Warning //
        //
        projectEditor_ = NULL;
        qDebug("WARNING 1003111729! ProjectWidget::mainEditorChanged() Editor unknown");
    }
}

/** \brief Creates a new untitled project.
*
* This function creates a unique "untitledX.odd" file and sets
* the window title and project menu entry tile accordingly.
* Flags the project as untitled.
*/
void
ProjectWidget::newFile()
{
    // Create a unique name by counting up numbers.
    static int documentNumber = 0;
    ++documentNumber;
    fileName_ = tr("untitled%1.%2").arg(documentNumber).arg(FILEEXTENSIONSTR);
    strippedFileName_ = fileName_;

    // Set name in window title and project menu.
    setWindowTitle(strippedFileName_ + "[*]"); // [*] is the place for unsaved-marker
    projectMenuAction_->setText(strippedFileName_);

    // Create a Tile
    Tile *tile = new Tile("Tile0", "0");
    projectData_->getTileSystem()->addTile(tile);
    projectData_->getTileSystem()->setCurrentTile(tile);

    // Mark this file as untitled and modified.
    isUntitled_ = true;
    setProjectClean(false);

    return;
}

/** \brief Opens and reads in the specified file and starts the parser.
*
*	\todo read strategy (xodr, native, etc)
*/
bool
ProjectWidget::loadFile(const QString &fileName)
{
    // Print //
    //
    qDebug("Loading file: " + fileName.toUtf8());

    // Open file //
    //
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
                                                  .arg(fileName)
                                                  .arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }

    // Parse file //
    //

    // TODO: read strategy (xodr, native, etc)
    QApplication::setOverrideCursor(Qt::WaitCursor);
    DomParser *parser = new DomParser(projectData_);
    bool success = parser->parseXODR(&file);
    topviewGraph_->updateSceneSize();
    delete parser;
    // TODO

    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();

    // Check for success //
    //
    if (!success)
        return false;

    // Set file //
    //
    setFile(fileName);

    return true;
}
/**! \brief
*
*/
bool
ProjectWidget::loadTile(const QString &fileName)
{
    qDebug("Loading Tile from file: " + fileName.toUtf8());

    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
                                                  .arg(fileName)
                                                  .arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);
    DomParser *parser = new DomParser(projectData_);
    bool success = parser->parseXODR(&file);
    topviewGraph_->updateSceneSize();
    delete parser;

    QApplication::restoreOverrideCursor();
    file.close();

    if (!success)
        return false;

    setFile(fileName);

    return true;
}

float ProjectWidget::getLinearError(size_t start, size_t len)
{
    double sx = XVector[start];
    double sy = YVector[start];
    double dx = XVector[start + len - 1] - sx;
    double dy = YVector[start + len - 1] - sy;
    float maxDist = 0;
    float lenDx = sqrt(dx * dx + dy * dy);

    for (size_t i = start + 1; i < start + len - 1; i++) // do not need to check the first and last point both distances are 0
    {
        float dist;
        double px = XVector[i];
        double py = YVector[i];
        double bx = px - sx;
        double by = py - sy;
        dist = fabs(((dx * by) - (dy * bx)) / lenDx); // abstand Punkt Gerade
        if (dist > maxDist)
        {
            maxDist = dist;
        }
    }
    return maxDist;
}

size_t ProjectWidget::getMaxLinearLength(size_t start)
{
    size_t maxLen = XVector.size() - start;
    for (size_t i = 3; i <= maxLen; i++)
    {
        if (getLinearError(start, i) > ImportSettings::instance()->LinearError)
            return i - 1;
    }
    return maxLen;
}

float ProjectWidget::getArcError(size_t start, size_t len, TrackSpiralArcSpiral *curve)
{
    float maxDist = 0;
    RSystemElementRoad *tmpRoad = new RSystemElementRoad("testRoad", "rID_test", "");
    tmpRoad->addTrackComponent(curve);
    double clen = curve->getLength();
    for (size_t i = start + 1; i < start + len - 1; i++) // do not need to check the first and last point both distances are 0
    {

        QPointF testPos(XVector[i], YVector[i]);
        double s = tmpRoad->getSFromGlobalPoint(testPos, 0.0, clen);
        QPointF RoadPos = tmpRoad->getGlobalPoint(s);
        QPointF diff = testPos - RoadPos;
        float dist = sqrt(diff.x() * diff.x() + diff.y() * diff.y());
        if (dist > maxDist)
        {
            maxDist = dist;
        }
    }

    delete tmpRoad;
    return maxDist;
}

size_t ProjectWidget::getMaxArcLength(size_t start, double startHeadingDeg)
{
    size_t maxLen = XVector.size() - start;
    size_t maxtestedLen = 0;
    int numErrors = 0;
    maxEndHeading = 0;

    QPointF startPos(XVector[start], YVector[start]);
    if (maxLen > 500)
        maxLen = 500;

    for (size_t i = 2; i <= maxLen; i++)
    {
        double dxe = 0;
        double dye = 0;

        if (start + i == XVector.size())
        {
            dxe = XVector[start + i - 1] - XVector[start + i - 2];
            dye = YVector[start + i - 1] - YVector[start + i - 2];
        }
        else
        {
            dxe = XVector[start + i] - XVector[start + i - 1];
            dye = YVector[start + i] - YVector[start + i - 1];
        }
        QPointF endPos(XVector[start + i - 1], YVector[start + i - 1]);
        double endHeadingDeg = atan2(dye, dxe) * RAD_TO_DEG;
        double HeadingDiff = endHeadingDeg - startHeadingDeg; // TODO curvature
        if ((fabs(HeadingDiff) > 0.001) && (endPos != startPos))
        {

            TrackSpiralArcSpiral *curve = new TrackSpiralArcSpiral(startPos, endPos, startHeadingDeg, endHeadingDeg, 0.5);
            if (curve->validParameters())
            {

                if (getArcError(start, i, curve) > ImportSettings::instance()->CurveError)
                {
                    //delete curve;
                    //return i-1;

                    numErrors++;
                    if (numErrors > 20)
                    {
                        return maxtestedLen;
                    }
                }
                else
                {
                    numErrors = 0;
                    maxtestedLen = i;
                    maxEndHeading = endHeadingDeg;
                }
            }
            else
            {
                if (start + i == XVector.size())
                {
                    dxe = XVector[start + i - 1] - XVector[start + i - 2];
                    dye = YVector[start + i - 1] - YVector[start + i - 2];
                }
                else
                {
                    dxe = XVector[start + i] - XVector[start + i - 1];
                    dye = YVector[start + i] - YVector[start + i - 1];
                }
                QPointF endPos(XVector[start + i - 1], YVector[start + i - 1]);
                double endHeadingDeg = atan2(dye, dxe) * RAD_TO_DEG;
                // if we can't create a curve, we might try a different end heading ( the heading of the last segment )

                HeadingDiff = endHeadingDeg - startHeadingDeg; // TODO curvature
                if ((fabs(HeadingDiff) > 0.001) && (endPos != startPos))
                {
                    curve = new TrackSpiralArcSpiral(startPos, endPos, startHeadingDeg, endHeadingDeg, 0.5);
                    if (curve->validParameters())
                    {
                        if (getArcError(start, i, curve) > ImportSettings::instance()->CurveError)
                        {
                            //delete curve;
                            //return i-1;

                            numErrors++;
                            if (numErrors > 20)
                            {
                                return maxtestedLen;
                            }
                        }
                        else
                        {
                            numErrors = 0;
                            maxtestedLen = i;
                            maxEndHeading = endHeadingDeg;
                        }
                    }
                }
            }
        }
    }
    return maxtestedLen;
}
void ProjectWidget::addLineStrip(QString name)
{
    roadSystem = projectData_->getRoadSystem();
    QString number = QString::number(numLineStrips);

    RSystemElementRoad *road = new RSystemElementRoad("", name, "");

    SVector.reserve(XVector.size());
    SVector.resize(XVector.size());

    double dxs = 0;
    double dys = 0;
    dxs = XVector[1] - XVector[0];
    dys = YVector[1] - YVector[0];
    double startHeadingDeg = atan2(dys, dxs) * RAD_TO_DEG;
    bool lastLine = false;
    for (size_t i = 0; i < XVector.size() - 1; i++)
    {
        SVector[i] = road->getLength();
        size_t len = getMaxLinearLength(i);
        size_t arcLen = getMaxArcLength(i, startHeadingDeg);
        if (arcLen > len || (lastLine && (arcLen > 0)))
        {
            double dxe = 0;
            double dye = 0;
            if (i + arcLen == XVector.size())
            {
                dxe = XVector[i + arcLen - 1] - XVector[i + arcLen - 2];
                dye = YVector[i + arcLen - 1] - YVector[i + arcLen - 2];
            }
            else
            {
                dxe = XVector[i + arcLen] - XVector[i + arcLen - 1];
                dye = YVector[i + arcLen] - YVector[i + arcLen - 1];
            }
            QPointF startPos(XVector[i], YVector[i]);
            QPointF endPos(XVector[i + arcLen - 1], YVector[i + arcLen - 1]);
            double endHeadingDeg = atan2(dye, dxe) * RAD_TO_DEG;
            endHeadingDeg = maxEndHeading;

            TrackSpiralArcSpiral *curve = new TrackSpiralArcSpiral(startPos, endPos, startHeadingDeg, endHeadingDeg, 0.5);
            if (curve->validParameters())
            {
                curve->setSStart(road->getLength());
                road->addTrackComponent(curve);
            }
            else
            {
                fprintf(stderr, "Can't create curve which worked before...\n");
            }
            for (size_t j = 1; j < arcLen; j++)
            {
                SVector[i + j] = SVector[i + j - 1] + SegLen(i + j, i + j - 1);
            }
            i += arcLen - 2;
            lastLine = false;
            startHeadingDeg = endHeadingDeg;
        }
        else
        {
            double dxs = XVector[i + len - 1] - XVector[i];
            double dys = YVector[i + len - 1] - YVector[i];
            startHeadingDeg = atan2(dys, dxs) * RAD_TO_DEG;
            double length = sqrt(dxs * dxs + dys * dys);
            TrackElementLine *line = new TrackElementLine(XVector[i], YVector[i], atan2(dys, dxs) * RAD_TO_DEG, road->getLength(), length);
            road->addTrackComponent(line);
            for (int j = 1; j < len; j++)
            {
                SVector[i + j] = SVector[i] + SegLen(i, i + j);
            }
            i += len - 2;
            lastLine = true;
        }
    }
    // Calculate Slopes //
    SlopeVector.reserve(XVector.size());
    SlopeVector.resize(XVector.size());
    SlopeVector[0] = (ZVector[1] - ZVector[0]) / SegLen(1, 0);
    for (int i = 1; i < XVector.size() - 1; ++i)
    {
        SlopeVector[i] = 0.5 * (ZVector[i] - ZVector[i - 1]) / SegLen(i, i - 1) + 0.5 * (ZVector[i + 1] - ZVector[i]) / SegLen(i + 1, i);
    }
    SlopeVector[XVector.size() - 1] = (ZVector[XVector.size() - 1] - ZVector[XVector.size() - 2]) / SegLen(XVector.size() - 1, XVector.size() - 2);

    QMap<double, ElevationSection *> newSections;

    for (size_t i = 0; i < XVector.size() - 1; i++) // set elevation
    {
        size_t len = getMaxElevationLength(i);
        {
            double length = SVector[i + len - 1] - SVector[i];
            double length2 = length * length;
            double a = ZVector[i];
            double b = SlopeVector[i];
            double d = (SlopeVector[i + len - 1] + b - 2.0 * ZVector[i + len - 1] / length + 2.0 * a / length) / length2;
            double c = (ZVector[i + len - 1] - d * length2 * length - b * length - a) / length2;

            ElevationSection *currentSection = new ElevationSection(SVector[i], a, b, c, d);
            newSections.insert(SVector[i], currentSection);

            i += len - 2;
        }
    }

    if (newSections.isEmpty())
    {
        ElevationSection *section = new ElevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
        newSections.insert(0.0, section);
    }

    road->setElevationSections(newSections);
    road->superposePrototype(currentRoadPrototype_);

    roadSystem->addRoad(road); // This may change the ID!
#ifdef tesselatedROAD
    // add a testroad
    road = new RSystemElementRoad("tr_" + number, "trID_" + number, "");
    for (int i = 0; i < XVector.size() - 1; i++)
    {
        double dx = XVector[i + 1] - XVector[i];
        double dy = YVector[i + 1] - YVector[i];
        double length = sqrt(dx * dx + dy * dy);
        TrackElementLine *line = new TrackElementLine(XVector[i], YVector[i], atan2(dy, dx) * RAD_TO_DEG, road->getLength(), length);
        road->addTrackComponent(line);
    }
    road->superposePrototype(testRoadPrototype_);
    roadSystem->addRoad(road); // This may change the ID!
#endif

    numLineStrips++;
}

size_t ProjectWidget::getMaxElevationLength(size_t start)
{
    size_t maxLen = XVector.size() - start;

    double a = ZVector[start];
    double b = SlopeVector[start];
    for (size_t i = 2; i < maxLen; i++)
    {

        double length = SVector[start + i] - SVector[start];
        if (length == 0)
        {
            return i;
        }
        double length2 = length * length;
        double d = (SlopeVector[start + i] + b - 2.0 * ZVector[start + i] / length + 2.0 * a / length) / length2;
        double c = (ZVector[start + i] - d * length2 * length - b * length - a) / length2;

        ElevationSection *currentSection = new ElevationSection(0.0, a, b, c, d);
        for (size_t j = 1; j < i - 2; j++) // ton't test start and end error should be 0
        {
            double cLen = SVector[start + j] - SVector[start];
            if ((ZVector[start + j] - currentSection->getElevation(cLen)) > 0.1)
                return i;
        }
    }
    return maxLen;
}
double ProjectWidget::SegLen(size_t i1, size_t i2) // return length of segment from i1 to i2
{
    double dx, dy;
    dx = XVector[i2] - XVector[i1];
    dy = YVector[i2] - YVector[i1];
    return sqrt((dx * dx) + (dy * dy));
}

/** \brief imports an intermap Road file.
*
*/
bool
ProjectWidget::importIntermapFile(const QString &fileName)
{
    numLineStrips = 0;
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
                                                  .arg(fileName)
                                                  .arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }
    QTextStream in(&file);

    QString line = in.readLine();
    while (!line.isNull())
    {
        if (line.length() != 0)
        {
            double latitude, longitude, alt;
            int feature, vfom;
#ifdef WIN32
            int num = sscanf_s(line.toUtf8(), "%lf, %lf, %lf, %d, %d", &latitude, &longitude, &alt, &feature, &vfom);
#else
            int num = sscanf(line.toUtf8(), "%lf, %lf, %lf, %d, %d", &latitude, &longitude, &alt, &feature, &vfom);
#endif
            if (num == 5) // we read everything
            {
                double x = longitude * DEG_TO_RAD;
                double y = latitude * DEG_TO_RAD;
                double z = alt;
                projectionSettings->transform(x, y, z);
                XVector.push_back(x);
                YVector.push_back(y);
                ZVector.push_back(z);
                FeatVector.push_back(feature);
                VFOMVector.push_back(vfom);
            }
        }
        else if (XVector.size() > 1)
        {
            // add line segment
            addLineStrip();
            XVector.clear();
            YVector.clear();
            ZVector.clear();
            FeatVector.clear();
            VFOMVector.clear();
        }
        line = in.readLine();
    }
    // add last line segment
    if (XVector.size() > 1)
    {
        // add line segment
        addLineStrip();
        XVector.clear();
        YVector.clear();
        ZVector.clear();
        FeatVector.clear();
        VFOMVector.clear();
    }

    topviewGraph_->updateSceneSize();
    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
    return true;
}

/** \brief imports a csv Road file.
*
*/
bool
ProjectWidget::importCSVFile(const QString &fileName)
{
    numLineStrips = 0;
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
                                                  .arg(fileName)
                                                  .arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }
    QTextStream in(&file);

    QString line = in.readLine();
    while (!line.isNull())
    {
        if (line.length() != 0)
        {
            double latitude, longitude, alt;

            line.replace(',', '.');
#ifdef WIN32
            int num = sscanf_s(line.toUtf8(), "%lf; %lf; %lf", &longitude, &latitude, &alt);
#else
            int num = sscanf(line.toUtf8(), "%lf; %lf; %lf", &longitude, &latitude, &alt);
#endif
            if (num == 3) // we read everything
            {
                double x = longitude * DEG_TO_RAD;
                double y = latitude * DEG_TO_RAD;
                double z = alt;
                projectionSettings->transform(x, y, z);

                XVector.push_back(x);
                YVector.push_back(y);
                ZVector.push_back(z);
            }
        }
        else if (XVector.size() > 1)
        {
            // add line segment
            addLineStrip();
            XVector.clear();
            YVector.clear();
            ZVector.clear();
        }
        line = in.readLine();
    }
    // add last line segment
    if (XVector.size() > 1)
    {
        // add line segment
        addLineStrip();
        XVector.clear();
        YVector.clear();
        ZVector.clear();
    }

    topviewGraph_->updateSceneSize();
    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
    return true;
}


/** \brief imports a CarMaker Road file.
*
*/
bool
ProjectWidget::importCarMakerFile(const QString &fileName)
{
    numLineStrips = 0;
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
                                                  .arg(fileName)
                                                  .arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }
    QTextStream in(&file);
    
    QString line = in.readLine();
    while (!line.isNull())
    {
        if (line.length() != 0)
        {
            if(line[0]==':' && line[2]=='x')
            {
                break;
            }
            if(line[0]=='#')
            {
                // TODO read Traffic signs
            }
        }
        line = in.readLine();
    }

    line = in.readLine();
    while (!line.isNull())
    {
        if (line.length() != 0)
        {
            double x, y, z;

            line.replace(',', '.');
#ifdef WIN32
            int num = sscanf_s(line.toUtf8(), "%lf %lf %lf", &x, &y, &z);
#else
            int num = sscanf(line.toUtf8(), "%lf %lf %lf", &x, &y, &z);
#endif
            if (num == 3) // we read everything
            {

                XVector.push_back(x);
                YVector.push_back(y);
                ZVector.push_back(z);
            }
        }
        else if (XVector.size() > 1)
        {
            // add line segment
            addLineStrip();
            XVector.clear();
            YVector.clear();
            ZVector.clear();
        }
        line = in.readLine();
    }
    // add last line segment
    if (XVector.size() > 1)
    {
        // add line segment
        addLineStrip();
        XVector.clear();
        YVector.clear();
        ZVector.clear();
    }

    topviewGraph_->updateSceneSize();
    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
    return true;
}

/** \brief Called when a file has been loaded or saved.
*
* Sets the file name, window title and project menu entry.
* Flags the project as titled.
*/
void
ProjectWidget::setFile(const QString &fileName)
{
    // Set file name //
    //
    fileName_ = QFileInfo(fileName).canonicalFilePath();
    strippedFileName_ = QFileInfo(fileName_).fileName();

    isUntitled_ = false;

    projectData_->getUndoStack()->setClean();

    // If the undo stack is clean already this will not be called,
    // so call it just in case.
    setProjectClean(true);

    // Window title //
    //
    setWindowTitle(strippedFileName_ + "[*]"); // [*] is the place for unsaved-marker

    // Menu //
    //
    projectMenuAction_->setText(strippedFileName_);
}

/** \brief Chooses between saveAs() and saveFile(filename).
*
* Checks if the file is untitled and calls saveAs if so.
* Triggers saveFile(filename) directly otherwise.
*/
bool
ProjectWidget::save()
{
    if (isUntitled_)
        return saveAs();
    else
        return saveFile(fileName_);
}

/** \brief Lets the user choose a file name and calls saveFile(filename).
*
* Filename, window title, etc are set later by saveFile(filename).
*/
bool
ProjectWidget::saveAs()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), fileName_);
    if (fileName.isEmpty())
        return false;
    return saveFile(fileName);
}

/** \brief Opens the specified file and writes out the text stream.
*
*	\todo write strategy (xodr, native, etc)
*/
bool
ProjectWidget::saveFile(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("ODD"),
                             tr("Cannot write file %1:\n%2.")
                                 .arg(fileName)
                                 .arg(file.errorString()));
        return false;
    }

    // Export //
    //
    QTextStream out(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);

    // TODO: write strategy (xodr, native, etc)
    const int indentSize = 3;
    DomWriter *domWriter = new DomWriter(projectData_);
    domWriter->runToTheHills();
    domWriter->getDomDocument()->save(out, indentSize);
    // TODO

    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();

    // Set file //
    //
    setFile(fileName);

    return true;
}

/*! \brief Exports the track chord line.
*
*/
bool
ProjectWidget::exportSpline()
{
    // Directory //
    //
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setOption(QFileDialog::ShowDirsOnly, true);

    QString dirName = dialog.getExistingDirectory(this, tr("Save in Directory"), fileName_);
    if (dirName.isEmpty())
    {
        return false;
    }

    // Export //
    //
    SplineExportVisitor *splineExportVisitor = new SplineExportVisitor(dirName);
    projectData_->getRoadSystem()->accept(splineExportVisitor);

    return true;
}

/** \brief Checks if the project is modified and should be saved before closing.
*
* Called when the user wants to close a project. Queries the user
* whether the file should be saved when there are unsaved modifications.
* Returns true if it is OK to close the project.
*/
bool
ProjectWidget::maybeSave()
{
    // Project modified //
    //
    if (isModified_)
    {
        // Ask user //
        QMessageBox::StandardButton ret = QMessageBox::warning(this, tr("ODD"),
                                                               tr("'%1' has been modified.\nDo you want to save your changes?")
                                                                   .arg(strippedFileName_),
                                                               QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

        // Try to save //
        //
        if (ret == QMessageBox::Save)
            return save();

        // Cancel //
        //
        else if (ret == QMessageBox::Cancel)
            return false;
    }

    // Project not modified //
    //
    return true;
}

//################//
// SLOTS          //
//################//

/*! \brief Called when this project becomes the active one.
*
*/
void
ProjectWidget::setProjectActive(bool active)
{
    if (active)
    {
        mainWindow_->setProjectTree(projectTree_);
        mainWindow_->setProjectSettings(projectSettings_);
    }

    projectData_->projectActivated(active); // Undo, etc

    projectTree_->projectActivated(active);
    projectSettings_->projectActivated(active);
}

/*! \brief Called when this project has been modified.
*
* \li Sets isModified_ which is used by save-on-quit-if-modified.
* \li Marks the window with a *.
*/
void
ProjectWidget::setProjectClean(bool clean)
{
    isModified_ = !clean;
    setWindowModified(isModified_);
}

/*! \brief Passes a ToolAction to the Editor and Graph.
*
*/
void
ProjectWidget::toolAction(ToolAction *toolAction)
{
    // Change Editor if necessary //
    //
    ODD::EditorId id = toolAction->getEditorId();
    if (id != ODD::ENO_EDITOR)
    {
        setEditor(id);
    }

    // Pass to Editor/Graph //
    if (projectEditor_)
    {
        projectEditor_->toolAction(toolAction);
    }
    topviewGraph_->toolAction(toolAction);
    profileGraph_->toolAction(toolAction);
}

/*! \brief Passes a MouseAction to the Editor and Graph.
*
* The editor can intercept the action. So the event will not be
* passed back to the graph (for selection management, etc).
*
*/
void
ProjectWidget::mouseAction(MouseAction *mouseAction)
{
    // Editor //
    //
    if (projectEditor_)
    {
        projectEditor_->mouseAction(mouseAction);
    }
    else
    {
        qDebug("TODO: ProjectWidget, mouseAction, NEED EDITOR AT STARTUP!");
    }

    // Graph //
    //
    if (!mouseAction->isIntercepted())
    {
        topviewGraph_->mouseAction(mouseAction);
    }
}

/*! \brief Passes a KeyAction to the Editor and Graph.
*
*/
void
ProjectWidget::keyAction(KeyAction *keyAction)
{
    projectEditor_->keyAction(keyAction);
    topviewGraph_->keyAction(keyAction);
    mainWindow_->getToolManager()->getSelectionTool()->keyAction(keyAction);
}

//################//
// EVENTS         //
//################//

/*! \brief Close Event. Checks first whether file should be saved.
*
* The close event will be ignored if the user cancels the query
* or the saving fails.
*/
void
ProjectWidget::closeEvent(QCloseEvent *event)
{
    if (maybeSave())
    {
        event->accept();
        qDebug("Project: Closed");
    }
    else
    {
        event->ignore();
    }
}
