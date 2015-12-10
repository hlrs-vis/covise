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

#ifndef PROJECTWIDGET_HPP
#define PROJECTWIDGET_HPP

#include <QWidget>
#include <QMap>
#include <src/gui/osmimport.hpp>
#include "src/util/odd.hpp"

#include "src/gui/projectionsettings.hpp"
#include "src/gui/lodsettings.hpp"

class QAction;

class MainWindow;

class ProjectData;
class ChangeManager;

class ProjectEditor;

class TopviewGraph;
class ProfileGraph;

class ProjectTree;
class ProjectSettings;
class LODSettings;

class ToolAction;
class MouseAction;
class KeyAction;
class RoadSystem;
class RSystemElementRoad;

class TrackSpiralArcSpiral;

namespace OpenScenario
{
class OpenScenarioBase;
class oscObject;
}


/** \brief The ProjectWidget class is the main class for each project.
* It is a container for the Model-View-Controller classes.
*
* The ProjectWidget has the following responsibilities:
* \li Project Handling: Manage the (de)activation of this project.
* \li File Handling: Create, load, save and close files for this project.
* \li MVC: Create the Model (ProjectData), View (TopviewGraph) and Controller(s) (ProjectEditor).
* \li MVC: Manage the (de)activation of the Controller(s).
* \li Tool Routing: Route the tool events that come from the MainWindow to the Controller (ProjectEditor).
*
*/
class ProjectWidget : public QWidget
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectWidget(MainWindow *mainWindow);
    virtual ~ProjectWidget();

    // MainWindow //
    //
    MainWindow *getMainWindow() const
    {
        return mainWindow_;
    }

    // Project Handling //
    //
    QAction *getProjectMenuAction() const
    {
        return projectMenuAction_;
    }

    // File Handling //
    //
    void newFile();
    bool loadFile(const QString &fileName);
    bool loadTile(const QString &fileName);
    void setFile(const QString &fileName);
    bool save();
    bool saveAs();
    bool saveFile(const QString &fileName);
    bool exportSpline();
    bool importIntermapFile(const QString &fileName);
    bool importCSVFile(const QString &fileName);
    bool importCarMakerFile(const QString &fileName);
    bool maybeSave();

	// Add catalogs //
	//
	void addCatalogTree(const QString & type, OpenScenario::oscObject *object);
    
    RSystemElementRoad *addLineStrip(QString name = "");
    RSystemElementRoad *addLineStrip(QString name,int maxspeed, bool bridge, int numLanes, osmWay::wayType type);
    size_t getMaxLinearLength(size_t start);
    float getLinearError(size_t start, size_t len);

    size_t getMaxArcLength(size_t start, double startHeadingDeg);
    double maxEndHeading;
    float getArcError(size_t start, size_t len, TrackSpiralArcSpiral *curve);

    QString getFileName()
    {
        return fileName_;
    }
    QString getStrippedFileName()
    {
        return strippedFileName_;
    }

    // MVC //
    //
    ProjectData *getProjectData() const
    {
        return projectData_;
    }
    TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }
    ProjectEditor *getProjectEditor() const
    {
        return projectEditor_;
    }
    ProfileGraph *getProfileGraph() const
    {
        return profileGraph_;
    }
    ProfileGraph *getHeightGraph() const
    {
        return heightGraph_;
    }
    ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }
    ProjectSettings *getProjectSettings() const
    {
        return projectSettings_;
    }
    ProjectionSettings *getProjectionSettings() const
    {
        return projectionSettings;
    }
    LODSettings *getLODSettings() const
    {
        return lodSettings;
    }
	OpenScenario::OpenScenarioBase *getOpenScenarioBase()
	{
		return osdb_;
	}

    void setEditor(ODD::EditorId id);

    std::vector<double> XVector;
    std::vector<double> YVector;
    std::vector<double> ZVector;
    size_t numLineStrips;
    //################//
    // EVENTS         //
    //################//

protected:
    ProjectionSettings *projectionSettings;
    LODSettings *lodSettings;
    std::vector<double> SlopeVector;
    std::vector<double> SVector; // S on road for each segment
    std::vector<int> FeatVector;
    std::vector<int> VFOMVector;
    double SegLen(size_t i1, size_t i2); // return length of segment from i1 to i2
    RoadSystem *roadSystem;

    RSystemElementRoad *currentRoadPrototype_;
    RSystemElementRoad *testRoadPrototype_;
    size_t getMaxElevationLength(size_t start);

    // Project Handling //
    //
    void closeEvent(QCloseEvent *event);

//################//
// SIGNALS        //
//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:

    // Project Handling //
    //
    void setProjectActive(bool);
    void setProjectClean(bool);

    // Tools, Mouse & Key //
    //
    void toolAction(ToolAction *);
    void mouseAction(MouseAction *);
    void keyAction(KeyAction *);

    //################//
    // PROPERTIES     //
    //################//

private:
    // MainWindow //
    //
    MainWindow *mainWindow_; // linked

    // File Handling //
    //
    QAction *projectMenuAction_;

    QString fileName_;
    QString strippedFileName_;

    bool isUntitled_; // used by save vs. saveAs
    bool isModified_;

    // MVC //
    //
    ProjectData *projectData_; // model			// owned

    TopviewGraph *topviewGraph_; // view			// owned
    ProfileGraph *profileGraph_; // view			// owned
    ProfileGraph *heightGraph_; // view			// owned

    ProjectEditor *projectEditor_; // controller
    QMap<ODD::EditorId, ProjectEditor *> editors_; // owned

    ProjectTree *projectTree_; // view			// owned
    ProjectSettings *projectSettings_; // view			// owned

    // ChangeManager //
    //
    ChangeManager *changeManager_; // owned

	// OpenScenarioBase //
	//
	OpenScenario::OpenScenarioBase *osdb_;
};

#endif // PROJECTWIDGET_HPP
