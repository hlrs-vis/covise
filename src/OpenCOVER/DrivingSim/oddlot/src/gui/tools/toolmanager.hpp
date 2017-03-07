/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#ifndef TOOLMANAGER_HPP
#define TOOLMANAGER_HPP

#include <QObject>

#include <QList>

class ToolAction;
class ToolWidget;
class ZoomTool;

class PrototypeManager;
class SelectionTool;
class OpenScenarioEditorTool;

class QToolBox;
class QMenu;
class QToolBar;
class QTabWidget;

class ToolManager : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ToolManager(PrototypeManager *prototypeManager, QObject *parent);
    virtual ~ToolManager()
    { /* does nothing */
    }

    // Menu //
    //
    QList<QMenu *> getMenus()
    {
        return menus_;
    }
    void addMenu(QMenu *menu);

    // Tool Box //
    //
    QToolBox *getToolBox()
    {
        return toolBox_;
    }
    
    QTabWidget *getRibbonWidget()
    {
        return ribbon_;
    }
    void setRibbon(QTabWidget *r)
    {
        ribbon_=r;
    }
    void addToolBoxWidget(ToolWidget *widget, const QString &title);
    void addRibbonWidget(ToolWidget *widget, const QString &title);

    void resendCurrentTool();

    SelectionTool *getSelectionTool()
    {
        return selectionTool_;
    }

	ZoomTool *getZoomTool()
	{
		return zoomTool_;
	}

    void enableOSCEditorToolButton(bool state);
	void setPushButtonColor(const QString &name, QColor color);

protected:
private:
    ToolManager(); /* not allowed */
    ToolManager(const ToolManager &); /* not allowed */
    ToolManager &operator=(const ToolManager &); /* not allowed */

    void initTools();

    //################//
    // SLOTS          //
    //################//

public slots:
    void toolActionSlot(ToolAction *);

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    PrototypeManager *prototypeManager_;

    // Last ToolAction //
    //
    ToolAction *lastToolAction_;

    // Menu //
    //
    QList<QMenu *> menus_;

    // Tool Box //
    //
    QToolBox *toolBox_;

    // Ribbon based Toolbox
    //
    QTabWidget *ribbon_;

    // Selection Tool
    //
    SelectionTool *selectionTool_;

	// ZoomTool //
	//
	ZoomTool *zoomTool_;

    // OpenScenarioEditorTool //
    //
    OpenScenarioEditorTool *oscEditorTool_;

};

#endif // TOOLMANAGER_HPP
