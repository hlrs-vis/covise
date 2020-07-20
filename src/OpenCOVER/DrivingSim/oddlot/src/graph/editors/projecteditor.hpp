/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.03.2010
**
**************************************************************************/

#ifndef PROJECTEDITOR_HPP
#define PROJECTEDITOR_HPP

#include <QObject>

#include "src/util/odd.hpp"

class ProjectWidget;
class ProjectData;
class ProjectGraph;
class TopviewGraph;
class ToolParameter;
class ToolParameterSettings;
class ToolParameterSettingsApplyBox;
class Tool;

class ToolAction;
class MouseAction;
class KeyAction;

/** \brief MVC: Controller. Baseclass for all editors.
*
*
*/
class ProjectEditor : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~ProjectEditor();

    // Tool //
    //
    virtual void toolAction(ToolAction *toolAction);
    void setTool(ODD::ToolId id);
    ODD::ToolId getCurrentTool()
    {
        return currentTool_;
    }
	void setParameterTool(ODD::ToolId id);
	ODD::ToolId getCurrentParameterTool()
	{
		return currentParameterTool_;
	}
    bool isCurrentTool(ODD::ToolId toolId)
    {
        if (toolId == currentTool_)
            return true;
        else
            return false;
    }

    // Mouse & Key //
    //
    virtual void mouseAction(MouseAction *mouseAction);
    virtual void keyAction(KeyAction *keyAction);

    // Project, Data, Graph //
    //
    ProjectWidget *getProjectWidget() const
    {
        return projectWidget_;
    }
    ProjectData *getProjectData() const
    {
        return projectData_;
    }
    ProjectGraph *getProjectGraph() const;
    TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }

    // StatusBar //
    //
    void printStatusBarMsg(const QString &text, int milliseconds);

protected:
    virtual void init() = 0;
    virtual void kill() = 0;

	// ToolParameters //
	//
	void createToolParameterSettingsApplyBox(Tool *tool, const ODD::EditorId &editorID);
	void createToolParameterSettings(Tool *tool, const ODD::EditorId &editorID);
	void deleteToolParameterSettings();
	void generateToolParameterUI(Tool *tool);
	void updateToolParameterUI(ToolParameter *param);
	void delToolParameters();
	template<class T>
	void setToolValue(T *object, const QString &valueDisplayed);
	template<class T>
	void createToolParameters(T *object);
	template<class T>
	void removeToolParameters(T *object);

private:
    ProjectEditor(); /* not allowed */
    ProjectEditor(const ProjectEditor &); /* not allowed */
    ProjectEditor &operator=(const ProjectEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:
    void show();
    void hide();

	// Parameter Settings //
	//
	virtual void apply() = 0;
	virtual void reject();
	virtual void reset() = 0;

    //################//
    // PROPERTIES     //
    //################//
protected:

	// ToolParameters //
	//
	Tool *tool_;
	ToolParameterSettingsApplyBox *settingsApplyBox_;
	ToolParameterSettings *settings_;

private:
    // Project, Data, Graph //
    //
    ProjectWidget *projectWidget_;
    ProjectData *projectData_;
    TopviewGraph *topviewGraph_;
	MainWindow *mainWindow_;

    // Tool //
    //
    ODD::ToolId currentTool_;
	ODD::ToolId currentParameterTool_;

};

#endif // PROJECTEDITOR_HPP
