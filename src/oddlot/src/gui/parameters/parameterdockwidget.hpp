/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef PARAMETERDOCKWIDGET_HPP
#define PARAMETERDOCKWIDGET_HPP

#include <QDockWidget>
#include <QFrame>

class MainWindow;
class ProjectEditor;

class QPushButton;
class QGroupBox;


class ParameterDockWidget : public QDockWidget
{
public:

	explicit ParameterDockWidget(const QString& title, MainWindow* parent = NULL);
	virtual ~ParameterDockWidget();

	void init();
	void setVisibility(bool, const QString& helpText, const QString& windowTitle);

	QFrame* getParameterBox()
	{
		return paramBox_;
	}
	QFrame* getParameterDialogBox()
	{
		return dialogBox_;
	}

	QGroupBox* getParamGroupBox()
	{
		return paramGroupBox_;
	}

	ProjectEditor* getProjectEditor()
	{
		return projectEditor_;
	}

protected:
	bool eventFilter(QObject* object, QEvent* event) override;

	//################//
   // EVENTS         //
   //################//

protected:
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;
	
	//################//
	// SIGNALS        //
	//################//
private slots:
	void setFocus(QWidget* oldWidget, QWidget* newWidget);

private:
	friend class DockFrame;
public:
	MainWindow* mainWindow_;
	ProjectEditor* projectEditor_;

	QFrame *paramBox_;
	QFrame *dialogBox_;
	QGroupBox* paramGroupBox_;
};




#endif // PARAMETERDOCKWIDGET_HPP
