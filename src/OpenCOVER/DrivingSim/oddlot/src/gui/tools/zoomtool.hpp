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

#ifndef ZOOMTOOL_HPP
#define ZOOMTOOL_HPP

#include "tool.hpp"
#include "toolaction.hpp"

#include <QComboBox>
#include <QAction>
#include <QDoubleSpinBox>

class ZoomTool : public Tool
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief Ids of the zoom tools.
	*
	* This enum defines the Id of each tool.
	*/
    enum ZoomToolId
    {
        TZM_UNKNOWN,
        TZM_ZOOMTO,
        TZM_ZOOMIN,
        TZM_ZOOMOUT,
        TZM_ZOOMBOX,
        TZM_VIEW_SELECTED,
        TZM_RULERS,
        TZM_SELECT_INVERSE,
        TZM_HIDE_SELECTED,
        TZM_HIDE_SELECTED_ROADS,
        TZM_HIDE_DESELECTED,
        TZM_UNHIDE_ALL
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ZoomTool(ToolManager *toolManager);
    virtual ~ZoomTool()
    { /* does nothing */
    }

protected:
private:
    ZoomTool(); /* not allowed */
    ZoomTool(const ZoomTool &); /* not allowed */
    ZoomTool &operator=(const ZoomTool &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);

    //################//
    // SLOTS          //
    //################//
public slots:
	void zoomIn();
    void zoomOut();


private slots:
    void activateProject(bool);

    void zoomTo(const QString &zoomFactor);
    void zoomBox();
    void viewSelected();

    void activateRulers(bool);

    void hideSelected();
    void hideSelectedRoads();
    void hideDeselected();
    void unhideAll();
    void selectInverse();

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Actions //
    //
    QComboBox *zoomComboBox_;
    QAction *zoomInAction_;
    QAction *zoomOutAction_;
    //	QAction *		zoomBoxAction_;
    QAction *viewSelectedAction_;
    QAction *rulerAction_;

    QAction *selectInverseAction_;
    QAction *hideSelectedAction_;
    QAction *hideSelectedRoadsAction_;
    QAction *hideDeselectedAction_;
    QAction *unhideAllAction_;
};

class ZoomToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ZoomToolAction(ZoomTool::ZoomToolId zoomToolId, bool toggled = true);
    explicit ZoomToolAction(const QString &zoomFactor);
    virtual ~ZoomToolAction()
    { /* does nothing */
    }

    ZoomTool::ZoomToolId getZoomToolId() const
    {
        return zoomToolId_;
    }
    QString getZoomFactor() const
    {
        return zoomFactor_;
    }

    bool isToggled() const
    {
        return toggled_;
    }

protected:
private:
    ZoomToolAction(); /* not allowed */
    ZoomToolAction(const ZoomToolAction &); /* not allowed */
    ZoomToolAction &operator=(const ZoomToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    ZoomTool::ZoomToolId zoomToolId_;
    QString zoomFactor_;

    bool toggled_;
};

#endif // ZOOMTOOL_HPP
