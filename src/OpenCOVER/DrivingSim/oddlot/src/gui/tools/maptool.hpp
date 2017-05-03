/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   6/11/2010
**
**************************************************************************/

#ifndef MAPTOOL_HPP
#define MAPTOOL_HPP

#include "tool.hpp"
#include "toolaction.hpp"

#include <QComboBox>
#include <QAction>
#include <QString>
#include <QDoubleSpinBox>

class MapTool : public Tool
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief Ids of the MapTools.
	*
	* This enum defines the Id of each tool.
	*/
    enum MapToolId
    {
        TMA_LOAD,
        TMA_DELETE,
        TMA_GOOGLE,
        TMA_LOCK,
        TMA_OPACITY,
        TMA_X,
        TMA_Y,
        TMA_WIDTH,
        TMA_HEIGHT
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit MapTool(ToolManager *toolManager);
    //	virtual ~MapTool(){ /* does nothing */ }

protected:
private:
    MapTool(); /* not allowed */
    MapTool(const MapTool &); /* not allowed */
    MapTool &operator=(const MapTool &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);

    //################//
    // SLOTS          //
    //################//

private slots:
    void activateProject(bool hasActive);
    void loadMap();
    void deleteMap();
    void loadGoogleMap();
    void lockMap(bool lock);
    void setOpacity(const QString &opacity);
    //	void						setX();
    //	void						setY();
    //	void						setWidth();
    //	void						setHeight();

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Actions //
    //
    QAction *loadMapAction_;
    QAction *deleteMapAction_;
    QAction *loadGoogleAction_;
    QAction *lockMapAction_;
    QComboBox *opacityComboBox_;
    //	QDoubleSpinBox *		xLineEdit_;
    //	QDoubleSpinBox *		yLineEdit_;
    //	QDoubleSpinBox *		widthLineEdit_;
    //	QDoubleSpinBox *		heightLineEdit_;

    //	bool						keepRatio_;
    //	double					lastX_;
    //	double					lastY_;
    //	double					lastWidth_;
    //	double					lastHeight_;

    bool active_;
};

class MapToolAction : public ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit MapToolAction(MapTool::MapToolId mapToolId);
    explicit MapToolAction(const QString &opacity);
    //	virtual ~MapToolAction(){ /* does nothing */ }

    MapTool::MapToolId getMapToolId() const
    {
        return mapToolId_;
    }
    QString getOpacity() const
    {
        return opacity_;
    }

    void setToggled(bool toggled);
    bool isToggled() const
    {
        return toggled_;
    }

    //	void						setX(double x);
    //	double					getX() const { return x_; }

    //	void						setY(double y);
    //	double					getY() const { return y_; }

    //	void						setWidth(double width, bool keepRatio);
    //	double					getWidth() const { return width_; }

    //	void						setHeight(double height, bool keepRatio);
    //	double					getHeight() const { return height_; }

    //	bool						isKeepRatio() const { return keepRatio_; }

protected:
private:
    MapToolAction(); /* not allowed */
    MapToolAction(const MapToolAction &); /* not allowed */
    MapToolAction &operator=(const MapToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    MapTool::MapToolId mapToolId_;
    QString opacity_;

    bool toggled_;

    //	double					x_;
    //	double					y_;

    //	double					width_;
    //	double					height_;
    //	bool						keepRatio_;
};

#endif // MAPTOOL_HPP
