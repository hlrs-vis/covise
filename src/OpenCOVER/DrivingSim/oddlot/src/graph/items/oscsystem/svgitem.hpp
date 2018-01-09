/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef SVGITEM_HPP
#define SVGITEM_HPP

#include <QtSvg/QGraphicsSvgItem>
#include <QtSvg/QSvgWidget>

class OSCItem;

class SVGItem : public QGraphicsSvgItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit SVGItem(OSCItem *oscItem, std::string file);
    virtual ~SVGItem();

    //################//
    // EVENTS         //
    //################//

public:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);


    //################//
    // PROPERTIES     //
    //################//

private:
	OSCItem *parentItem_;
	std::string file_;

	void init();
};

#endif // SVGITEM_HPP
