/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#ifndef GRAPHVIEW_HPP
#define GRAPHVIEW_HPP

#include <QGraphicsView>
#include <QRubberBand>

class TopviewGraph;
class GraphScene;
class ZoomTool;

class Ruler;
class ScenerySystemItem;

class ToolAction;

class GraphView : public QGraphicsView
{
    Q_OBJECT

public:
    enum BoundingBoxStatusId // BoundingBox Button Status
    {
        BBOff, // BoundingBox selection mode not selected
        BBActive, // BoundingBox active
        BBPressed // BoundingBox Button pressed, but not active, e.g. during pan
    };

    enum CircleStatusId // Circle Button Status
    {
        CircleOff, // Circle selection mode not selected
        CircleActive, // Circle active
        CirclePressed // Circle Button pressed, but not active, e.g. during pan
    };

public:
    explicit GraphView(GraphScene *graphScene, TopviewGraph *topviewGraph);
    virtual ~GraphView();

    double getScale() const;
    QPointF getCircleCenter()
    {
        return circleCenter_;
    };

    void resetViewTransformation();

    //################//
    // EVENTS         //
    //################//

public:
    virtual void resizeEvent(QResizeEvent *event);
    virtual void scrollContentsBy(int dx, int dy);

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);

    virtual void wheelEvent(QWheelEvent *event);

    virtual void keyPressEvent(QKeyEvent *event);
    virtual void keyReleaseEvent(QKeyEvent *event);

    //################//
    // SLOTS          //
    //################//

public slots:
    // Tools, Mouse & Key //
    //
    void toolAction(ToolAction *);
    //	void						mouseAction(MouseAction *);
    //	void						keyAction(KeyAction *);

    // Rulers //
    //
    void rebuildRulers();
    void activateRulers(bool activate);

    // Zoom //
    //
    void zoomTo(const QString &zoomFactor);
    void zoomIn();
    void zoomIn(double zoom);
    void zoomOut();
    void zoomBox();
    void viewSelected();
    void scaleView(qreal sx, qreal sy);
    double getScaling() 
    {
        return scaling_;
    }

    // Background Images //
    //
    void loadMap();
    void deleteMap();
    void lockMap(bool lock);
    void setMapOpacity(const QString &opacity);
    void setMapX(double x);
    void setMapY(double y);
    void setMapWidth(double width, bool keepRatio);
    void setMapHeight(double height, bool keepRatio);

    //################//
    // PROPERTIES     //
    //################//

private:
    TopviewGraph *topviewGraph_;
    GraphScene *graphScene_;
	ZoomTool *zoomTool_;

    bool doPan_;
    bool doKeyPan_;

    BoundingBoxStatusId doBoxSelect_;
    CircleStatusId doCircleSelect_;
    double radius_;
    QGraphicsPathItem *circleItem_;
    QPointF circleCenter_;

    Ruler *horizontalRuler_;
    Ruler *verticalRuler_;
    bool rulersActive_;

    QPoint mp_;
    QRubberBand *rubberBand_;
    bool additionalSelection_;

    // ScenerySystem //
    //
    ScenerySystemItem *scenerySystemItem_;

    double scaling_;
};

#endif // GRAPHVIEW_HPP
