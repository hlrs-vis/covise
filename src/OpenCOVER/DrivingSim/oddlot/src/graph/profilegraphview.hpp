/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.06.2010
**
**************************************************************************/

#ifndef PROFILEGRAPHVIEW_HPP
#define PROFILEGRAPHVIEW_HPP

#include <QGraphicsView>
#include <QRubberBand>

class ProfileGraphScene;
class Ruler;

class ToolAction;

class ProfileGraphView : public QGraphicsView
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProfileGraphView(ProfileGraphScene *scene, QWidget *parent);
    virtual ~ProfileGraphView();

    void resetViewTransformation();

protected:
private:
    ProfileGraphView(); /* not allowed */
    ProfileGraphView(const ProfileGraphView &); /* not allowed */
    ProfileGraphView &operator=(const ProfileGraphView &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    void zoomIn(Qt::Orientations);
    void zoomOut(Qt::Orientations);
    void rebuildRulers();

    void toolAction(ToolAction *);

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
    // PROPERTIES     //
    //################//

protected:
private:
    bool doPan_;
    bool doKeyPan_;

    int doBoxSelect_;
    QRubberBand *rubberBand_;
    bool additionalSelection_;

    Ruler *horizontalRuler_;
    Ruler *verticalRuler_;

    QPoint mp_;
};

#endif // PROFILEGRAPHVIEW_HPP
