/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COLORMAPLISTWIDGET_H
#define COLORMAPLISTWIDGET_H

#include <QWidget>
#include <QPixmap>

class PixmapWidget : public QWidget
{
public:
    PixmapWidget(const QPixmap &);
    PixmapWidget(const QPixmap &, const QString &);
    ~PixmapWidget();

    const QPixmap *pixmap() const
    {
        return &pm;
    }

    int height() const;
    int width() const;

    int rtti() const;
    enum
    {
        RTTI = 2
    };

protected:
    void paint(QPainter *);

private:
    QPixmap pm;
    QString name;
};

#endif
