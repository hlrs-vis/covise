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

#ifndef SCENERYMAP_HPP
#define SCENERYMAP_HPP

#include "src/data/dataelement.hpp"

class ScenerySystem;

#include <QImage>

class SceneryMap : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum SceneryMapChange
    {
        CSM_Id = 0x1, // TODO ID in SystemItem
        CSM_Filename = 0x2,
        CSM_X = 0x4,
        CSM_Y = 0x8,
        CSM_Width = 0x10,
        CSM_Height = 0x20,
        CSM_Opacity = 0x40,
        CSM_ScenerySystemChanged = 0x80,
    };

    enum SceneryMapType
    {
        DMT_Aerial = 0x1,
        DMT_Heightmap = 0x2,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SceneryMap(const QString &id, const QString &filename, double width, double height, SceneryMap::SceneryMapType mapType);
    virtual ~SceneryMap();

    // SceneryMap //
    //
    QString getId() const
    {
        return id_;
    }
    QString getFilename() const
    {
        return filename_;
    }

    double getX() const
    {
        return x_;
    }
    double getY() const
    {
        return y_;
    }
    double getWidth() const
    {
        return width_;
    }
    double getHeight() const
    {
        return height_;
    }
    double getOpacity() const
    {
        return opacity_;
    }
    SceneryMap::SceneryMapType getMapType() const
    {
        return mapType_;
    }

    const QImage &getImage() const
    {
        return image_;
    }
    bool isLoaded() const
    {
        return loaded_;
    }

    void setId(const QString &id);
    void setX(double x);
    void setY(double y);
    void setWidth(double width);
    void setHeight(double height);
    void setOpacity(double opacity);
    //	void						setMapType(SceneryMap::SceneryMapType mapType);
    void setFilename(const QString &filename);

    int getImageWidth() const
    {
        return imageWidth_;
    }
    int getImageHeight() const
    {
        return imageHeight_;
    }

    double getScaling() const;

    bool isIntersectedBy(const QPointF &point);

    // ScenerySystem //
    //
    ScenerySystem *getParentScenerySystem() const
    {
        return parentScenerySystem_;
    }
    void setParentScenerySystem(ScenerySystem *scenerySystem);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getSceneryMapChanges() const
    {
        return sceneryMapChanges_;
    }
    void addSceneryMapChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    SceneryMap(); /* not allowed */
    SceneryMap(const SceneryMap &); /* not allowed */
    SceneryMap &operator=(const SceneryMap &); /* not allowed */

    bool loadFile();

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // ScenerySystem //
    //
    ScenerySystem *parentScenerySystem_;

    // Change flags //
    //
    int sceneryMapChanges_;

    // SceneryMap //
    //
    QString id_;
    QString filename_;

    double x_; // in [m]
    double y_; // in [m]
    double width_; // in [m]
    double height_; // in [m]
    double opacity_; // [0,1]

    int imageWidth_;
    int imageHeight_;

    QImage image_;
    bool loaded_;

    mutable double scaling_;

    // Map Type //
    //
    SceneryMap::SceneryMapType mapType_;
};

#endif // SCENERYMAP_HPP
