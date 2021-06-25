/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/12/2010
**
**************************************************************************/

#ifndef HEIGHTMAP_HPP
#define HEIGHTMAP_HPP

#include "scenerymap.hpp"

class Heightmap : public SceneryMap
{

    //################//
    // STATIC         //
    //################//

public:
    enum HeightmapChange
    {
        CHM_DataFileChanged = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Heightmap(const QString &id, const QString &filename, double width, double height, SceneryMap::SceneryMapType mapType, const QString &heightMapFilename);
    virtual ~Heightmap();

    // Heightmap //
    //
    QString getHeightmapDataFilename() const
    {
        return heightmapFilename_;
    }
    bool isHeightmapDataLoaded() const
    {
        return heightmapDataLoaded_;
    }
    double getHeightmapValue(int i, int h) const;
    double getHeightmapValue(double x, double y) const;

    void setHeightmapDataFilename(const QString &filename);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getHeightmapChanges() const
    {
        return heightmapChanges_;
    }
    void addHeightmapChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    Heightmap(); /* not allowed */
    Heightmap(const Heightmap &); /* not allowed */
    Heightmap &operator=(const Heightmap &); /* not allowed */

    bool loadHeightmapDataFile();

    //################//
    // EVENTS         //
    //################//

public:
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
    int heightmapChanges_;

    // Heightmap //
    //
    QString heightmapFilename_;
    double *heightmapData_;
    bool heightmapDataLoaded_;
    QByteArray heightmapDataArray_;
};

#endif // HEIGHTMAP_HPP
