/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"

/**
  * Screen Plane Klasse.
  */
class ScreenPlane : public Screen
{
public:
    /**
     * Konstruktor.
     */
    ScreenPlane();

    ScreenPlane(float width,
                float height,
                unsigned int resWidth = 1,
                unsigned int resHeight = 1);

    /**
     * Destruktor.
     */
    ~ScreenPlane(void);

    /**
     * Läd Einstellungen aus XML-file
     *
     */
    void loadFromXML();

    /**
     * Speichert Einstellungen in XML-file
     *
     */
    void saveToXML();

    /**
     * Screen-Typ zurückgeben
     *
     * @return Typ der Projektionsgeometrie.
     */
    std::string getShapeType()
    {
        return "Plane";
    }

    /**
     * Höhe der Fläche setzen
     *
     * @param height Höhe der Fläche.
     */
    void setHeight(float height);

    /**
     * Höhe der Fläche zurückgeben.
     *
     * @return Höhe der Fläche.
     */
    float getHeight() const
    {
        return cHeight;
    }

    /**
     * Breite der Fläche setzen
     *
     * @param width Breite der Fläche.
     */
    void setWidth(float width);

    /**
     * Breite der Fläche zurückgeben.
     *
     * @return Breite der Fläche.
     */
    float getWidth() const
    {
        return cWidth;
    }

    /**
     * Auflösung der Fläche in der Höhe setzen.
     *
     * @param resolution Auflösung der Fläche in der Höhe.
     */
    void setHeightResolution(unsigned int resolution);

    /**
     * Auflösung der Fläche in der Höhe zurückgeben.
     *
     * @return Auflösung des Domes in Polarrichtung.
     */
    unsigned int getHeightResolution() const
    {
        return cHeightResolution;
    }

    /**
     * Auflösung der Fläche in der Breite setzen.
     *
     * @param resolution Auflösung der Fläche in der Breite.
     */
    void setWidthResolution(unsigned int resolution);

    /**
     *  Auflösung der Fläche in der Breite zurückgeben.
     *
     * @return Auflösung der Fläche in der Breite
     */
    unsigned int getWidthResolution() const
    {
        return cWidthResolution;
    }

    /**
     * Projektionsfläche erstellen.
     *
     * @param mesh True:Gittermodell, False: Polygonmodell.
     */
    osg::Geode *drawScreen()
    {
        return drawScreen(stateMesh);
    };
    osg::Geode *drawScreen(bool gitter);

private:
    float cHeight; //Höhe der Projektionsfläche (US bei cHeigth/2)
    float cWidth; //Breite der Projektionsfläche (US bei cWidth/2)
    unsigned int cHeightResolution; //Auflösung in Höhe (z-Richtung)
    unsigned int cWidthResolution; //Auflösung in Breite (x-Richtung)
};
