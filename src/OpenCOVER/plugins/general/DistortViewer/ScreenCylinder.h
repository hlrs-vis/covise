/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"

/**
  * Screen Zylinder Klasse.
  */
class ScreenCylinder : public Screen
{
public:
    /**
     * Konstruktor.
     *
     * @param screenData Daten zum Erstellen des Screens.
     */
    ScreenCylinder(void);

    /**
     * Destruktor.
     */
    ~ScreenCylinder(void);

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
    std::string getShapeType() const
    {
        return "Cylinder";
    }

    /**
     * Radius des Zylinders setzen
     *
     * @param radius Radius des Zylinders.
     */
    void setRadius(float radius);

    /**
     * Radius des Zylinders zurückgeben.
     *
     * @return Radius des Zylinders.
     */
    float getRadius() const
    {
        return cRadius;
    }

    /**
     * Höhe des Zylinders setzen
     *
     * @param height Radius des Zylinders.
     */
    void setHeight(float height);

    /**
     * Höhe des Zylinders zurückgeben.
     *
     * @return Höhe des Zylinders.
     */
    float getHeight() const
    {
        return cHeight;
    }

    /**
     * Auflösung des Zylinders in z-Richtung (Höhe) setzen.
     *
     * @param resolution Auflösung des Zylinders in z-Richtung.
     */
    void setZResolution(unsigned int resolution);

    /**
     * Auflösung des Zylinders in z-Richtung (Höhe) zurückgeben.
     *
     * @return zResolution Auflösung des Zylinders in z-Richtung.
     */
    unsigned int getZResolution() const
    {
        return cZResolution;
    }

    /**
     * Auflösung des Zylinders in Azimutalrichtung setzen.
     *
     * @param resolution Azimuth-Auflösung des Zylinders.
     */
    void setAzimResolution(unsigned int resolution);

    /**
     *  Auflösung des Zylinders in Azimutalrichtung zurückgeben.
     *
     * @return azimResolution Azimuth-Auflösung des Zylinders.
     */
    unsigned int getAzimResolution() const
    {
        return cAzimResolution;
    }

    /**
     * Größe des Zylinderabschnitts setzen .
     *
     * @param segSize Größe des Zylinderabschnitts (Azimutwinkel in Grad).
     */
    void setSegmentSize(float segSize);

    /**
     * Größe des Zylinderabschnitts zurückgeben .
     *
     * @return Größe des Zylinderabschnitts (Azimutwinkel in Grad).
     */
    float getSegmentSize() const
    {
        return cSegSize;
    }

    /**
     * Zylinder Projektionsgeometrie erstellen.
	 *
	 * @param mesh Geometrie als Gittermodell(True) oder Polygonmodell(False) erstellen?
     */
    osg::Geode *drawScreen()
    {
        return drawScreen(stateMesh);
    };
    osg::Geode *drawScreen(bool gitter);

private:
    float cRadius; //Abstand von Oberflächenpunkt P zum Ursprung
    float cHeight; //Höhe des Zylinders
    float cSegSize; //Größe des Zylinderabschnitts: Azimutwinkel (in x-y-Ebene) in Grad
    unsigned int cZResolution; //Auflösung in z-Richtung (Höhe)
    unsigned int cAzimResolution; //Auflösung in Azimutalrichtung
};
