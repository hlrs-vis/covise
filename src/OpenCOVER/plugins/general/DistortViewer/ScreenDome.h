/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"

/**
  * Screen Dome Klasse.
  */
class ScreenDome : public Screen
{
public:
    /**
     * Konstruktor.
     *
     * @param screenData Daten zum Erstellen des Screens.
     */
    ScreenDome();

    /**
     * Destruktor.
     */
    ~ScreenDome(void);

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
        return "Dome";
    }

    /**
     * Radius des Domes setzen
     *
     * @param radius Radius des Domes.
     */
    void setRadius(float radius);

    /**
     * Radius des Domes zurückgeben.
     *
     * @return Radius des Domes.
     */
    float getRadius() const
    {
        return cRadius;
    }

    /**
     * Auflösung des Domes in Polarrichtung (Höhe) setzen.
     *
     * @param resolution Auflösung des Domes in Polarrichtung.
     */
    void setPolarResolution(unsigned int resolution);

    /**
     * Auflösung des Domes in Polarrichtung (Höhe) zurückgeben.
     *
     * @return Auflösung des Domes in Polarrichtung.
     */
    unsigned int getPolarResolution() const
    {
        return cPolarResolution;
    }

    /**
     * Auflösung des Domes in Azimutalrichtung setzen.
     *
     * @param resolution Azimuth-Auflösung des Domes.
     */
    void setAzimResolution(unsigned int resolution);

    /**
     *  Auflösung des Domes in Azimutalrichtung zurückgeben.
     *
     * @return azimResolution Azimuth-Auflösung des Domes.
     */
    unsigned int getAzimResolution() const
    {
        return cAzimResolution;
    }

    /**
     * Größe des Domeabschnitts in Polarrichtung setzen .
     *
     * @param segSize Größe des Domeabschnitts in Polarrichtung (Polarwinkel in Grad).
     */
    void setPolarSegmentSize(float segSize);

    /**
     * Größe des Domeabschnitts in Polarrichtung zurückgeben .
     *
     * @return Größe des Domeabschnitts in Polarrichtung (Polarwinkel in Grad).
     */
    float getPolarSegmentSize() const
    {
        return cPolarSegSize;
    }

    /**
     * Größe des Domeabschnitts in Azimutrichtung setzen .
     *
     * @param segSize Größe des Domeabschnitts in Azimutrichtung (Azimutwinkel in Grad).
     */
    void setAzimSegmentSize(float segSize);

    /**
     * Größe des Domeabschnitts in Azimutrichtung zurückgeben .
     *
     * @return Größe des Domeabschnitts in Azimutrichtung (Azimutwinkel in Grad).
     */
    float getAzimSegmentSize() const
    {
        return cAzimSegSize;
    }

    /**
     * Dome Projektionsgeometrie erstellen.
     *
     * @param mesh True:Gittermodell, False: Polygonmodell).
     */
    osg::Geode *drawScreen()
    {
        return drawScreen(stateMesh);
    };
    osg::Geode *drawScreen(bool gitter);

private:
    float cRadius; //Abstand von Oberflächenpunkt P zum Ursprung
    unsigned int cAzimResolution; //Auflösung in Azimutalrichtung (x-y-Ebene)
    unsigned int cPolarResolution; //Auflösung in Polarrichtung (z-Richtung)
    float cAzimSegSize; //Größe des Domeabschnitts in Azimutrichtung (Azimutwinkel in Grad)
    float cPolarSegSize; //Größe des Domeabschnitts in Polarrichtung (Polarwinkel in Grad)
     };
