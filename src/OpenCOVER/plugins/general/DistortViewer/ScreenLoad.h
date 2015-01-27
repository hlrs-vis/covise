/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"

class ScreenLoad : public Screen
{
public:
    ScreenLoad(void);
    ~ScreenLoad(void);

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
     * Dateipfad des Screens setzen
     *
     * @param Dateipfad zur Screengeometrie.
     */
    void setFilename(std::string new_fileName);

    /**
     * Prüft ob Dateipfad exisitert
     *
     * @return ja=true/nein=false.
     */
    bool fileIsValid();
    bool fileIsValid(std::string filePath);

    /**
     * Dateipfad des Screens zurückgeben
     *
     * @return Dateipfad zur Screengeometrie.
     */
    std::string getFilename()
    {
        return fileName;
    };

    /**
     * Screen-Typ zurückgeben
     *
     * @return Typ der Projektionsgeometrie.
     */
    std::string getShapeType() const
    {
        return "Custom Geometry";
    };

    /**
     * Projektionsgeometrie laden und erstellen.
	 *
     */
    osg::Geode *drawScreen()
    {
        return drawScreen(stateMesh);
    };
    osg::Geode *drawScreen(bool gitter);

private:
    std::string fileName;
};
