/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>

#include <cmath>
#include <iostream>

//Abstrakt -> kann nicht Instanziiert werden!
class Screen
{
public:
    /**
	* Konstruktor
	*/
    Screen();

    /**
	* Destruktor
	*/
    virtual ~Screen() = 0;

    /**
	* Läd einstellungen aus XML-file
	*
	*/
    virtual void loadFromXML();

    /**
	* Speichert einstellungen in XML-file
	*
	*/
    virtual void saveToXML();

    /** Form der Proj-Geometrie zurückgeben
	*
	* @return: Form der proj. Geometrie
	*/
    std::string getShapeType(void)
    {
        return geoShape;
    };

    /** Position der Proj-Geometrie setzen
	*
	* @param new_centerPos: Koordinaten des Mittelpunkts der Proj.-Geometrie
	*/
    void setCenterPos(osg::Vec3 new_centerPos);

    /** Position der Proj-Geometrie zurückgeben
	*
	* @return: Koordinaten des Mittelpunkts der Proj.-Geometrie
	*/
    osg::Vec3 getCenterPos(void)
    {
        return centerPos;
    };

    /** Skalierungsvektor der Proj.-Geometrie setzen
	*
	* @param new_scaleVec: Skalierungsvector der Proj.-Geometrie
	*/
    void setScaleVec(osg::Vec3 new_scaleVec);

    /** Skalierungsvektor der Proj.-Geometrie zurückgeben
	*
	* @return scaleVec: Skalierungsvektor der Proj.-Geometrie
	*/
    osg::Vec3 getScaleVec(void)
    {
        return scaleVec;
    };

    /** Orientierung der Proj.-Geometrie setzen
	*
	* @param new_orientation: Neue Orientierung der Proj.-Geometrie
	*/
    void setOrientation(osg::Vec3 new_orientation);

    /** Geometrie als Gitternetz anzeigen?
	*
	* @param new_state true/false
	*/
    void setStateMesh(bool new_state);

    /** Geometrie als Gitternetz
	*
	* @return Gitternetz true/false
	*/
    bool getStateMesh(void)
    {
        return stateMesh;
    };

    /** Orientierung der Proj.-Geometrie zurückgeben
	*
	* @return: Orientierung der Proj.-Geometrie
	*/
    osg::Vec3 getOrientation(void)
    {
        return orientation;
    };

    /** Transformationsmatrix des Screens neu berechnen
	*
	*/
    void calcTransMat(void);

    /** Transformationsmatrix des Screens zurückgeben
	*
	* @return Transformationsmatrix des Screens
	*/
    osg::Matrix getTransMat()
    {
        return sTransMat;
    };

    /**
     * Projektionsgeometrie erstellen.
     *
     * @param mesh True:Gittermodell, False: Polygonmodell).
     */
    virtual osg::Geode *drawScreen() = 0;
    virtual osg::Geode *drawScreen(bool gitter) = 0;

    /** Screen darstellen 
	*
	* @return Geometrie-Node des Screens 
	*/
    osg::MatrixTransform *draw()
    {
        return draw(stateMesh);
    };
    osg::MatrixTransform *draw(bool gitter);

protected:
    std::string geoShape; //Form der Projektionsgeometrie
    osg::Vec3 centerPos;
    osg::Vec3 scaleVec;
    osg::Vec3 orientation;
    osg::Matrix sTransMat; // Transformationsmatrix des screens.
    bool stateMesh; //Geometrie als Gittermodell(True) oder Polygonmodell(False) erstellen?
};