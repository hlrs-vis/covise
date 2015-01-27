/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"
#include "VisScene.h"

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Camera>
#include <osgUtil/SmoothingVisitor>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/LineWidth>
#include <osgText/Text>
#include <osgUtil/LineSegmentIntersector>

#include <cmath>
#include <iostream>

class Projector
{
public:
    /**
	* Konstruktor
	*/
    Projector(bool load = false);

    /**
	* Destruktor
	*/
    ~Projector(void);

    /**
	* Läd einstellungen aus XML-file
	*
	* @return Laden erfolgreich? true/false
	*/
    bool loadFromXML();

    /**
	* Speichert einstellungen in XML-file
	*
	* @return Speichern erfolgreich? true/false
	*/
    bool saveToXML();

    /**
	* Gibt die Scene zur Visualisierung zurück
	*
	* @return Visualisierungsscene
	*/
    VisScene *getVisScene(void)
    {
        return visScene;
    };

    /**
	* Setzt die aktuelle Projektornummer (position Array)
	*
	* @param new_projNum Neue Projektornummer.
	*/
    void setProjectorNum(int new_projNum);
    int getProjectorNum(void)
    {
        return projNum;
    };

    /**
	* Projektionsverhältnis des Projektors setzen
	*
	* @param ratio Projektionsverhältnis des Projektors
	*/
    void setProjRatio(float ratio);

    /**
	* Projektionsverhältnis des Projektors zurückgeben
	*
	* @return Projektionsverhältnis des Projektors
	*/
    float getProjRatio()
    {
        return projRatio;
    };

    /**
	* Öffnungswinkel des Projektor-Frustums zur Y, bzw- X-Achse zurückgeben
	*
	* @return Öffnungswinkel (in Grad) des Projektor-Frustums
	*/
    float getFovY();
    float getFovX();

    /**
	* AspectRatio des Projektor-Frustums setzen
	*
	* @param aspectRatio AspectRatio (in Grad) des Projektor-Frustums
	*/
    void setAspectRatioH(float aspectRatio);
    void setAspectRatioW(float aspectRatio);

    /**
	* AspectRatio des Projektor-Frustums zurückgeben
	*
	* @return AspectRatio (in Grad) des Projektor-Frustums
	*/
    float getAspectRatioH()
    {
        return aspectRh;
    };
    float getAspectRatioW()
    {
        return aspectRw;
    };

    /**
	* LensShift des Projektor-Frustums setzen
	*
	* @param lensShift LensShift des Projektor-Frustums
	*/
    void setLensShiftH(float lensShift);
    void setLensShiftV(float lensShift);

    /**
	* LensShift des Projektor-Frustums zurückgeben
	*
	* @return LensShift des Projektor-Frustums
	*/
    float getLensShiftH()
    {
        return shiftx;
    };
    float getLensShiftV()
    {
        return shifty;
    };

    /**
	* NearClipping-Ebene des Projektor-Frustums setzen
	*
	* @param nearClipping Entfernung der NearClipping-Ebene des Projektor-Frustums
	*/
    void setNearClipping(float nearClipping);

    /**
	* NearClipping-Ebene des Projektor-Frustums zurückgeben
	*
	* @return Entfernung der NearClipping-Ebene des Projektor-Frustums
	*/
    float getNearClipping()
    {
        return near_c;
    };

    /**
	* FarClipping-Ebene des Projektor-Frustums setzen
	*
	* @param farClipping Entfernung der NearClipping-Ebene des Projektor-Frustums
	*/
    void setFarClipping(float farClipping);

    void setStateFrust(bool new_state);
    bool getStateFrust(void)
    {
        return stateFrust;
    };

    /**
	* FarClipping-Ebene des Projektor-Frustums zurückgeben
	*
	* @return Entfernung der farClipping-Ebene des Projektor-Frustums
	*/
    float getFarClipping()
    {
        return far_c;
    };

    /**
	* Projektionsmatrix des Projektors zurückgeben
	*
	* @return Projektionsmatrix des Projektors
	*/
    osg::Matrix getProjMat();

    /**
	* Transformationsmatrix des Projektors zurückgeben
	*
	* @param matrix Projektionsmatrix des Projektors
	*/
    osg::Matrix getTransMat();

    /**
	* Position des Projektors setzen
	*
	* @param Positionsvektor (x,y,z)
	*/
    void setPosition(osg::Vec3 pos);

    /**
	* Position des Projektors zurückgeben
	*
	* @return Positionsvektor des Projektors (x,y,z)
	*/
    osg::Vec3 getPosition()
    {
        return position;
    };

    /**
	* Projektionsrichtung des Projektors setzen
	*
	* @param Projektionsrichtung des Projektors als Vektor
	*/
    void setProjDirection(osg::Vec3 projDir);

    /**
	* Projektionsrichtung des Projektors zurückgeben
	*
	* @return Projektionsrichtung des Projektors als Vektor
	*/
    osg::Vec3 getProjDirection()
    {
        return projDirection;
    };

    /**
	* Vertikale Achse des Projektors setzen
	*
	* @param Vertikale Achse des Projektors als Vektor
	*/
    void setUpDirection(osg::Vec3 projUp);

    /**
	* vertikale Achse des Projektors zurückgeben
	*
	* @return Vertikale Achse des Projektors als Vektor
	*/
    osg::Vec3 getUpDirection()
    {
        return upDirection;
    };

    /**
	* Rotation des Projektors
	*
	* @param Rotation als Quaternion
	*/
    void rotate(osg::Matrix rotMat);

    /**
	* Translation des Projektors
	*
	* @param translation Verschiebungsvektor (Vec3)
	*/
    void translate(osg::Vec3f translation);

    /**
	* Viewmatrix des Projektors setzen
	*
	* @param matrix Viewmatrix des Projektors
	*/
    void setViewMat(osg::Matrix matrix);

    /**
	* Viewmatrix des Projektors zurückgeben
	*
	* @return Viewmatrix des Projektors
	*/
    osg::Matrix getViewMat();

    /**
	* Gibt die Lage des Projektorfrustums in der Near-Clippingebene zurück
	*
	* @return linke Seite, rechte Seite, oben, unten
	*/
    osg::Vec4 getFrustumSizeNear();

    /**
	* Gibt die Lage des Projektorfrustums in der Far-Clippingebene zurück
	*
	* @return linke Seite, rechte Seite, oben, unten
	*/
    osg::Vec4 getFrustumSizeFar();

    /**
	* Zeichnet das Projektorfrustum als Liniengeometrie mit Blickrichtung neg. z-Achse
	*
	* @return Geometrie des Frustums
	*/
    osg::Geometry *drawFrustum();

    /**
	* Zeichnet gesamte Projektorgruppe (frustum, virt. Projektionsschirm) in Endlage
	*
	* @return Gruppe mit Geometrie des Projektorfrustums und -screens
	*/
    osg::Group *draw();

    /**
	* Erstellt einen Kameraknoten mit Projektoreigenschaften
	*
	* @return Kameraknoten mit Projektoreigenschaften
	*/
    osg::Camera *getProjCam();

    //---------------------------------------
    // Virt. PROJEKTIONSSCREEN
    //---------------------------------------

    /**
	* virt. Projektionscreens in Config autom. berechnen?
	*
	* @param enabled ja=true/nein=false
	*/
    void setAutoCalc(bool enabled);

    /**
	* virt. Projektionscreens in Config autom. berechnen
	*
	* @return true/false
	*/
    bool getAutoCalc()
    {
        return autoCalc;
    };

    /**
	* Ebene des virt. Projektionscreens zurückgeben
	*
	* @return Ebene des Projektionsscreens
	*/
    osg::Plane getScreenPlane()
    {
        return projScreenPlane;
    };

    /**
	* Ebene des virt. Projektionscreens setzen
	*
	* @param plane Ebene des Projektionsscreens
	*/
    void setScreenPlane(osg::Plane plane);

    /**
	* Euler-Winkel des virt. Projektionscreens zurückgeben
	*
	* @return Eulerwinkel (h,p,r) als Vektor
	*/
    osg::Vec3 getEulerAngles()
    {
        return hprVec;
    };

    /**
	* Center-Position des virt. Projektionscreens zurückgeben
	*
	* @return Zentrum des Projektionsscreens
	*/
    osg::Vec3 getScreenCenter()
    {
        return projScreenCenter;
    };

    /**
	* Center-Position des virt. Projektionscreens setzen
	*
	* @param centerPos Koordinaten des Zentrums des Projektionsscreens
	*/
    void setScreenCenter(osg::Vec3 centerPos);

    /**
	* Höhe des virt. Projektionscreens zurückgeben
	*
	* @return Höhe des Projektionsscreens
	*/
    float getScreenHeight()
    {
        return projScreenHeight;
    };

    /**
	* Höhe des virt. Projektionscreens setzen
	*
	* @param height Höhe des Projektionsscreens
	*/
    void setScreenHeight(float height);

    /**
	* Breite des virt. Projektionscreens zurückgeben
	*
	* @return Breite des Projektionsscreens
	*/
    float getScreenWidth()
    {
        return projScreenWidth;
    };

    /**
	* Breite des virt. Projektionscreens setzen
	*
	* @param width Breite des Projektionsscreens
	*/
    void setScreenWidth(float width);

    /**
	* Transformationsmatrix vom KS des virt. Projektionscreens in WeltKS zurückgeben
	* Bsp: projScreenCenter * inverse(screenTransMat) = US im WeltKS
	*
	* @return Transformationsmatrix des Projektionsscreens
	*/
    osg::Matrix getScreenTransMat();

    /**
	* Gibt die Eckpunkte des Projektor-Frustums in der Far-Clippingebene zurück
	*
	* @return Eckpunkte des Projektor-Frustums in der Far-Clippingebene
	*/
    osg::Vec3Array *getFarCorners();

    /**
	* Gibt die Schnittpunkte der Projektionsgeometrie mit den Kanten des Projektor-Frustums zurück
	*
	* @return Schnittpunkte des Projektor-Frustums mit der Projektionsgeometrie (li-o,re-o,re-u,li-u)
	*/
    osg::Vec3Array *getScreenIntersecPnts();

    /**
	* Ermittelt die Ebene der virt. Projektionsfläche aus den drei zum Projektor nächstgelegenen Schnittpunkten
	* der Projektionsgeometrie mit den Kanten des Projektor-Frustums und gibt diese zurück.
	*
	* @return Ebene der virt. Projektionsfläche
	*/
    osg::Plane calcScreenPlane();

    /**
	* Gibt die Schnittpunkte der Ebene der virt. Projektionsfläche mit den Kanten des Projektor-Frustums zurück
	*
	* @return Schnittpunkte des Projektor-Frustums mit der Ebene der virt. Projektionsfläche (li-o,re-o,re-u,li-u,centerFrustum)
	*/
    osg::Vec3Array *getPlaneIntersecPnts();

    /**
	* Errechnet aus den Schnittpunkten der Ebene des virt. Projektionsfläche und den Kanten des Projektor-Frustums
	* die Eckpunkte der virt. Projektionsfläche. Unter der Voraussetzung, dass die Fläche minimal groß und rechteckig ist!
	* Zudem wird ScreenSenter, Hoehe und Breite gesetzt!
	*/
    void calcScreen();

    /**
	* Berechnet die nötigen Werte des virt. Projektionsscreens neu, nachdem eine Projektoreigenschaft neu gesetzt wurde.
	*
	*/
    void update();

    /**
	* Zeichnet den virt. Projektionsscreen in Endlage als Quad
	*
	* @param alpha Alpha-Wert in dem Screen dargestellt werden soll
	* @return Geometrie des virt. Projektionsscreens
	*/
    osg::Geometry *drawScreen(osg::Vec4 color = osg::Vec4(0.0f, 1.0f, 0.0f, 0.3));

    osg::Vec3 getHPR();

    bool active; //Projektor im Editor ausgewählt ja/nein?

private:
    VisScene *visScene; //Scene zur Visualisierung der Distortion

    //Projektornummern
    static int num; //Anzahl der existierenden Projektoren
    int projNum; //Nummer des Projektors (start bei 1)
    std::string projNum_str; //Projektornummer als String

    //für Projektionsmatrix der Kamera
    bool stateFrust; //Frustum darstellen ja/nein?
    float projRatio; //Projektionsverhältnis (projektionsabstand a / Bildbreite b)
    float aspectRh; //Aspect Ratio, Seitenverhältnis
    float aspectRw; //Aspect Ratio, Seitenverhältnis
    float near_c; //Clipping-Ebenen des Frustums (Nähe)
    float far_c; //Clipping-Ebenen des Frustums (Ferne)
    float shifty; //lense shift vertikal (vertikale verschiebung des proj. Bildes)
    float shiftx; //lense shift horizontal (horizontale verschiebung des proj. Bildes)

    //für View-Matrix des Projektors -> Position, Orientierung
    osg::Vec3 position;
    osg::Vec3 projDirection;
    osg::Vec3 upDirection;

    //virt. ProjektionsScreen
    bool autoCalc;
    osg::Plane projScreenPlane;
    osg::Vec3 hprVec; //Vektor mit Eulerwinkel (h,p,r)
    osg::ref_ptr<osg::Vec3Array> cornerPnts;
    osg::Vec3 projScreenCenter;
    float projScreenHeight;
    float projScreenWidth;
};
