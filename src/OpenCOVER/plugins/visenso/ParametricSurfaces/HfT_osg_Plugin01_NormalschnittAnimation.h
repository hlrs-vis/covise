/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HfT_osg_Plugin01_NormalschnittAnimation_H_
#define HfT_osg_Plugin01_NormalschnittAnimation_H_

#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/Geometry>

#include "HfT_osg_Plugin01_ParametricSurface.h"
#include "HfT_osg_Plugin01_ParametricPlane.h"

using namespace osg;

class HfT_osg_Plugin01_NormalschnittAnimation
{
    friend class HfT_osg_Plugin01;

public:
    float fontSize;
    Vec4 color1;
    Vec4 color2;
    std::string text_1;
    std::string text_2;
    Vec3 position1;
    Vec3 position2;
    HfT_osg_Plugin01_NormalschnittAnimation();
    bool m_pfeil_exist;
    bool KruemmungsMethode_aktiv();
    void Remove_Pfeil(MatrixTransform *mts1, MatrixTransform *mts2);
    void Create_Pfeil(osg::Vec3 startPickPos, HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    double m_normalwinkel;
    bool Normalschnitt_Animation(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    bool m_normalebene_exist;
    bool Remove_NormalEbenen(MatrixTransform *mts1);
    bool Remove_Schnittkurve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    double m_normalwinkel_schritt;
    bool m_hauptkruemmungsrichtung_exist;
    bool Hauptkruemmungsrichtungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    bool m_hauptkruemmung_exist;
    void Remove_Hauptkruemmungsrichtungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    bool Hauptkruemmungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    void Remove_Hauptkruemmungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    bool m_schmiegtangente_exist;
    void Schmiegtangenten(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    void Remove_Schmiegtangenten(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    bool Schiefschnitt_Animation(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    double m_schiefwinkel;
    double m_schiefwinkel_schritt;
    bool m_kruemmungskreise_exist;
    bool m_meusnierkugel_exist;
    bool Remove_MeusnierKugel(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    bool Remove_Kruemmungskreise(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1);
    void Meusnier_Kugel(HfT_osg_Plugin01_ParametricSurface *surf, int i, MatrixTransform *mts1);
    bool Remove_SchiefEbenen(MatrixTransform *mts1);
    bool Remove_SchiefSchnittkurve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);
    bool Clear(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts1, MatrixTransform *mts2);

protected:
    void quicksort(Vec2Array *s, int left, int right);
    int partition(Vec4Array *s, int left, int right);
    void Quicksort(ref_ptr<Vec3Array> punkte);
    int m_punkt;
    bool m_schiefebene_exist;
    bool m_schnittkurve_exist;
    bool m_schiefschnittkurve_exist;
    ref_ptr<osg::Switch> m_Pfeil;
    ref_ptr<osg::Switch> m_Pfeil_in_Ebene;
    ref_ptr<osg::Switch> m_normalEbenen;
    ref_ptr<osg::Switch> m_schiefEbenen;
    ref_ptr<osg::Switch> m_Schnittkurven;
    ref_ptr<osg::Switch> m_Schnittkurven_in_Ebene;
    ref_ptr<osg::Switch> m_SchiefSchnittkurven;
    ref_ptr<osg::Switch> m_SchiefSchnittkurven_in_Ebene;
    ref_ptr<osg::Switch> m_HptKrPfeile;
    ref_ptr<osg::Switch> m_MeusnierKugel;
    ref_ptr<osg::Switch> m_Kruemmungskreise;
    Geode *mp_Hauptkruemmungen_auf_Flaeche;
    Geode *mp_Hauptkruemmungen_in_Ebene;
    Geode *mp_Schmiegtangente_auf_Flaeche;
    ref_ptr<Vec3dArray> m_geradenPunkteNeu;
    ref_ptr<Vec3Array> m_schnittpunkte;
    ref_ptr<Drawable> Create_Plane(HfT_osg_Plugin01_ParametricSurface *surf, ref_ptr<MatrixTransform> v, ref_ptr<MatrixTransform> nd, ref_ptr<MatrixTransform> d);
    ref_ptr<Drawable> Schnittkurvenberechnung_auf_Flaeche(HfT_osg_Plugin01_ParametricSurface *surf);
    ref_ptr<Drawable> Schnittkurvenberechnung_in_Ebene(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane);
    bool Change_Mode(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, int mode1);
    float Abstand3(osg::Vec3, osg::Vec3);
    Vec2d NewtonVerfahren(osg::Vec3, osg::Vec2, HfT_osg_Plugin01_ParametricSurface *surf);
    double Abstand2(osg::Vec2d, osg::Vec2d);
    double Skalarprodukt(osg::Vec3d, osg::Vec3d);
    ref_ptr<Drawable> Create_Kruemmungskreis(double k);
};
#endif
