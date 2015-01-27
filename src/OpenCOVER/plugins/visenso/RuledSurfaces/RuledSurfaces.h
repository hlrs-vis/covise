/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RuledSurfaces_H_
#define RuledSurfaces_H_
//
#include <cover/coVRPlugin.h>
#include <osg/Vec3>
#include <osg/Array>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <cover/coVRLabel.h>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Image>
//
#include <osg/AnimationPath>

using namespace osg;
using namespace opencover;
using namespace vrui;

class RuledSurfaces : public coVRPlugin, public coMenuListener
{
public:
    /****** variables ******/
    /*
   * Static member variable as pointer for the plugin.
   */
    static RuledSurfaces *plugin;
    //
    //RuledSurfaces(int number,double mScaleX, double mScaleY);
    //RuledSurfaces(std::string name, int number,double mScaleX, double mScaleY);
    RuledSurfaces();

    virtual ~RuledSurfaces();
    /****** methods ******/
    /*
   * Initializes the plugin.
   *
   * return:       bool
   *               true if the initialization
   *               is valid
   */
    virtual bool init();

    /*
   * Defines which modifications will be
   * done before the next rendering step.
   *
   * return:       void
   */
    void preFrame();
    bool TextureMode;
    bool Trennung;
    bool Striktionslinie;
    bool Gradlinie;
    bool Tangente;
    bool drall;
    bool stop; //stop Animation
    bool dreibeinAnim;
    bool tangEbene;
    bool setMenuEvent(std::string menuname);
    void ChangePresentationStep();

    //Torsen
    void Ebene();
    void Kegel(int, float);
    void OK(int, float);
    void Zylinder(int, float);
    void Oloid(int);
    void OZ(int, float);
    void Tangentialflaeche(int, float);
    //wsRF
    void Hyper(int, float, float);
    void Sattel(int);
    void Konoid(int, float);
    void Helix(int, float, float, float, float, float, float);
    // void Helix2(int,float,float,float,float,float,float);
    //Striktionslinien
    void StriktionslinieHyper(int, float, float);
    void StriktionslinieSattel(int);
    void StriktionslinieKonoid(int, float);
    void StriktionslinieHelix(int, float, float, float, float, float, float);
    void GradlinieTf(int, float);
    void TangenteTf();
    void normalenMitteln_Oloid(Vec3Array *Normals);
    void normalenMitteln_Quads(Vec3Array *Normals);
    void normalenMitteln_Triangles(Vec3Array *Normals);
    void createVec3_lines_Quads(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int);
    void createVec3_lines_Tangentialflaeche(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int numFaces);

    void createVec3_lines_Triangles(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int);
#pragma endregion
    Vec4Array *colorArray;
#pragma region Geoden
    //Torsen
    osg::ref_ptr<osg::Geode> ebenegeo;
    osg::ref_ptr<osg::Geode> ebenegeo2;
    osg::ref_ptr<osg::Geode> kegelgeo;
    osg::ref_ptr<osg::Geode> okgeo;
    osg::ref_ptr<osg::Geode> zylindergeo;
    osg::ref_ptr<osg::Geode> oloidgeo;
    osg::ref_ptr<osg::Geode> ozgeo;
    osg::ref_ptr<osg::Geode> tangentialflaechegeo;
    //wsRF
    osg::ref_ptr<osg::Geode> hypergeo;
    osg::ref_ptr<osg::Geode> sattelgeo;
    osg::ref_ptr<osg::Geode> konoidgeo;
    osg::ref_ptr<osg::Geode> helixgeo;
    osg::ref_ptr<osg::Geode> helix2geo;
    //Striktionslinien
    osg::ref_ptr<osg::Geode> shypergeo;
    osg::ref_ptr<osg::Geode> ssattelgeo;
    osg::ref_ptr<osg::Geode> skonoidgeo;
    osg::ref_ptr<osg::Geode> shelixgeo;
    osg::ref_ptr<osg::Geode> gradliniegeo;
    osg::ref_ptr<osg::Geode> tangentegeo;

    void Windschiefe();
    void Torsen();
    void Konoidvergleich();
    void step1();
    void step2();
    osg::ref_ptr<osg::Group> torsengroup;
    osg::ref_ptr<osg::Group> windschiefegroup;
    osg::ref_ptr<osg::Group> konoidgroup;

    osg::ref_ptr<osg::Geode> g_Kegel;
    osg::ref_ptr<MatrixTransform> MT_Kegel;
    osg::ref_ptr<osg::Geode> g_OK;
    osg::ref_ptr<MatrixTransform> MT_OK;
    osg::ref_ptr<osg::Geode> g_Zylinder;
    osg::ref_ptr<MatrixTransform> MT_Zylinder;
    osg::ref_ptr<osg::Geode> g_OZ;
    osg::ref_ptr<MatrixTransform> MT_OZ;
    osg::ref_ptr<osg::Geode> g_Oloid;
    osg::ref_ptr<MatrixTransform> MT_Oloid;
    osg::ref_ptr<osg::Geode> g_Hyper;
    osg::ref_ptr<MatrixTransform> MT_Hyper;
    osg::ref_ptr<osg::Geode> g_Sattel;
    osg::ref_ptr<MatrixTransform> MT_Sattel;
    osg::ref_ptr<osg::Geode> g_Konoid;
    osg::ref_ptr<MatrixTransform> MT_Konoid;
    osg::ref_ptr<osg::Geode> g_Schraube;
    osg::ref_ptr<MatrixTransform> MT_Schraube;

    osg::ref_ptr<osg::Geode> g1_Konoid;
    osg::ref_ptr<MatrixTransform> MT1_Konoid;
    osg::ref_ptr<osg::Geode> g2_Konoid;
    osg::ref_ptr<MatrixTransform> MT2_Konoid;
    void startvariablen();

    bool showTorsen;

    bool showWindschiefe;

    bool greenKegel, greenOK, greenZylinder, greenOZ, greenOloid, greenHyper, greenSattel, greenKonoid, greenHelix;
    bool interact;

    Vec4 black, white, red, red2, lime, blue, yellow, aqua, fuchsia, gold, silver, purple, olive, teal, grey, navy, maroon, green, pink, brown;
    Vec4 color003, color102, color201, color300, color012, color111, color210, color021, color120, color030;
    Vec4 randomColor();

    //osg::AnimationPath* createAnimationPath(osg::ref_ptr<osg::Vec3Array>);

    ref_ptr<Group> mp_GeoGroup;
    osg::ref_ptr<MatrixTransform> v010;

    float en_u(float u, float v);
    float nz_u(float u, float v);
    float z_u(float u, float v);

    float x(float u);
    float y(float u);
    float z(float v);

    //HfT_osg_Input *mp_inputSurf;
    double mSx, mSy;

private:
    /*
   * Root of all Nodes
   */
    osg::ref_ptr<osg::Group> root;
    /*
	* Counter variable to store the current presentation step
	*/
    int m_presentationStepCounter;

    /*
	* Counter variable to store the number of presentation steps
	*/
    int m_numPresentationSteps;

    /*
	* Variable to store the current presentation step number
	*/
    int m_presentationStep;

    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu1;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu2;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu3;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu4;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu5;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu6;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu7;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu8;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu9;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu10;
    /* 
	*Pointer to the button for the Animation Dreibein
	*/
    coButtonMenuItem *m_pButtonMenuDreibein;
    /* 
	*Pointer to the checkbox for the Animation Tangentialebene
	*/
    coCheckboxMenuItem *m_pCheckboxMenuTangEbene;
    /*
	* Pointer to the slider for the Level of Detail
	*/
    coSliderMenuItem *m_pSliderMenuDetailstufe;
    int m_sliderValueDetailstufe;
    /*
	* Pointer to the slider for the variable phi
	*/
    coSliderMenuItem *m_pSliderMenuPhi;
    double m_sliderValuePhi;
    /*
	* Pointer to the slider for the variable "Erzeugung"
	*/
    coSliderMenuItem *m_pSliderMenuErzeugung;
    double m_sliderValueErzeugung;

    /*
	* Pointer to the slider for the outside radius
	*/
    coSliderMenuItem *m_pSliderMenuRadAussen;
    double m_sliderValueRadAussen;
    /*
	* Pointer to the slider for the outer height
	*/
    coSliderMenuItem *m_pSliderMenuHoeheAussen;
    double m_sliderValueHoeheAussen;
    /*
	* Pointer to the slider for the "Schraubhoehe"
	*/
    coSliderMenuItem *m_pSliderMenuSchraubhoehe;
    double m_sliderValueSchraubhoehe;
    /*
	*Pointer to the checkbox for enabling the wireframe
	*/
    coCheckboxMenuItem *m_pCheckboxMenuGitternetz;

    /* 
	*Pointer to the checkbox for enabling the line of striction
	*/
    coCheckboxMenuItem *m_pCheckboxMenuStriktion;
    /* 
	*Pointer to the checkbox for enabling the separation of surface oloid
	*/
    coCheckboxMenuItem *m_pCheckboxMenuTrennung;
    /* 
	*Pointer to the checkbox for enabling the "drall"
	*/
    coCheckboxMenuItem *m_pCheckboxMenuDrall;
    /* 
	*Pointer to the checkbox for enabling the tangent of the "Tangentialflaeche"
	*/
    coCheckboxMenuItem *m_pCheckboxMenuTangente;
    /* 
	*Pointer to the checkbox for enabling the line of the "Tangentialflaeche"
	*/
    coCheckboxMenuItem *m_pCheckboxMenuGrad;

    string surface;
    int m_Mode, m_ModeAnz;
    bool m_permitDeleteGeode;
    bool m_permitSelectGeode;
    bool m_permitMoveGeode;

    void dreibeinAchse(Geode *geod, Vec4 color);
    void setMaterial_surface(Geometry *geom, Vec4 color);
    void setMaterial_line(Geometry *geom2, Vec4 color);
    void setMaterial_striktline(Geometry *geom2, Vec4 color);
    void removeAddSurface(string surface_);
    void Normale(Vec3f);

    void runden(double phi);
    void fillNormalArrayQuads(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int);
    void fillNormalArrayTangentialflaeche(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int);
    void fillNormalArrayTriangles(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int);
    void createTextureCoordinates(int, osg::ref_ptr<osg::Geometry>);
    void setColorGeode(osg::ref_ptr<osg::Geode>, Vec4);
    void DreibeinKegel();
    void DreibeinZylinder();
    void DreibeinHyper();
    void DreibeinHyper2();
    void DreibeinSattel();
    void DreibeinKonoid();
    void DreibeinHelix();
    void Tangentialebene1(); //Konoid
    void Tangentialebene2();
    void Tangentialebene3();
    void Tangentialebene4();
    void Tangentialebene5();
    void Tangentialebene_Kegel();
    void Tangentialebene_OK();
    void Tangentialebene_Zylinder();
    void Tangentialebene_OZ();
    void Tangentialebene_Tangentialflaeche();
    void Tangentialebene_Hyper();
    void Tangentialebene_Helix();
    void Tangentialebene_Sattel();
    osg::ref_ptr<osg::Group /*Node*/> dreibein;

    osg::ref_ptr<osg::Vec3Array> pathpoints;
    osg::ref_ptr<osg::Vec3Array> pathpoints2;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BKegel;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BOK;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BZylinder;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BOZ;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BHyper;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BHyper2;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BSattel;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BSattel2;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BKonoid;
    osg::ref_ptr<osg::Vec2Array> angle3BKonoid;
    osg::ref_ptr<osg::Vec3Array> pathpoints3BHelix;
    osg::ref_ptr<osg::Vec3Array> pathpointst1;
    osg::ref_ptr<osg::Vec3Array> pathpointst2;
    osg::ref_ptr<osg::Vec3Array> pathpointst3;
    osg::ref_ptr<osg::Vec3Array> pathpointst4;
    osg::ref_ptr<osg::Vec3Array> pathpointst5;
    osg::ref_ptr<osg::Vec3Array> pathpoints_kegel;
    osg::ref_ptr<osg::Vec3Array> pathpoints_ok;
    osg::ref_ptr<osg::Vec3Array> pathpoints_zylinder;
    osg::ref_ptr<osg::Vec3Array> pathpoints_oz;
    osg::ref_ptr<osg::Vec3Array> pathpoints_tangentialflaeche;
    osg::ref_ptr<osg::Vec3Array> pathpoints_sattel;
    osg::ref_ptr<osg::Vec3Array> pathpoints_hyper;
    osg::ref_ptr<osg::Vec3Array> pathpoints_helix;
    osg::ref_ptr<osg::Vec3Array> pathpoints_helix2;

    void createMenu();

    /*
	* Calculates the presenatation step number with modulo.
	* Defines the visualisation for each presentation step.
	* Initiates a recalculation of the showing object for each step.
	*
	* return:       void
	*/
    void changePresentationStep();
    /*
	* Controls the messages which are sent
	* from the toolbar by pushing the toolbar buttons.
	* A counter stores the current presentation step.
	*
	* Parameters:       const char *msg
	*                   pointer to char for the message
	*
	* return:       void
	*/
    void guiToRenderMsg(const char *msg);
    /*
	* Controls which slider button in the slider menu is moved.
	* Dependent on the current presentation step, the correct
	* directrix is shown.
	*
	* Parameters:       coMenuItem *iMenuItem
	*                   chosed menu button
	*
	* return:       void
	*/
    //void menuReleaseEvent(coMenuItem *iMenuItem);
    void menuEvent(coMenuItem *iMenuItem);
    void setMenuVisible(int step);
    coNavInteraction *interactionA; //Button A for selecting a point
};
#endif /* RuledSurfaces_H_ */
