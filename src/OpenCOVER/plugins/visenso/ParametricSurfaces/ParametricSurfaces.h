/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARAMETRICSURFACES_H
#define PARAMETRICSURFACES_H
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
#include "cover/coVRLabel.h"
//
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Image>

#include "HfT_osg_Plugin01_ParametricSurface.h"
#include "HfT_osg_Plugin01_ParametricPlane.h"
#include "HfT_osg_ReadTextureImage.h"
#include "HfT_osg_Plugin01_NormalschnittAnimation.h"

using namespace osg;

using namespace vrui;
using namespace opencover;
class ParametricSurfaces : public coVRPlugin, public coMenuListener
{

public:
    /****** variables ******/
    /*
   * Static member variable as pointer for the plugin.
   */
    static ParametricSurfaces *plugin;
    //
    osg::Vec3 _startPickPos;
    ParametricSurfaces();

    virtual ~ParametricSurfaces();

    /****** methods ******/
    /*
   * Initializes the plugin.
   * Sets the number of presentation steps.
   * Manages the creation of the surfaces and
   * creates the scene graph tree hierarchy.
   * Creates the slider menu to move the u and v
   * parameter lines.
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
    float x_posDocView;
    float y_posDocView;
    float z_posDocView;
    float scale_DocView;
    float hsize_DocView;
    float vsize_DocView;
    bool changesize;

protected:
    ref_ptr<MatrixTransform> mtplane;
    bool m_enable; // disable <--> enable
    int m_startAnimation; // Start bzw. stopped Animationskugeln
    int m_maxcreateAnimation; // max. Anzahl der zulässigen Animationskugeln
    int m_iscreateAnimation; // mom. Anzahl der Animationskugeln;
    Vec4f *m_ColorAnimation; // Farbe der Animationskugeln;
    int m_Imagelistcounter; // Für Anzahl der verschiedenen TexturImages
    int m_Kugel_Index;
    bool m_natbound;
    double m_Pua, m_Pue, m_Pva, m_Pve;
    std::string m_Surfname;
    Matrix maplane; //Translation der Parameterebene
    ref_ptr<Group> mp_GeoGroup;
    HfT_osg_Plugin01_ParametricSurface *mp_Surf;
    HfT_osg_Plugin01_ParametricPlane *mp_Plane;
    HlParametricSurface3d *mp_ParserSurface;
    HfT_osg_ReadTextureImage *mp_TextureImage;
    ref_ptr<Geometry> mp_GeomNormals;
    HfT_osg_Plugin01_NormalschnittAnimation *mp_NormalschnittAnimation;
    void updateMenu(HfT_osg_Plugin01_ParametricSurface *surf);
    void updateBox0(HfT_osg_Plugin01_ParametricSurface *surf);
    void updateBox3(HfT_osg_Plugin01_ParametricSurface *surf);
    void updateBox6(HfT_osg_Plugin01_ParametricSurface *surf);
    void Change_Mode(HfT_osg_Plugin01_ParametricSurface *surf, int increase);
    bool Change_Geometry(HfT_osg_Plugin01_ParametricSurface *surf, int lod);
    bool Change_Cons(HfT_osg_Plugin01_ParametricSurface *surf, ConsType ctype);
    MatrixTransform *Get_MatrixTransformNode_Parent(Node *node);

    bool Sphere_AnimationCreate(HfT_osg_Plugin01_ParametricSurface *surf);

    bool Change_Image(HfT_osg_Plugin01_ParametricSurface *surf);
    bool Change_ConsPosition(HfT_osg_Plugin01_ParametricSurface *surf);
    Switch *Get_SwitchNode_Parent(Node *node);
    bool Disable_Plane(bool disable);
    void ChangePlanePosition(HfT_osg_Plugin01_ParametricSurface *surf);
    bool Change_Radius(HfT_osg_Plugin01_ParametricSurface *surf, double rad);
    bool Change_Length(HfT_osg_Plugin01_ParametricSurface *surf, double len);
    bool Change_Height(HfT_osg_Plugin01_ParametricSurface *surf, double height);
    bool Change_Ua(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double du);
    bool Change_Ue(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double du);
    bool Change_Va(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double dv);
    bool Change_Ve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double dv);

    bool Disable_Boundary(bool disable);
    bool Disable_Normals(HfT_osg_Plugin01_ParametricSurface *surf, bool disable);
    bool Remove_Animation();
    bool Gauss_Curvature_Mode(HfT_osg_Plugin01_ParametricSurface *surf);
    bool Mean_Curvature_Mode(HfT_osg_Plugin01_ParametricSurface *surf);
    bool CheckParametrization(HfT_osg_Plugin01_ParametricSurface *surf, std::string xpstr, std::string ypstr, std::string zpstr);
    bool Change_Surface(Group *surfgroup, HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane);
    HfT_osg_Plugin01_ParametricSurface *
    Create_Surf(Group *surfgroup, double a, double b, double c,
                int n, int m, int su, int sv,
                double ua, double ue, double va, double ve,
                string xstr, string ystr, string zstr,
                SurfMode smode, ConsType ctype, int canz, Image *image);
    HfT_osg_Plugin01_ParametricPlane *
    Create_Plane(HfT_osg_Plugin01_ParametricSurface *plane,
                 int n, int m, int su, int sv,
                 double m_Pua, double m_Pue, double m_Pva, double m_Pve,
                 double ua, double ue, double va, double ve,
                 SurfMode smode, ConsType ctype, int canz, Image *image);
    bool ReadSurfGroup(std::string path);
    bool WriteSurfGroup(std::string path);
    bool Read_Surface(Group *surfgroup, std::string surfname);
    void SliderMenu_Values();
    ref_ptr<Group> InitSurfgroup(std::string surfpath, std::string imagepath);
    bool Fill_SurfGroup(Group *surfgroup, double a, double b, double c, int n, int m, int su, int sv,
                        double ua, double ue, double va, double ve, string xstr, string ystr, string zstr,
                        SurfMode smode, ConsType ctype, int canz, Image *image);

private:
    int n_lod;
    int m_lod;
    int su_lod;
    int sv_lod;
    int changeMode;
    ConsType constyp;
    /****** variables ******/
    //for translating the plane in z because of collision with the surface if changing ABC
    /*std::string	surfpath;
	std::string imagepath;*/
    /*
   * Root of all Nodes
   */
    osg::ref_ptr<osg::Group> root;
    // actualize the slider in step 4 because of changing it in step 8
    bool actualizeSlider;
    // startvalue of the slider in step 4
    bool sliderStart;
    // actualize the slider in step 8 because of changing it in step 8 without being in step 4
    bool actualizeSliderStep8;
    //notice the lastStep of the coCheckboxMenuItem Plane on/off to change the status in the current step
    int lastStepPlane;
    bool checkboxPlaneState;
    /*
   * Counter variable to store the old m_sliderValueU; 
   */
    double m_sliderValueU_old;
    /*
   * Counter variable to store the old m_sliderValueV; 
   */
    double m_sliderValueV_old;
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
   * Variable to store the current Surface
   * for m_formula.png: formula field
   */
    int m_formula;
    /*
   * Variable to store the current u-slider value
   */
    double m_sliderValueU;
    /*
   * Variable to store the current v-slider value
   */
    double m_sliderValueV;
    /*
   * Variable to store the current A-slider value(Radius)
   */
    double m_sliderValueA;
    double m_sliderValueA_;
    double sliderStartValueA;
    /*
   * Variable to store the current B-slider value(Länge)
   */
    double m_sliderValueB;
    double m_sliderValueB_;
    double sliderStartValueB;
    /*
   * Variable to store the current C-slider value(Höhe)
   */
    double m_sliderValueC;
    double m_sliderValueC_;
    double sliderStartValueC;
    /*
   * Variable to store the current Ua-slider value(Anfang des U-Parameterbereichs)
   */
    double m_sliderValueUa;
    /*
   * Variable to store the current Ue-slider value(Ende des U-Parameterbereichs)
   */
    double m_sliderValueUe;
    /*
   * Variable to store the current Va-slider value(Anfang des V-Parameterbereichs)
   */
    double m_sliderValueVa;
    /*
   * Variable to store the current Ve-slider value(Ende des V-Parameterbereichs)
   */
    double m_sliderValueVe;
    /*
   * Variable to store the current LOD value
   */
    int m_sliderValueLOD;
    /*
   * Variable to store the old LOD value
   */
    int m_sliderValueLOD_old;
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
    * Pointer to the slider for the u parameter
    */
    coSliderMenuItem *m_pSliderMenuU;

    /*
   * Pointer to the slider for the v parameter
   */
    coSliderMenuItem *m_pSliderMenuV;
    /*
   * Pointer to the slider for the A parameter(radius)
   */
    coSliderMenuItem *m_pSliderMenuA;
    coSliderMenuItem *m_pSliderMenuA_;
    /*
   * Pointer to the slider for the B parameter(length)
   */
    coSliderMenuItem *m_pSliderMenuB;
    coSliderMenuItem *m_pSliderMenuB_;
    /*
   * Pointer to the slider for the C parameter(hight)
   */
    coSliderMenuItem *m_pSliderMenuC;
    coSliderMenuItem *m_pSliderMenuC_;
    /*
   * Pointer to the slider for the U-beginning parameter
   */
    coSliderMenuItem *m_pSliderMenuUa;
    /*
   * Pointer to the slider for the U-ending parameter
   */
    coSliderMenuItem *m_pSliderMenuUe;
    /*
   * Pointer to the slider for the V-beginning parameter
   */
    coSliderMenuItem *m_pSliderMenuVa;
    /*
   * Pointer to the slider for the V-ending parameter
   */
    coSliderMenuItem *m_pSliderMenuVe;
    /*
   * Pointer to the slider for the LOD parameter
   */
    coSliderMenuItem *m_pSliderMenuLOD;
    /*
   * Pointer to the button for the empty line
   */
    coLabelMenuItem *m_pButtonMenuLeerzeile; //aendern Name Button in Label
    /*
   * Pointer to the button for the Underline
   */
    coLabelMenuItem *m_pLabelMenuUnterstreichen1;
    coLabelMenuItem *m_pLabelMenuUnterstreichen2;
    coLabelMenuItem *m_pLabelMenuUnterstreichen3;
    coLabelMenuItem *m_pLabelMenuUnterstreichen4;
    /*
   * Pointer to the button for the Surfmode of the surface
   */
    coLabelMenuItem *m_pButtonMenuDarst;
    /*
   * Pointer to the button for the different Surfmodes
   */
    coButtonMenuItem *m_pButtonMenuPoints;
    coButtonMenuItem *m_pButtonMenuLines;
    coButtonMenuItem *m_pButtonMenuTriangles;
    coButtonMenuItem *m_pButtonMenuQuads;
    coButtonMenuItem *m_pButtonMenuShade;
    coButtonMenuItem *m_pButtonMenuTransparent;
    coButtonMenuItem *m_pButtonMenuPoints_;
    coButtonMenuItem *m_pButtonMenuLines_;
    coButtonMenuItem *m_pButtonMenuTriangles_;
    coButtonMenuItem *m_pButtonMenuQuads_;
    coButtonMenuItem *m_pButtonMenuShade_;
    coButtonMenuItem *m_pButtonMenuTransparent_;
    coCheckboxMenuItem *m_pCheckboxMenuGauss;
    coCheckboxMenuItem *m_pCheckboxMenuMean;
    coLabelMenuItem *m_pButtonMenuDarstellungAend;
    /* 
  *Pointer to the button for the choise of surfaces
   */
    coLabelMenuItem *m_pButtonMenuSurface;
    /* 
  *Pointer to the button for the different surfaces
   */
    coButtonMenuItem *m_pButtonMenuKegel;
    coButtonMenuItem *m_pButtonMenuKugel;
    coButtonMenuItem *m_pButtonMenuMoebius;
    coButtonMenuItem *m_pButtonMenuParaboloid;
    coButtonMenuItem *m_pButtonMenuZylinder;
    coButtonMenuItem *m_pButtonMenuBonan;
    coButtonMenuItem *m_pButtonMenuBoy;
    coButtonMenuItem *m_pButtonMenuCrossCap;
    coButtonMenuItem *m_pButtonMenuDini;
    coButtonMenuItem *m_pButtonMenuEnneper;
    coButtonMenuItem *m_pButtonMenuHelicalTorus;
    coButtonMenuItem *m_pButtonMenuKatenoid;
    coButtonMenuItem *m_pButtonMenuKlein;
    coButtonMenuItem *m_pButtonMenuKuen;
    coButtonMenuItem *m_pButtonMenuPluecker;
    coButtonMenuItem *m_pButtonMenuPseudoSphere;
    coButtonMenuItem *m_pButtonMenuRevolution;
    coButtonMenuItem *m_pButtonMenuRoman;
    coButtonMenuItem *m_pButtonMenuShell;
    coButtonMenuItem *m_pButtonMenuSnake;
    coButtonMenuItem *m_pButtonMenuTrumpet;
    coButtonMenuItem *m_pButtonMenuTwistedSphere;
    /* 
  *Pointer to the button for the Cons-Types(Flaechenkurven)
   */
    coLabelMenuItem *m_pButtonMenuCons;
    /* 
  *Pointer to the button for the different Cons-Types
   */
    coButtonMenuItem *m_pButtonMenuNothing;
    coButtonMenuItem *m_pButtonMenuUCenter;
    coButtonMenuItem *m_pButtonMenuVCenter;
    coButtonMenuItem *m_pButtonMenuDiagonal;
    coButtonMenuItem *m_pButtonMenuTriangle;
    coButtonMenuItem *m_pButtonMenuEllipse;
    coButtonMenuItem *m_pButtonMenuSquare;
    coButtonMenuItem *m_pButtonMenuNatbound;
    /* 
  *Pointer to the button for the different textures
   */
    coButtonMenuItem *m_pButtonMenuTextur1;
    coButtonMenuItem *m_pButtonMenuTextur2;
    coButtonMenuItem *m_pButtonMenuTextur1_;
    coButtonMenuItem *m_pButtonMenuTextur2_;
    /* 
  *Pointer to the button for the animation: shooting 6 spheres
   */
    coButtonMenuItem *m_pButtonMenuAnimationSphere;
    coButtonMenuItem *m_pButtonMenuAnimationOff;
    /* 
  *Pointer to the button for the NormalschnittAnimation...
   */
    coButtonMenuItem *m_pButtonMenuNormalschnitt;
    coCheckboxMenuItem *m_pCheckboxMenuHauptKrRichtungen;
    coCheckboxMenuItem *m_pCheckboxMenuHauptKr;
    coCheckboxMenuItem *m_pCheckboxMenuSchmiegTang;
    coButtonMenuItem *m_pButtonMenuSchiefschnitt;
    coCheckboxMenuItem *m_pCheckboxMenuDarstMeusnierKugel;
    coCheckboxMenuItem *m_pCheckboxMenuTransparent;
    coButtonMenuItem *m_pButtonMenuClearNormalschnitt;
    /* 
  *Pointer to the checkbox for disabling the plane
   */
    coCheckboxMenuItem *m_pCheckboxMenuPlane;
    coCheckboxMenuItem *m_pCheckboxMenuPlane2;
    coCheckboxMenuItem *m_pCheckboxMenuPlane3;
    /* 
  *Pointer to the checkbox for disabling the bound of the surface(Flaechenrand)
   */
    coCheckboxMenuItem *m_pCheckboxMenuNatbound;
    /* 
   *Pointer to the checkbox for disabling the normals
   */
    coCheckboxMenuItem *m_pCheckboxMenuNormals;
    //Labels for Hauptkruemmung1/2
    coVRLabel *labelKr1;
    coVRLabel *labelKr2;
    //fuer Formelanzeige
    coMovableBackgroundMenuItem *imageItemList_;
    coRowMenu *imageMenu_; // submenu
    /****** methods ******/
    Node *Get_Node_byName(const std::string searchName);

    void ResetSliderUV();
    /*
   * Creates the slider menu for the directrices
   *
   * return:       void
   */
    void createMenu();
    void createMenu3();

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
    //clear the Normalschnittanimation
    void clearNormalAnim();
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
    void menuReleaseEvent(coMenuItem *iMenuItem);
    void menuEvent(coMenuItem *iMenuItem);
    void setMenuVisible(int step);
    coNavInteraction *interactionA; //Button A for selecting a point
};
#endif /* PARAMETRICSURFACES_H_ */
