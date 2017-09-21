/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TriangleBezierSurfaces_H_
#define TriangleBezierSurfaces_H_

#include <cover/coVRPlugin.h>
#include "InteractionPoint.h"
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <cover/coVRLabel.h>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/Array>
#include <osg/Matrix>

using namespace osg;
using namespace osgViewer;
using namespace covise;
using namespace opencover;

class TriangleBezierSurfaces : public coVRPlugin, public coMenuListener
{
public:
    /****** variables ******/
    /*
   * Static member variable as pointer for the plugin.
   */
    static TriangleBezierSurfaces *plugin;
    TriangleBezierSurfaces();

    virtual ~TriangleBezierSurfaces();
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
    // ------------------------------------
    void updateDegrees();

    // Step 1: Bernsteinpolynome vom Grad 1 und 2
    void step1();

    // Step 2: Bernsteinpolynome vom Grad 3
    void step2();

    // Step 3: beliebiges Bernsteinpolynom
    void step3();

    // Step 4: Bezierflaeche
    void step4();

    // Step 5: Graderhoehung
    void step5();

    // Step 6: Casteljau-Plot
    void step6();

    int segmentCounter;

private:
    //Labels for Hauptkruemmung1/2
    coVRLabel *label_b003;
    coVRLabel *label_b030;
    coVRLabel *label_b300;
    coVRLabel *label_b012;
    coVRLabel *label_b021;
    coVRLabel *label_b120;
    coVRLabel *label_b210;
    coVRLabel *label_b102;
    coVRLabel *label_b201;
    coVRLabel *label_b111;

    void step6_interact();
    void clear_ControlPoints();
    int interactPlot; //zeigt Stelle in controlPoints für bewegten Interaktor
    std::vector<InteractionPoint *> controlPoints;

    std::string HfT_int_to_string(int d);

    osg::ref_ptr<osg::Group> sw;
    osg::ref_ptr<osg::MatrixTransform> mt;

    //osg::ref_ptr<osg::Geode> roteKugel;
    bool changeShowSegm;
    bool changeGenauigkeit;
    bool changeShowFlaeche;
    bool changeGrad;
    int precision;

    float linienstaerke;

    // Farben
    Vec4 black, white, red, lime, orange, blue, yellow, aqua, fuchsia, gold, gold2, silver, purple, olive, teal,
        grey, navy, maroon, green, pink, brown, gruen_grad, purple_segm;
    Vec4 color003, color102, color201, color300, color012, color111, color210, color021, color120, color030;

    // Fakultaet-Methoden
    int fak(int k);
    int fakRek(int k);
    void setMaterial_line(Geometry *geom2, Vec4 color);
    void setMaterial(Geometry *geom, Vec4 color);
    // Bernstein-Methoden
    float bernsteinBary(unsigned int grad, unsigned int i, unsigned int j, unsigned int k, float u, float v, float w);
    osg::ref_ptr<osg::Geode> bernsteinBaryPlot(unsigned int grad, unsigned int i, unsigned int j, unsigned int k, unsigned int unterteilungen, Vec4 color);

    // Casteljau-Methoden
    Vec3 triangleCasteljau(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w);

    // Hilfsmethoden zur Erstellung von Punkten
    osg::ref_ptr<osg::Geode> createSphere();
    osg::ref_ptr<osg::Geode> createSphere(float radius);
    osg::ref_ptr<osg::Geode> createSphere(float radius, Vec4 color);
    osg::ref_ptr<osg::Geode> createSphere(float radius, Vec4 color, Vec3 center);

    osg::ref_ptr<osg::Geode> createSpheres(float radius, Vec4 color, Vec3Array *wo);

    // ------------------------------------------
    // Step 1: Bernsteinpolynome vom Grad 1 und 2
    // ------------------------------------------

    void grad1();
    bool showGrad1;

    osg::ref_ptr<osg::Group> grad1group;

    osg::ref_ptr<osg::Geode> bernstein100;
    osg::ref_ptr<MatrixTransform> v100;

    osg::ref_ptr<osg::Geode> bernstein010;
    osg::ref_ptr<MatrixTransform> v010;

    osg::ref_ptr<osg::Geode> bernstein001;
    osg::ref_ptr<MatrixTransform> v001;

    void grad2();
    bool showGrad2;

    osg::ref_ptr<osg::Group> grad2group;

    osg::ref_ptr<osg::Geode> bernstein200;
    osg::ref_ptr<MatrixTransform> v200;

    osg::ref_ptr<osg::Geode> bernstein020;
    osg::ref_ptr<MatrixTransform> v020;

    osg::ref_ptr<osg::Geode> bernstein002;
    osg::ref_ptr<MatrixTransform> v002;

    osg::ref_ptr<osg::Geode> bernstein110;
    osg::ref_ptr<MatrixTransform> v110;

    osg::ref_ptr<osg::Geode> bernstein101;
    osg::ref_ptr<MatrixTransform> v101;

    osg::ref_ptr<osg::Geode> bernstein011;
    osg::ref_ptr<MatrixTransform> v011;

    osg::ref_ptr<Geode> trennlinie;
    void initializeTrennlinie();
    void checkTrennlinie();

    // ------------------------------------------
    // Step 2: Bernsteinpolynome vom Grad 3
    // ------------------------------------------

    void grad3(int genauigkeit);

    osg::ref_ptr<osg::Group> grad3group;

    osg::ref_ptr<osg::Geode> bernstein300;
    osg::ref_ptr<MatrixTransform> v300;

    osg::ref_ptr<osg::Geode> bernstein030;
    osg::ref_ptr<MatrixTransform> v030;

    osg::ref_ptr<osg::Geode> bernstein003;
    osg::ref_ptr<MatrixTransform> v003;

    osg::ref_ptr<osg::Geode> bernstein012;
    osg::ref_ptr<MatrixTransform> v012;

    osg::ref_ptr<osg::Geode> bernstein021;
    osg::ref_ptr<MatrixTransform> v021;

    osg::ref_ptr<osg::Geode> bernstein102;
    osg::ref_ptr<MatrixTransform> v102;

    osg::ref_ptr<osg::Geode> bernstein201;
    osg::ref_ptr<MatrixTransform> v201;

    osg::ref_ptr<osg::Geode> bernstein210;
    osg::ref_ptr<MatrixTransform> v210;

    osg::ref_ptr<osg::Geode> bernstein120;
    osg::ref_ptr<MatrixTransform> v120;

    osg::ref_ptr<osg::Geode> bernstein111;
    osg::ref_ptr<MatrixTransform> v111;

    // ------------------------------------------
    // Step 3: beliebiges Bernsteinpolynom
    // ------------------------------------------

    std::string nString, iString, jString, kString;
    unsigned int nInt, iInt, jInt, kInt;

    osg::ref_ptr<osg::Geode> bernstein;

    // ------------------------------------------
    // Step 4: Bezierflaeche
    // ------------------------------------------

    int colorSelectorMeshIndex;
    int colorSelectorSurfaceIndex;
    Vec4 OrgColorSurface, OrgColorMesh;
    Vec4Array *colorArray;

    bool step4showMesh, step4showSurface, step4showCaption;

    Vec3 getSurfacePoint(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w);

    osg::ref_ptr<osg::Group> bezierSurfacePlot(unsigned int grad, Vec3Array *bezPoints, unsigned int unterteilungen, Vec4 surfaceColor, Vec4 meshColor, bool showSurface, bool showMesh, bool showCaption, bool useCasteljau);

    osg::ref_ptr<osg::Geode> bezierSurfPointsPlot(unsigned int grad, Vec3Array *bezPoints, unsigned int unterteilungen, Vec4 surfaceColor, bool useCasteljau);

    osg::ref_ptr<osg::Geode> bezierNetPlot(unsigned int grad, Vec3Array *bezPoints, Vec4 meshColor);

    osg::ref_ptr<osg::Group> bezierPointsPlot(unsigned int grad, Vec3Array *bezPoints, Vec4 meshColor);

    Vec3 step4b003, step4b102, step4b201, step4b300, step4b012, step4b111, step4b210, step4b021, step4b120, step4b030;

    //nicht Vec3Array*!!
    //da nicht merkt, dass noch von wo anders darauf gezeigt wird,
    //denkt man braucht es nicht mehr
    //-> setzt selbststaendig das Array auf size=0.
    //->osg::ref_ptr, dieser erkennt alle Zeiger
    osg::ref_ptr<Vec3Array> step4BPgrad3;

    osg::ref_ptr<osg::Group> step4BSgrad3;

    // ------------------------------------------
    // Step 5: Graderhoehung
    // ------------------------------------------

    Vec3Array *degreeElevation(unsigned int gradVorErhoehung, Vec3Array *bezpoints);
    void selectDegree(unsigned int grad);

    int step5degree;
    std::string step5string;

    bool step5showMesh, step5showSurface, step5showCaption, step5showOrigin;

    Vec3 step5b002, step5b101, step5b200, step5b011, step5b110, step5b020;

    //nicht Vec3Array*!!
    //da nicht merkt, dass noch von wo anders darauf gezeigt wird,
    //denkt man braucht es nicht mehr
    //-> setzt selbststaendig das Array auf size=0.
    //->osg::ref_ptr, dieser erkennt alle Zeiger
    osg::ref_ptr<Vec3Array> step5BPgrad2;
    Vec3Array *step5BPgrad3;
    Vec3Array *step5BPgrad4;
    Vec3Array *step5BPgrad5;
    Vec3Array *step5BPgrad6;
    Vec3Array *step5BPgrad7;
    Vec3Array *step5BPgrad8;
    Vec3Array *step5BPgrad9;
    Vec3Array *step5BPgrad10;

    Vec3Array *currentPoints;

    osg::ref_ptr<osg::Group> step5BSgrad2;
    osg::ref_ptr<osg::Group> step5BSgrad3;
    osg::ref_ptr<osg::Group> step5BSgrad4;
    osg::ref_ptr<osg::Group> step5BSgrad5;
    osg::ref_ptr<osg::Group> step5BSgrad6;
    osg::ref_ptr<osg::Group> step5BSgrad7;
    osg::ref_ptr<osg::Group> step5BSgrad8;
    osg::ref_ptr<osg::Group> step5BSgrad9;
    osg::ref_ptr<osg::Group> step5BSgrad10;

    osg::ref_ptr<osg::Group> currentDegree;

    // ------------------------------------------
    // Step 6: Casteljau-Plot
    // ------------------------------------------

    unsigned int step6casteljauSchritt;
    std::string step6string, step6uString, step6vString, step6wString;

    unsigned int step6degree;

    bool step6showSegmentation, step6allowSegmentation, casteljauIsOn;
    osg::ref_ptr<osg::Group> step6BSsegmentation;

    float step6u, step6v, step6w;

    Vec3 step6b003, step6b102, step6b201, step6b300, step6b012, step6b111, step6b210, step6b021, step6b120, step6b030;

    Vec3Array *step6BPgrad3;
    Vec3Array *step6BPgrad4;

    osg::ref_ptr<osg::Group> step6BSgrad4, mp_seg13, mp_seg23, mp_seg33;
    osg::ref_ptr<osg::Group> casteljauGroup;

    std::vector<std::vector<std::vector<std::vector<Vec3> > > > triangleCasteljauPoints(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w);

    osg::ref_ptr<osg::Group> triangleCasteljauPlot(unsigned int grad, unsigned int schritt, Vec3Array *bezPoints, float u, float v, float w);

    //fuer schwarzen Flaechenpunkt
    void Casteljau_FlaechenPkt();

    float runden(float wert);
    bool sliderChanged; //for u,v,w Slider

    void showCasteljauNetz();
    void Menu_Schritt();
    void Menu_Segment1();
    void Menu_Segment2();
    void Menu_Segment3();

    void createMenu();
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

    ref_ptr<osg::Geode> flaechenPkt;
    /*
	* Pointer to the menu
	*/
    coRowMenu *m_pObjectMenu1;
    /* 
	*Pointer to the button for drawing the steps of the Casteljau-Algorithmus
	*/
    coButtonMenuItem *m_pButtonMenuSchritt;
    /*
	* Pointer to the slider for the Level of Detail
	*/
    coSliderMenuItem *m_pSliderMenuGenauigkeit;
    /*
	* Pointer to the slider for the value n
	*/
    coSliderMenuItem *m_pSliderMenu_n;
    /*
	* Pointer to the slider for the value i
	*/
    coSliderMenuItem *m_pSliderMenu_i;
    /*
	* Pointer to the slider for the value j
	*/
    coSliderMenuItem *m_pSliderMenu_j;
    /*
	* Pointer to the slider for the value k
	*/
    coSliderMenuItem *m_pSliderMenu_k;
    /*
	* Pointer to the slider for the polynomial degree of the Triangle Bezier Surface
	*/
    coSliderMenuItem *m_pSliderMenuGrad;
    /*
	* Pointer to the slider for the u-Parameter
	*/
    coSliderMenuItem *m_pSliderMenu_u;
    /*
	* Pointer to the slider for the v-Parameter
	*/
    coSliderMenuItem *m_pSliderMenu_v;
    /*
	* Pointer to the slider for the w-Parameter
	*/
    coSliderMenuItem *m_pSliderMenu_w;
    /*
	*Pointer to the checkbox for enabling the Bernstein polynomials of polynomial degree 1
	*/
    coCheckboxMenuItem *m_pCheckboxMenuGrad1;
    /*
	*Pointer to the checkbox for enabling the Bernstein polynomials of polynomial degree 2
	*/
    coCheckboxMenuItem *m_pCheckboxMenuGrad2;
    /*
	*Pointer to the checkbox for enabling the mesh of the Triangle Bezier Surface
	*/
    coCheckboxMenuItem *m_pCheckboxMenuNetz;
    /*
	*Pointer to the checkbox for enabling the originating mesh of the Triangle Bezier Surface
	*/
    coCheckboxMenuItem *m_pCheckboxMenuUrsprungsnetz;
    /*
	*Pointer to the checkbox for enabling the surface of the Triangle Bezier Surface
	*/
    coCheckboxMenuItem *m_pCheckboxMenuFlaeche;
    /*
	*Pointer to the checkbox for enabling the labels of the Triangle Bezier Surface
	*/
    coCheckboxMenuItem *m_pCheckboxMenuLabels;
    /*
	*Pointer to the checkbox for enabling the segment 1 
	*/
    coCheckboxMenuItem *m_pCheckboxMenuSegment1;
    /*
	*Pointer to the checkbox for enabling the segment 2 
	*/
    coCheckboxMenuItem *m_pCheckboxMenuSegment2;
    /*
	*Pointer to the checkbox for enabling the segment 3 
	*/
    coCheckboxMenuItem *m_pCheckboxMenuSegment3;
    /*
	*Pointer to the checkbox for enabling the Casteljau-mesh
	*/
    coCheckboxMenuItem *m_pCheckboxMenuCasteljauNetz;
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
};
#endif /* TriangleBezierSurfaces_H_ */
