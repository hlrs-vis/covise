// Sa abend eingecheckt
// Do 17:30 eingecheckt
// Di abend eingecheckt
/****************************************************************************\
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Star Region Interactor base class                           **
 **                                                                          **
 **                                                                          **
 ** Author: D. Rainer                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** November-99                                                              **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#ifndef _STAR_REGION_INTERACTOR_H
#define _STAR_REGION_INTERACTOR_H
#ifdef WIN32
#include <winsock2.h>
#endif

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Node>
#include <osg/ShapeDrawable>

#include <osgText/Text>

#include <kernel/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;


#include <string>

//    scene
//      |
// scene xform
//      |
// scene scale
//      |
//  objectsRoot
//      |
// .............
//      |
//   worldDCS  // transform interactor from object coodinates to world coordinates
//      |
// .............
//
// base class for all interactors
// interactors are modeled with normal = -yaxis
// (facing the default view direction)

class StarRegionInteractor
{
 protected:
   osg::ref_ptr<osg::Group>           objectsRoot;
   osg::ref_ptr<osg::MatrixTransform> worldDCS;                            // position/orientation/scale in world
   osg::ref_ptr<osg::MatrixTransform> regionDCS;                           // rotate arrow

   osg::Vec3f interPos, interNormal;               // position and direction of interactor
   osg::Vec3f defAxis;                             // interactor default axis  = -y
   float interScale;

   std::string paramName;
   //osg::ref_ptr<osg::ShapeDrawable> sensableGeoset;                     // not all interactor geometry is pick/touch/isect sensable
   //pfHighlight *intersectedHl, *selectedHl;
   int isected;
   int selected;
   osg::ref_ptr<osg::StateSet> defGeoState;
   osg::ref_ptr<osg::StateSet> unlightedGeoState;
   //pfFont *font;

   osg::StateSet *loadDefaultGeostate();
   osg::StateSet *loadUnlightedGeostate();
//    pfFont *loadFont();
   int debug;
   osg::Matrixf localMat;
   osg::Matrixf oldMat;
   osg::Matrixf thisMat;
   bool bVisible;

 public:
   // interactor types
   enum{ None = 0, Vector, Scalar};

   // constructor called in addFeedback
   StarRegionInteractor(const std::string & name, const osg::Vec3f & pos, const osg::Vec3f & normal, const osg::Vec3f & local, float size, int debug);

   virtual void updateScale();

   // ...
   const std::string & getParamName() {
      return paramName;
   }

   // ...
   virtual ~StarRegionInteractor();

   // return type
   virtual int getType() const;

   virtual void setLocal(const osg::Vec3f & l);
   virtual void setCenter(const osg::Vec3f & p);

   // called in addFeedback or preFrame
   virtual void setValue(float v1, float v2, float v3) = 0;

   virtual void setLocalValue(float v1, float v2, float v3, const osg::Matrixf & m)
   {
      (void) v1;(void) v2;(void) v3;(void) m;
   }

   // called in addFeedback or preFrame
   virtual void getValue(float *v1, float *v2, float *v3) = 0;

   // show interactor geometry
   virtual void show();

   // hide interactor geometry
   virtual void hide();

   // set the normal appearance
   void setNormalHighLight();

   // set the appearance if intersected
   void setIntersectedHighLight();

   // set the appearance if selected
   void setSelectedHighLight();

   // ...
   int isIsected() const
   {
      return isected;
   }

   // ...
   int isSelected() const
   {
      return selected;
   }

   // ...
   osg::Node * getSensableNode()
   {
      return(worldDCS.get());
   }

   /// update transformation of the vertex (for AR interface)
   void setTransform(const osg::Matrixf & m);

   /// true, if marker became visible
   bool becameVisible() const;

};

//          .............  Objektkoordinaten
//               |
//            worldDCS
//               |
//          .............  Regionkoordinaten
//               |
//            regionDCS
//               |                 Interactor rotiert
//           arrow rot dcs
//            |         |
//      arrow group   text transl dcs      Interaktor
//                      |
//                   text billboard dcs
//                      |
//                    text
//
class StarRegionVectorInteractor: public StarRegionInteractor
{
 private:

   osg::ref_ptr<osg::MatrixTransform> arrowRotDCS;                         // rotate arrow
   osg::ref_ptr<osg::MatrixTransform> textBillboardDCS;
   osg::ref_ptr<osg::MatrixTransform> textTransDCS;
   osg::ref_ptr<osg::Group> arrowGroup;                        // arrow geometry
   //pfText *labels;

   osg::Matrixf oldHandMat;

   osg::Vec3f currentAxis;
   osg::Vec3f currentVector;

   float vmag;

   osg::Group * loadArrow();
   //pfText *createLabels(float vx, float vy, float vz);
   //void updateLabels();

   //static int preAppCallback(pfTraverser *_trav, void *_userData);
   osg::Matrix getAllTransformations(osg::Node *node);
   osg::Vec3f projectPointToPlane(osg::Vec3f point, osg::Vec3f planeNormal, osg::Vec3f planePoint);
   void createLocalAxis();


 public:

   StarRegionVectorInteractor(const std::string & name,
                              const osg::Vec3f & pos, const osg::Vec3f & normal,
                              const osg::Vec3f & local,
                              float size, int debug, float v1, float v2, float v3);
   virtual ~StarRegionVectorInteractor();

   // update interactor DCS and text
   virtual void setValue(float v1, float v2, float v3);
   virtual void setLocalValue(float v1, float v2, float v3, const osg::Matrixf & m);
   virtual void getValue(float *v1, float *v2, float *v3);

   void startInteraction(const osg::Matrixf & initHandMat);
   void doInteraction(const osg::Matrixf & currentHandMa);
   //void preApp(pfTraverser *trav);
   //void preApp1(pfTraverser *trav);

   // ...
   float getMagnitude() const
   {
      return vmag;
   }

   // return the type
   virtual int getType() const;

   // return the current vector (object coordinates)
   osg::Vec3f getCurrentAxis() const
   {
      return currentAxis;
   }
};

//  dial interactor
//                                   .............
//                                         |
//                                      worldDCS
//                                         |
//                                   .............
//                                         |
//                      ---------------------------------------
//                      |                                     |
//                    dial group                       rotation dcs
//    |     |     |     |      |    |                  |       |
//   txt  txt   line  line  line  circle       button geode  buttonmarker geode
//
// modeled with axis in -y direction
// with minium at 0 and maximum at 360
//

class StarRegionScalarInteractor : public StarRegionInteractor
{

 private:
   osg::ref_ptr<osg::Group> dial;                              // labels
   osg::ref_ptr<osg::Geode> button;
   osg::ref_ptr<osg::Group> buttonMarker;            // button and marker for actual position
   //pfText * movingLabel;                       // label which moves with the marker
   osg::ref_ptr<osg::MatrixTransform> rotationDCS;                         // for rotating the button and the marker
   osg::Vec3f origin, zaxis;
   float minValue, maxValue, currentValue, minAngle, maxAngle;
   //pfText *currentLabel, *minLabel, *maxLabel;

   osg::Group * makeDial();
   //pfText *makeLabel(float angle, float value);
   //pfText *makeCurrentLabel();
   //void updateLabels();
   //pfText *makeTitle(char *name);

   osg::Geode * makeLine(float angle);
   osg::Geode * makePanel();
   osg::Geode *loadButton();
   osg::Group *loadButtonMarker();
   osg::Matrixf oldHandMat;
   float currentAngle;
   float regionRadius;
   int dindex;

 public:
   // constructor for scalar interactor
   // with built in geometry
   // panel is a quad with edge lenghts 2
   // dial is a cylinder with radius 1, axis from [0 0 0] to [0 -1 0]
   // pos, normal used to place it in the world coordinate system
   // size scales the interactor
   StarRegionScalarInteractor(const char * name,
                              const osg::Vec3f pos, const osg::Vec3f normal,
                              const osg::Vec3f local,
                              float size, int debug,
                              float minValue, float maxValue, float currentValue,
                              float minAngle, float maxAngle, float regionRadius, int index);

   // constructor for scalar interactor
   // geometry fpr panel and button=marker loaded from file
   StarRegionScalarInteractor(const char * name,
                              const std::string & dialGeometryFile,
                              const std::string & buttonGeometryFile,
                              const osg::Vec3f & pos, const osg::Vec3f normal,
                              float size);

   // destructor
   virtual ~StarRegionScalarInteractor();

   // set minimum, maximum and current value of dial
   virtual void setValue(float min, float max, float value);

   // get the current min, max, value
   virtual void getValue(float *min, float *max, float *value);

   // set value through two rotation matrices
   //void setRotation(pfMatrix initMat, pfMatrix currentMat);
   void setRotation(const osg::Matrixf currentHandMat);
   void startRotation(const osg::Matrixf initHandMat);
   void endRotation();

   virtual int getType() const;
   void updateScale();
   void setRadius(float r)
   {
      regionRadius=r;
   }
};
#endif
