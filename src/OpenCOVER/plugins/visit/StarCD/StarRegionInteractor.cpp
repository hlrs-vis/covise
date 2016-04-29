#include <util/common.h>

#include "StarRegionInteractor.h"
#include <cmath>

#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <OpenVRUI/osg/mathUtils.h>

StarRegionInteractor::StarRegionInteractor (const std::string & name,
      const osg::Vec3f & p,
      const osg::Vec3f & n,
      const osg::Vec3 & l, float s, int debug)
{

   this->debug = debug;

   if (debug)
   {
      fprintf (stderr,"\nStarRegionInteractor::StarRegionInteractor name=[%s]\n", name.c_str());
      fprintf (stderr,"\tpos=[%f %f %f]\n", p[0], p[1], p[2]);
      fprintf (stderr,"\tnormal=[%f %f %f]\n", n[0], n[1], n[2]);
   }

   osg::Matrixf scaleMat, rotMat, offsetMat;

   paramName = name;

   interPos = p;
   interNormal = n;
   interScale = s;

   defAxis.set (0.0, -1.0, 0.0);

   // transform geometry to it's position in the world
   rotMat.makeRotate (defAxis, interNormal);
   rotMat.makeIdentity();
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());
   offsetMat.mult (scaleMat, rotMat);

   for (int ctr = 0; ctr < 3; ++ctr)
      offsetMat (3, ctr) = interPos[ctr];

   worldDCS = new osg::MatrixTransform();
   worldDCS->setMatrix (offsetMat);

   regionDCS = new osg::MatrixTransform();

   MAKE_EULER_MAT(localMat, l[0], l[1], l[2]);
   regionDCS->setMatrix (localMat);
   //localMat.print(0, 1, "localMat ", stderr);

//    intersectedHl = new pfHighlight();
//    intersectedHl->setMode(PFHL_FILL);
//    intersectedHl->setColor(PFHL_FGCOLOR, 0.7, 0.7, 0.7);
//    intersectedHl->setLineWidth(3.0);
//
//    selectedHl = new pfHighlight();
//    selectedHl->setMode(PFHL_FILL);
//    selectedHl->setColor(PFHL_FGCOLOR, 0.9, 0.9, 0.9);
//    selectedHl->setLineWidth(3.0);

   selected = false;
   isected = false;

   // scene graph
   objectsRoot = cover->getObjectsRoot();
   objectsRoot->addChild (worldDCS.get());

   // make it and all children invisible
   hide();

   // default appearance of the interactor
   defGeoState = loadDefaultGeostate();
   unlightedGeoState = loadUnlightedGeostate();

   //font = loadFont();

}


/// update transformation of the vertex (for AR interface)
void StarRegionInteractor::setTransform (const osg::Matrixf &m)
{
   //cerr << "oldMat:" << oldMat(3,0) << endl;
   //cerr << "m:" << m(3,0) << endl;
   if ( (oldMat (3, 0) == 10000) && (m (3, 0) != 10000))
   {
      bVisible = true;
      cerr << "true:"  << endl;
   }
   else
   {
      bVisible = false;
   }
   oldMat = m;
   if (m (0, 3) != 10000)
   {
      //cerr << "StarRegionInteractor::setTransform info: name=" << this->paramName << endl;
      osg::Matrixf t;
      osg::Matrixf m2;
      osg::Matrixf vrml;
      t.makeRotate (90,0,1,0);
      vrml.makeRotate (90,1,0,0);
      m2 = m * t *vrml;
      setLocalValue (m2 (0, 2), m2 (1, 2), m2 (2, 2), m2);
      //cerr << "Vector: "<< m(0,1) << "," << m(1,1) << "," << m(2,1) << endl;

   }
   //cerr << "Vector: "<< m[0][1] << "," << m[1][1] << "," << m[2][1] << endl;
   //cerr << "---------------------------------------------------------------------" << endl;
}


/// true, if marker bacame visible
bool StarRegionInteractor::becameVisible() const
{
   return bVisible;
}


void StarRegionInteractor::updateScale()
{
   osg::Matrixf scaleMat, rotMat, offsetMat;
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());

   rotMat.makeRotate (defAxis, interNormal);
   rotMat.makeIdentity();
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());
   offsetMat.mult (scaleMat, rotMat);
   for (int ctr = 0; ctr < 3; ++ctr)
      offsetMat (3, ctr) = interPos[ctr];

   worldDCS->setMatrix (offsetMat);

}


void StarRegionInteractor::setCenter (const osg::Vec3f & p)
{
   if (debug)
   {
      fprintf (stderr,"\nStarRegionInteractor::setCenter\n");
      fprintf (stderr,"\tpos=[%f %f %f]\n", p[0], p[1], p[2]);
   }
   osg::Matrixf scaleMat, rotMat, offsetMat;

   interPos = p;

   defAxis.set (0.0, -1.0, 0.0);

   // transform geometry to it's position in the world
   rotMat.makeRotate (defAxis, interNormal);
   rotMat.makeIdentity();
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());
   offsetMat.mult (scaleMat, rotMat);
   for (int ctr = 0; ctr < 3; ++ctr)
      offsetMat (3, ctr) = interPos[ctr];

   worldDCS->setMatrix (offsetMat);

}


void StarRegionInteractor::setLocal (const osg::Vec3f & l)
{
   osg::Matrixf localMat;
   MAKE_EULER_MAT(localMat, l[0], l[1], l[2]);
   regionDCS->setMatrix (localMat);
}


// pfFont * StarRegionInteractor::loadFont()
// {
//    pfFont *f = NULL;
//    const char *fontFileName;
//    char *fontName;
//
//    fontFileName = cover->getname("fonts/Helvetica.mf");
//    fontName = new char[strlen(fontFileName)+1];
//    strcpy(fontName, fontFileName);
//    fontName[strlen(fontName) -3] = 0;
//    fprintf(stderr,"FONT=[%s]\n", fontName);
//                                                   PFDFONT_FILLED  PFDFONT_TEXTURED
//    f = pfdLoadFont("type1", fontName, PFDFONT_FILLED);
//
//    return(f);
//
// }


int StarRegionInteractor::getType() const
{
   return None;
}


StarRegionInteractor::~StarRegionInteractor()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::~StarRegionInteractor\n");
   if (objectsRoot.valid())
   {
      if (worldDCS.valid())
      {
         objectsRoot->removeChild (worldDCS.get());
      }

   }
   /*   pfDelete(intersectedHl);
      pfDelete(selectedHl);*/
}


void StarRegionInteractor::show()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::show\n");

   // FIXME Trav mask?
   //worldDCS->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
}


void StarRegionInteractor::hide()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::hide\n");

   // FIXME Trav mask?
   //worldDCS->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
}


osg::StateSet * StarRegionInteractor::loadDefaultGeostate()
{

   osg::ref_ptr<osg::Material> material = new osg::Material();
   material->setColorMode (osg::Material::AMBIENT_AND_DIFFUSE);
   material->setAmbient (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.2f, 0.2f, 0.2f, 1.0f));
   material->setDiffuse (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.9f, 0.9f, 0.9f, 1.0f));
   material->setSpecular (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.9f, 0.9f, 0.9f, 1.0f));
   material->setEmission (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.0f, 0.0f, 0.0f, 1.0f));
   material->setShininess (osg::Material::FRONT_AND_BACK, 16.0f);

   osg::StateSet * geoState = new osg::StateSet();
   geoState->setGlobalDefaults();
   geoState->setAttributeAndModes (material.get(), osg::StateAttribute::ON);
   geoState->setMode (GL_LIGHTING,  osg::StateAttribute::ON);
   geoState->setMode (GL_BLEND,     osg::StateAttribute::ON);

   return geoState;

}


osg::StateSet * StarRegionInteractor::loadUnlightedGeostate()
{

   osg::ref_ptr<osg::Material> material = new osg::Material();
   material->setColorMode (osg::Material::AMBIENT_AND_DIFFUSE);
   material->setAmbient (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.2f, 0.2f, 0.2f, 1.0f));
   material->setDiffuse (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.9f, 0.9f, 0.9f, 1.0f));
   material->setSpecular (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.9f, 0.9f, 0.9f, 1.0f));
   material->setEmission (osg::Material::FRONT_AND_BACK, osg::Vec4 (0.0f, 0.0f, 0.0f, 1.0f));
   material->setShininess (osg::Material::FRONT_AND_BACK, 16.0f);

   osg::StateSet * geoState = new osg::StateSet();
   geoState->setGlobalDefaults();
   geoState->setAttributeAndModes (material.get(), osg::StateAttribute::ON);
   geoState->setMode (GL_LIGHTING,  osg::StateAttribute::OFF);
   geoState->setMode (GL_BLEND,     osg::StateAttribute::ON);

   return geoState;

}


void StarRegionInteractor::setIntersectedHighLight()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::setIntersectedHighLight\n");

   std::cerr << "StarRegionInteractor::setIntersectedHighLight fixme: stub" << std::endl;
   //sensableGeoset->setHlight(intersectedHl);
   isected = true;

}


void
StarRegionInteractor::setNormalHighLight()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::setNormalHighLight\n");

   std::cerr << "StarRegionInteractor::setNormalHighLight fixme: stub" << std::endl;

   //sensableGeoset->setHlight(PFHL_OFF);
   isected = false;
   selected = false;

}


void StarRegionInteractor::setSelectedHighLight()
{
   if (debug)
      fprintf (stderr,"StarRegionInteractor::setSelectedHighLight\n");

   std::cerr << "StarRegionInteractor::setSelectedHighLight fixme: stub" << std::endl;

   //sensableGeoset->setHlight(selectedHl);
   selected = true;

}


//----------------------------------------------------------------------------//

StarRegionVectorInteractor::StarRegionVectorInteractor (const std::string & name,
      const osg::Vec3f & p,
      const osg::Vec3f & n,
      const osg::Vec3f & l,
      float s, int d,
      float vx, float vy, float vz)
      : StarRegionInteractor (name, p, n, l, s, d)
{
   if (debug)
   {
      fprintf (stderr,"StarRegionVectorInteractor::StarRegionVectorInteractor\n");
      fprintf (stderr,"\tname=[%s]\n", name.c_str());
      fprintf (stderr,"\tposition=[%f %f %f]\n", p[0], p[1], p[2]);
      fprintf (stderr,"\tnormal=[%f %f %f]\n", n[0], n[1], n[2]);
      fprintf (stderr,"\tvec=[%f %f %f]\n", vx, vy, vz);
   }

   osg::Matrixf transMat;

   currentVector.set (vx, vy, vz);                // Region-Koordinaten

   // initialize current axis
   //currentAxis.set(0.0, 0.0, 1.0); // zAxis;

   arrowRotDCS = new osg::MatrixTransform();
   arrowGroup = loadArrow();

   textTransDCS = new osg::MatrixTransform();
   transMat.makeTranslate (0.0, -3.5, 0.35);
   textTransDCS->setMatrix (transMat);

   textBillboardDCS = new osg::MatrixTransform();
   std::cerr << "StarRegionVectorInteractor::<init> fixme: billboard and labels" << std::endl;

//    textBillboardDCS->setTravData(PFTRAV_APP,(void *)this);
//    textBillboardDCS->setTravMask(PFTRAV_APP,0xFFFF,PFTRAV_SELF,PF_SET);
//    textBillboardDCS->setTravFuncs(PFTRAV_APP, preAppCallback, NULL);

   //labels = createLabels(vx, vy, vz);

   worldDCS->addChild (regionDCS.get());
   regionDCS->addChild (arrowRotDCS.get());
   arrowRotDCS->addChild (arrowGroup.get());
   arrowRotDCS->addChild (textTransDCS.get());
   textTransDCS->addChild (textBillboardDCS.get());
//   textBillboardDCS->addChild(labels.get());

   // draw local coordinate system
   createLocalAxis();

   setValue (vx, vy, vz);

}


void StarRegionVectorInteractor::createLocalAxis()
{

   osg::ref_ptr<osg::Vec3Array> lc = new osg::Vec3Array (6);
   osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array (3);

   osg::ref_ptr<osg::Geometry> lineGeoset;
   osg::ref_ptr<osg::Geode> lineGeode;
   osg::ref_ptr<osg::StateSet> lineStateSet;

   (* lc) [0].set (0, 0, 0);
   (* lc) [1].set (4, 0, 0);
   (* lc) [2].set (0, 0, 0);
   (* lc) [3].set (0, 4, 0);
   (* lc) [4].set (0, 0, 0);
   (* lc) [5].set (0, 0, 4);

   (* colors) [0].set (1, 0, 0, 1);
   (* colors) [1].set (0, 1, 0, 1);
   (* colors) [2].set (0, 0, 1, 1);

   lineGeoset = new osg::Geometry();
   lineGeoset->setVertexArray (lc.get());
   lineGeoset->addPrimitiveSet (new osg::DrawArrays (osg::PrimitiveSet::LINES, 0, 6));
   lineGeoset->setColorArray (colors.get());

   lineStateSet = dynamic_cast<osg::StateSet*> (unlightedGeoState->clone (osg::CopyOp (osg::CopyOp::DEEP_COPY_ALL)));
   lineStateSet->setAttributeAndModes (new osg::LineWidth (5.0), osg::StateAttribute::ON);

   lineGeode = new osg::Geode();
   lineGeode->setStateSet (lineStateSet.get());
   lineGeode->addDrawable (lineGeoset.get());

   //FIXME Coord axis
   //regionDCS->addChild (lineGeode.get());

}

// FIXME: preAppCallback
// int StarRegionVectorInteractor::preAppCallback(pfTraverser *_trav, void *_userData)
// {
//    StarRegionVectorInteractor *i;
//    i = (StarRegionVectorInteractor *) _userData;
//    i->preApp(_trav);
//    return PFTRAV_CONT;


/*
void
StarRegionVectorInteractor::preApp(pfTraverser *trav)
{

    osg::Matrix m, minv, viewMat, bbMat;
    osg::Vec3 yaxis(0.0, 1.0, 0.0), viewPos, viewPos_o;

    //transformations from leaves to root
    m = getAllTransformations(textBillboardDCS->getParent(0));
    //fprintf(stderr,"interactor name: %s\n", paramName);
//m.print(0, 1, "all transformations", stderr);

minv.invertFull(m);

// viewer orientation
viewMat = cover->getViewerMat();
viewPos.set(viewMat[3][0], viewMat[3][1], viewMat[3][2]);

// viewer pos in text coordinates
viewPos_o.xformVec(-viewPos, minv);
viewPos_o.normalize();

// rotation from y to line_o
bbMat.makeVecRotVec(yaxis, viewPos_o);

textBillboardDCS->setMat(bbMat);

}
*/

// axial rotation billboard
// void StarRegionVectorInteractor::preApp(pfTraverser *trav)
// {
//    (void) trav;
//    pfVec3 axis(0.0, 0.0, 1.0);
//    osg::Matrix viewMat, m, minv, bbMat;
//    pfVec3 viewPos;
//    pfVec3 viewPos_o;                              // view position in object coordinates
//    pfVec3 viewPos_op;                             // projected on a plane where axis is the normal
//    pfVec3 line, yaxis_p, yaxis(0.0, 1.0, 0.0), origin(0.0, 0.0, 0.0);
//    float angle;
//    ////    float d, len;// distance view pos to plane perpendicular to axis

//    // get viewer position
//    viewMat = cover->getViewerMat();
//    viewMat.getRow(3, viewPos);
//    //fprintf(stderr,"viewPos=[%f % f% f]\n", viewPos[0], viewPos[1], viewPos[2]);

//    // get view pos in object coordinates
//    m = getAllTransformations(textBillboardDCS->getParent(0));
//    minv.invertFull(m);
//    viewPos_o.xformPt(viewPos, minv);
//    //fprintf(stderr,"viewPos_o=[%f % f% f]\n", viewPos_o[0], viewPos_o[1], viewPos_o[2]);

//    // project viewPos_o into plane normal to axis
//    //axis.normalize();
//    //d = axis[0] * viewPos_o[0] + axis[1] * viewPos_o[1] + axis[2] * viewPos_o[2];
//    //fprintf(stderr,"d=[%f]\n", d);
//    //viewPos_op = viewPos_o - d * axis;
//    //fprintf(stderr,"viewPos_op=[%f % f% f]\n", viewPos_op[0], viewPos_op[1], viewPos_op[2]);
//    viewPos_op = projectPointToPlane(viewPos_o, axis, origin);

//    // project y axis ontp plane
//    yaxis_p = projectPointToPlane(yaxis, axis, origin);

//    viewPos_op.normalize();
//    yaxis_p.normalize();
//    angle= acos(yaxis_p.dot(viewPos_op))/M_PI*180;

//    // test
//    pfVec3 tmp;
//    tmp.cross(yaxis_p,viewPos_op);
//    if(tmp.dot(axis)<0)
//       angle = -angle;

//    //fprintf(stderr,"angle=[%f]\n", angle);

//    // rotatation around axis
//    bbMat.makeRot(angle, axis[0], axis[1], axis[2]);

//    textBillboardDCS->setMat(bbMat);
// }


// pfVec3 StarRegionVectorInteractor::projectPointToPlane(pfVec3 point, pfVec3 planeNormal, pfVec3 planePoint)
// {
//    (void) planePoint;
//    float d;
//    pfVec3 point_proj, normal;

//    normal = planeNormal;
//    normal.normalize();

//    // distance point to plane
//    d = normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2];

//    point_proj = point - d * normal;

//    return(point_proj);
// }


/*
void
StarRegionVectorInteractor::preApp(pfTraverser *trav)
{
    osg::Matrix m, minv, bbMat, viewMat;

    m = getAllTransformations(textBillboardDCS->getParent(0));
    minv.invertFull(m);
    viewMat = cover->getViewerMat();
    viewMat[3][0] = viewMat[3][1] = viewMat[3][2] = 0.0;
    bbMat.mult(minv, viewMat);
textBillboardDCS->setMat(bbMat);
}
*/
/*
void
StarRegionVectorInteractor::preApp(pfTraverser *trav)
{
    pfNode *parent;
    float angle;
    pfVec3 axis(0.0, 0.0, 1.0);
    pfVec3 z(0,0,1);
    pfVec3 viewer;
    pfVec3 normal;
    pfVec3 waxis;
pfVec3 zaxnormal;
pfVec3 bbpos(0,0,0);
osg::Matrix rotMat;
osg::Matrix baseMat;

if(textBillboardDCS->getNumParents())
{
parent = textBillboardDCS->getParent(0);
baseMat = getAllTransformations1(parent);
//baseMat.print(0, 1, "baseMat: ", stderr);

// transform local z to world

cover->getViewerMat().getRow(3,viewer);
z.xformVec(z,baseMat);

// zaxis in world coordinates
waxis.xformVec(axis,baseMat);

// rotaxis in world coordinates
waxis.xformVec(axis,baseMat);
bbpos.fullXformPt(bbpos,baseMat);

// origin of billboard in wc
viewer = viewer - bbpos;
viewer.normalize();
normal.cross(viewer,waxis);
normal.normalize();

// plane normal of plane defined by rotation axis and Viewer
zaxnormal.cross(z,waxis);
zaxnormal.normalize();
angle= acos(zaxnormal.dot(normal))/M_PI*180;

// angle to rotate z/ax/Normal onto plane normal
pfVec3 tmp;
tmp.cross(zaxnormal,normal);
if(tmp.dot(waxis)<0)
angle = -angle;

rotMat.makeRot(angle,axis[0],axis[1],axis[2]);
textBillboardDCS->setMat(rotMat);

}
}
*/

osg::Matrix
 StarRegionVectorInteractor::getAllTransformations(osg::Node *node)
 {
   osg::Node *parent;
   osg::Matrix tr;
   osg::Matrix dcsMat;
   tr.makeIdentity();
   parent = node->getParent(0);
   while(parent!=NULL)
   {
      if (dynamic_cast<osg::MatrixTransform*>(parent))
      {
         dcsMat = (dynamic_cast<osg::MatrixTransform*>(parent))->getMatrix();
         tr.postMult(dcsMat);
      }
      if(parent->getNumParents())
         parent = parent->getParent(0);
      else
         parent = NULL;
   }
    return tr;
 }


StarRegionVectorInteractor::~StarRegionVectorInteractor()
{
   if (debug)
      fprintf (stderr,"StarRegionVectorInteractor::~StarRegionVectorInteractor\n");

   if (worldDCS.valid())
   {
      if (arrowRotDCS.valid())
      {
         if (arrowGroup.valid())
         {
            arrowRotDCS->removeChild (arrowGroup.get());
         }

         if (textTransDCS.valid())
         {
            if (textBillboardDCS.valid())
            {
//               if (labels.valid())
//               {
//                  textBillboardDCS->removeChild(labels.get());
//               }
               textTransDCS->removeChild (textBillboardDCS.get());
            }
            arrowRotDCS->removeChild (textTransDCS.get());
         }
         worldDCS->removeChild (arrowRotDCS.get());
      }
   }
}


int StarRegionVectorInteractor::getType() const
{
   return (StarRegionInteractor::Vector);
}


// set new velocity in interactor coordinates
void StarRegionVectorInteractor::setLocalValue (float vx, float vy, float vz, const osg::Matrixf & m)
{

   vmag = sqrt ( (currentVector[0] * currentVector[0])
                 + (currentVector[1] * currentVector[1])
                 + (currentVector[2] * currentVector[2]));
   osg::Matrixf rotMat;
   osg::Vec3f iaxis (0.0, -1.0, 0.0);                 // interactor axis, interactor is modelled in origin
   osg::Vec3f currentAxis_i;                          // values in interactor coord system
   currentAxis_i.set (vx,vy,vz);
   currentAxis_i.normalize();

   // make a rotation from the interactor axis to current axis
   //rotMat.makeVecRotVec(interNormal, currentAxis);
   rotMat.makeRotate (iaxis, currentAxis_i);
   arrowRotDCS->setMatrix (m);

   // get the current line in value coordinates
   //pfVec3 currentVector_i;
   osg::Vec3f a (0,1,0), a_c, a_v;

   osg::Matrixf vm = regionDCS->getMatrix();
   a_c = a * m;
   a_v = a_c * vm;
   //currentVector = a_v;
   currentVector = a_c;
   currentVector.normalize();
   currentVector = currentVector * vmag;
   fprintf(stderr,"currentVector = [%f %f %f]\n", currentVector[0], currentVector[1], currentVector[2]);

//   updateLabels();
}


// set new velocity in interactor coordinates
void StarRegionVectorInteractor::setValue (float vx, float vy, float vz)
{
   if (debug)
      fprintf (stderr,"StarRegionVectorInteractor::setValue: %f %f %f\n", vx, vy, vz);

   osg::Matrixf rotMat;
   osg::Vec3f iaxis (0.0, -1.0, 0.0);                 // interactor axis, interactor is modelled in origin
   osg::Vec3f currentAxis_i;                          // values in interactor coord system

   // the values come in their own coordinate system
   // worldDCS transforms from interactor to value system
   currentAxis.set (-vx, -vy, -vz);
   currentVector.set (vx, vy, vz);

   // transform the values into the interactor system
   osg::Matrixf m, minv;
   m = regionDCS->getMatrix();
   minv.invert (m);
   currentAxis_i=currentAxis;
   //currentAxis_i.xformVec(currentAxis, minv);
   currentAxis_i.normalize();

   // save the magnitude
   //vmag = currentAxis.length();

   // normalize it
   currentAxis.normalize();

   // make a rotation from the interactor axis to current axis
   //rotMat.makeVecRotVec(interNormal, currentAxis);
   rotMat.makeRotate (iaxis, currentAxis_i);
   arrowRotDCS->setMatrix (rotMat);

   //osg::Matrix test;
   //arrowRotDCS->getMat(test);
   //test.print(0, 1, "set value arrowRotDCS mat: ", stderr);

//   updateLabels();
}


void StarRegionVectorInteractor::getValue (float *vx, float *vy, float *vz)
{
   *vx = currentVector[0];
   *vy = currentVector[1];
   *vz = currentVector[2];
}


// void StarRegionVectorInteractor::updateLabels()
// {
//    // FIXME: update labels
//    pfString *strx, *stry, *strz;
//    char str[200];

//    sprintf(str, "%.2f", currentVector[0]);
//    strx = labels->getString(0);
//    strx->setString(str);

//    sprintf(str, "%.2f", currentVector[1]);
//    stry = labels->getString(1);
//    stry->setString(str);

//    sprintf(str, "%.2f", currentVector[2]);
//    strz = labels->getString(2);
//    strz->setString(str);

// }


void StarRegionVectorInteractor::startInteraction (const osg::Matrixf & initHandMat)
{
   oldHandMat = initHandMat;
   vmag = sqrt ( (currentVector[0] * currentVector[0])
                 + (currentVector[1] * currentVector[1])
                 + (currentVector[2] * currentVector[2]));

   fprintf (stderr,"startInteraction: vmag=[%f]\n", vmag);
}


void StarRegionVectorInteractor::doInteraction (const osg::Matrixf & currentHandMat)
{
   osg::Matrixf rm, mat, bm, bminv, baseMat;

   osg::Vec3f origin (0.0, 0.0, 0.0);

   // save matrix
   rm = arrowRotDCS->getMatrix();
   //rm.print(0, 1,"soInteraction arrowRotDCS mat before :", stderr);

   // transformations from leaves to root
   bm = getAllTransformations (arrowRotDCS.get());
   bminv.invert (bm);

   // old handPos world coordinates
   osg::Vec3f oldHandPos;
   oldHandPos.set (oldHandMat (3, 0), oldHandMat (3, 1), oldHandMat (3, 2));
   // oldHandPos in interactor coordinates
   osg::Vec3f oldHandPos_i;
   oldHandPos_i = oldHandPos * bminv;

   // line between oldHandPos_i and origin
   osg::Vec3f oldLine_i;
   oldLine_i = origin-oldHandPos_i;

   // current handPos in world coordinates
   osg::Vec3f currHandPos;
   currHandPos.set (currentHandMat (3, 0), currentHandMat (3, 1), currentHandMat (3, 2));

   // current handPos in interactor coordinates
   osg::Vec3f currHandPos_i;
   currHandPos_i = currHandPos * bminv;

   //line between currHandPos_i and origin
   osg::Vec3f currLine_i;
   currLine_i = origin-currHandPos_i;

   // rotation from old line to current line in interactor system
   currLine_i.normalize();
   oldLine_i.normalize();
   mat.makeRotate (oldLine_i, currLine_i);
   //mat.print(0, 1, "diff: ", stderr);

   rm.postMult (mat);
   //rm.print(0, 1,"dcs mat now :", stderr);

   arrowRotDCS->setMatrix (rm);

   // get the current line in value coordinates
   //pfVec3 currentVector_i;
   osg::Vec3f a (0,1,0), a_c, a_v;

   osg::Matrix vm;
   vm = regionDCS->getMatrix();
   a_c = a * rm;
   a_v = a_c * vm;
   //currentVector = a_v;
   currentVector = a_c;
   currentVector.normalize();
   currentVector *= vmag;
   //fprintf(stderr,"currentVector = [%f %f %f]\n", currentVector[0], currentVector[1], currentVector[2]);

//   updateLabels();

   oldHandMat = currentHandMat;
}

osg::Group *
StarRegionVectorInteractor::loadArrow()
{
   // pfdArrow
   //
   //   z
   //   |
   //   ^
   //   |
   // ------ x
   //
   osg::Vec4 *redcolor, *greycolor;
   osg::Matrix transMat, rotMat, scaleMat, mat;
   osg::ref_ptr<osg::ShapeDrawable>  coneGeoset, cyl1Geoset, cyl2Geoset;
   osg::ref_ptr<osg::Cylinder> cylinder;
   osg::ref_ptr<osg::Cone> cone;
   osg::ref_ptr<osg::TessellationHints> th = new osg::TessellationHints();
   osg::ref_ptr<osg::Geode> coneGeode, cyl1, cyl2;

   osg::ref_ptr<osg::MatrixTransform> coneTransform, cyl1Transform, cyl2Transform;

   osg::Group *group;
   /*   pfText *name=NULL;
      pfString *str;
   */
   // color red
   redcolor = new osg::Vec4 (1.0, 0.0, 0.0, 1.0);
   /*   redcolor = (osg::Vec4*)pfCalloc(1, sizeof(osg::Vec4), pfGetSharedArena());
      redcolor->set(1.0, 0.0, 0.0, 1.0);
   */
   // color yellow
   greycolor = new osg::Vec4 (0.5f, 0.5f, 0.5f, 1.0f);
   /*  greycolor = (osg::Vec4*)pfCalloc(1, sizeof(osg::Vec4), pfGetSharedArena());
      greycolor->set(0.5f, 0.5f, 0.5f, 1.0f);
   */
   // default arrow from z=0 to z=1

   /*   coneGeoset = pfdNewCone(60, pfGetSharedArena());
      coneGeoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, redcolor, NULL);
   */
   //pfdGSetColor(coneGeoset, 1.0f, 0.0f, 0.0f, 1.0f);
   // translate it so top is in origin

   cone = new osg::Cone();
   //cone.get()->setCenter (osg::Vec3 (0.0f, 0.0f, -1.0f));
   //cone.get()->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));
   transMat.makeTranslate(0.0f, 0.0f, -1.0f);

   //pfdXformGSet(coneGeoset, transMat);
      // rotate the arrow to y axis
   rotMat.makeRotate(-90.0, 1.0, 0.0, 0.0);
   //pfdXformGSet(coneGeoset, rotMat);
   // translate it so top is in origin
   scaleMat.makeScale(0.6, 1.0, 0.6);
   //pfdXformGSet(coneGeoset, scaleMat);

   coneTransform = new osg::MatrixTransform();
   coneTransform->setMatrix(scaleMat * rotMat * transMat);

   th->setCreateFrontFace (true);
   th->setCreateNormals (true);
   th->setCreateBottom (true);
   th->setCreateTop (true);
   th->setCreateBody (true);

   coneGeoset = new osg::ShapeDrawable (cone.get(), th.get());
   coneGeoset->setColor (*redcolor);
   coneGeoset->setStateSet (defGeoState.get());

   cylinder = new osg::Cylinder();
   //cylinder->setCenter (osg::Vec3 (0.0f, 0.0f, -2.0f));
   //cylinder->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));

   cyl1Geoset = new osg::ShapeDrawable (cylinder.get(), th.get());
   cyl1Geoset->setColor (*redcolor);

   transMat.makeTranslate(0.0f, 0.0f, -2.0f);
   rotMat.makeRotate(-90.0, 1.0, 0.0, 0.0);
   scaleMat.makeScale(0.3, 1.0, 0.3);

   cyl1Transform = new osg::MatrixTransform();
   cyl1Transform->setMatrix(scaleMat * rotMat * transMat);

   /*   cyl1Geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, redcolor, NULL);
      transMat.makeTranslate(0.0f, 0.0f, -2.0f);
      pfdXformGSet(cyl1Geoset, transMat);
      rotMat.makeRotate(-90.0, 1.0, 0.0, 0.0);
      pfdXformGSet(cyl1Geoset, rotMat);
      scaleMat.makeScale(0.3, 1.0, 0.3);
      pfdXformGSet(cyl1Geoset, scaleMat);
   */
   cyl1Geoset->setStateSet (defGeoState.get());


   cylinder = new osg::Cylinder();
   //cylinder->setCenter (osg::Vec3 (0.0f, 0.0f, -3.25f));
   //cylinder->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));

   cyl2Geoset = new osg::ShapeDrawable (cylinder.get(), th.get());
   cyl2Geoset->setColor (*greycolor);

   transMat.makeTranslate(0.0f, 0.0f, -3.25f);
   rotMat.makeRotate(-90.0, 1.0, 0.0, 0.0);
   scaleMat.makeScale(0.35, 0.35, 0.25);

   cyl2Transform = new osg::MatrixTransform();
   cyl2Transform->setMatrix(scaleMat * rotMat * transMat);

   /*   cyl2Geoset = pfdNewCylinder(60, pfGetSharedArena());
      cyl2Geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, greycolor, NULL);
      scaleMat.makeScale(0.35, 0.35, 0.25 );
      pfdXformGSet(cyl2Geoset, scaleMat);
      transMat.makeTranslate(0.0f, 0.0f, -3.25f);
      pfdXformGSet(cyl2Geoset, transMat);
      rotMat.makeRotate(-90.0, 1.0, 0.0, 0.0);
      pfdXformGSet(cyl2Geoset, rotMat);
   */

   cyl2Geoset->setStateSet (defGeoState.get());

   //sensableGeoset = cyl2Geoset;

   coneGeode = new osg::Geode();
   /*   fixNode(coneGeode);
      coneGeode->addGSet(coneGeoset);
   */
   coneGeode->addDrawable (coneGeoset.get());

   cyl1 = new osg::Geode();
   /*   fixNode(cyl1);
      cyl1->addGSet(cyl1Geoset);
   */
   cyl1->addDrawable (cyl1Geoset.get());

   cyl2 = new osg::Geode();
   /*   fixNode(cyl2);
      cyl2->addGSet(cyl2Geoset);
   */
   cyl2->addDrawable (cyl2Geoset.get());

   /*   if (font != NULL)
      {
         mat.makeScale(0.3, 0.3, 0.3);
         mat.postTrans(mat, 0.0, -3.505, -0.05);

         str = new pfString();
         str->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
         str->setColor(1.0, 1.0, 1.0, 1.0);
         str->setGState(unlightedGeoState);
         str->setFont(font);
         str->setMat(mat);
         str->setString(paramName);
         name = new pfText();
         fixNode(name);
         name->addString(str);

      }
   */

   coneTransform->addChild (coneGeode.get());
   cyl1Transform->addChild (cyl1.get());
   cyl2Transform->addChild (cyl2.get());


   group = new osg::Group();
//    group->addChild (coneTransform.get());
//    group->addChild (cyl1Transform.get());
//    group->addChild (cyl2Transform.get());
   //group->addChild(name);

   return (group);

}

/*
pfText *
StarRegionVectorInteractor::createLabels(float vx, float vy, float vz)
{
   (void) vx;(void) vy;(void) vz;

   pfText *text;
   osg::Matrix   mat;
   float scaleFactor;
   pfString *strx=NULL, *stry=NULL, *strz=NULL;
   scaleFactor = 0.4;

   if (font != NULL)
   {
      mat.makeScale(scaleFactor, scaleFactor, scaleFactor);
      mat.postTrans(mat, 0.0, 0.0, 0.2);

      strx = new pfString();
      strx->setMode(PFSTR_JUSTIFY, PFSTR_RIGHT);
      strx->setColor(1.0, 1.0, 0.0, 1.0);
      strx->setGState(defGeoState);
      strx->setFont(font);
      strx->setMat(mat);
      strx->setString("1.00");

      mat.makeScale(scaleFactor, scaleFactor, scaleFactor);
      mat.postTrans(mat, 0.0, 0.0, 0.0);

      stry = new pfString();
      stry->setMode(PFSTR_JUSTIFY, PFSTR_RIGHT);
      stry->setColor(1.0, 1.0, 0.0, 1.0);
      stry->setGState(defGeoState);
      stry->setFont(font);
      stry->setMat(mat);
      stry->setString("0.00");

      mat.makeScale(scaleFactor, scaleFactor, scaleFactor);
      mat.postTrans(mat, 0.0, 0.0, -0.2);

      strz = new pfString();
      strz->setMode(PFSTR_JUSTIFY, PFSTR_RIGHT);
      strz->setColor(1.0, 1.0, 0.0, 1.0);
      strz->setGState(defGeoState);
      strz->setFont(font);
      strz->setMat(mat);
      strz->setString("0.00");

   }
   else
   {
      fprintf(stderr,"ERROR in StarRegionVectorInteractor::createLabels: Cannot load font.\n");
   }

   text = new pfText();
   fixNode(text);
   text->addString(strx);
   text->addString(stry);
   text->addString(strz);

   return(text);
}
*/

//----------------------------------------------------------------------------

StarRegionScalarInteractor::StarRegionScalarInteractor (const char *name, osg::Vec3 p, osg::Vec3 n, osg::Vec3 l, float s, int d, float minValue, float maxValue, float currentValue, float minAngle, float maxAngle, float regionRadius, int index) :StarRegionInteractor (name, p, n, l, s, d)
{

   if (debug)
   {
      fprintf (stderr,"\nStarRegionScalarInteractor::StarRegionScalarInteractor\n");
      fprintf (stderr,"\tname=[%s]\n", name);
      fprintf (stderr,"\tminAngle=[%f], maxAngle=[%f]\n", minAngle, maxAngle);
      fprintf (stderr,"\tminValue=[%f], maxValue=[%f], currentValue=[%f]\n", minValue, maxValue, currentValue);
   }
   fprintf (stderr,"\tindex=[%d]\n", index);

   osg::Matrix rotMat;

   origin.set (0.0, 0.0, 0.0);
   zaxis.set (0.0, 0.0, 1.0);

   dial = NULL;
   buttonMarker = NULL;
//   currentLabel = NULL;

   this->minValue = minValue;
   this->maxValue = maxValue;
   this->currentValue = currentValue;
   this->minAngle = minAngle;
   this->maxAngle = maxAngle;
   this->regionRadius = regionRadius;
   dindex = index;

   dial = makeDial();
   //button = loadButton();
   rotationDCS = new osg::MatrixTransform();
   buttonMarker = loadButtonMarker();
//   currentLabel = makeCurrentLabel();

   worldDCS->addChild (rotationDCS.get());
   //rotationDCS->addChild(button);
   rotationDCS->addChild (buttonMarker.get());
//   rotationDCS->addChild(currentLabel.get());

   worldDCS->addChild (dial.get());

   currentAngle    = minAngle + (maxAngle-minAngle) * (currentValue-minValue) / (maxValue-minValue);

   //fprintf(stderr,"CURANGLE=%f\n", currentAngle);
   rotMat.makeRotate (currentAngle, 0.0, 1.0, 0.0);
   rotationDCS->setMatrix (rotMat);

}


StarRegionScalarInteractor::~StarRegionScalarInteractor()
{
   if (debug)
      fprintf (stderr,"StarRegionScalarInteractor::~StarRegionScalarInteractor\n");

   if (worldDCS.get())
   {

      if (rotationDCS.get())
      {
         //if (button)
         //{
         //    rotationDCS->removeChild(button);
         //    pfDelete(button);
         //}

         if (buttonMarker.get())
         {
            rotationDCS->removeChild (buttonMarker.get());
            /*            pfDelete(buttonMarker); */
         }

         worldDCS->removeChild (rotationDCS.get());
         /*         pfDelete(rotationDCS); */

      }

      if (dial.get())
      {
         for (int i = 0; i < dial->getNumChildren(); i++)
         {
            osg::Node *c = (osg::Node *) dial->getChild (i);
            dial->removeChild (c);
            /*            pfDelete(c); */

         }
         worldDCS->removeChild (dial.get());
         /*         pfDelete(dial); */
      }
   }

}


void
StarRegionScalarInteractor::setValue (float min, float max, float value)
{
   if (debug)
      fprintf (stderr, "StarRegionScalarInteractor::setValue\n");

   osg::Matrix rotMat;

   if (min<max && min<=value && max>=value)
   {
      this->minValue = min;
      this->maxValue = max;
      this->currentValue = value;

      currentAngle    = minAngle + (maxAngle-minAngle) * (currentValue-minValue) / (maxValue-minValue);

      //fprintf(stderr,"ANGLE=%f\n", currentAngle);
      rotMat.makeRotate (currentAngle, 0.0, 1.0, 0.0);
      rotationDCS->setMatrix (rotMat);
//      updateLabels();
   }
   else
   {
      fprintf (stderr,"ERROR in StarRegionScalarInteractor::setValue - wrong order: min=%f max=%f value=%f\n", min, max, value);
   }
}


void
StarRegionScalarInteractor::getValue (float *min, float *max, float *value)
{
   *min = minValue;
   *max = maxValue;
   *value = currentValue;
}


void
StarRegionScalarInteractor::setRotation (osg::Matrixf currentHandMat)
{
   //fprintf(stderr,"StarRegionScalarInteractor::setRotation\n");
   osg::Vec3 v;
   osg::Matrix rotMat, invMat, rz;
   float degree;

   //static float last_degree = 0;
   osg::Vec3 up_w, up_i, z (0, 0, 1);

   //initMat.print(0, 1, "initMat", stderr);
   //currentMat.print(0, 1, "currentMat", stderr);

   // pointer is modelled as line from origin to y
   // button marker is modelled in z
   // up_w is direction of unrotated button marker
   // if button marker would be mountet on pointer
   /*   up_w.xformVec(z, currentHandMat); */
   up_w = currentHandMat * z;

   //
   invMat.invert (oldHandMat);
   /*   up_i.xformVec(up_w, invMat); */
   up_i = invMat * up_w;

   // normalize the vector
   up_i.normalize();
   v = up_i;
   //fprintf(stderr," SCALAR MARKER: [%f %f %f]\n", v[0], v[1], v[2]);

   degree = asin (v[0]) * 180.0 / M_PI;

   //fprintf(stderr,"degree=%f\n", degree);

   if ( (currentAngle+degree) < minAngle)
   {
      degree = minAngle-currentAngle;
   }

   if ( (currentAngle+degree) > maxAngle)
   {
      degree = maxAngle-currentAngle;
   }
   rotMat = rotationDCS->getMatrix();
   osg::Matrix rm = osg::Matrix::rotate(osg::DegreesToRadians(degree), 0.0, 1.0, 0.0);
   /*   rotMat.postRotate(rotMat, degree, 0.0, 1.0, 0.0);*/
   rotMat = rotMat * rm;
   rotationDCS->setMatrix (rotMat);

   currentAngle += degree;
   //fprintf(stderr,"ANGLE=%f\n", currentAngle);

   oldHandMat = currentHandMat;

   currentValue = (maxValue-minValue) * (currentAngle-minAngle) / (maxAngle-minAngle) + minValue;
   //fprintf(stderr,"VALUE = %f\n", currentValue);

//   updateLabels();

}


void
StarRegionScalarInteractor::startRotation (osg::Matrixf initHandMat)
{
   //fprintf(stderr,"StarRegionScalarInteractor::startRotation\n");

   oldHandMat = initHandMat;
}


void
StarRegionScalarInteractor::endRotation()
{

   //feedback
   //...
}


int
StarRegionScalarInteractor::getType() const
{
   return (StarRegionInteractor::Scalar);
}

osg::Group *
StarRegionScalarInteractor::makeDial()
{

   //fprintf(stderr,"StarRegionScalarInteractor::makeDial\n");
   osg::Group *group;
   osg::Geode *panel, *minLine, *maxLine, *midLine;

   //pfText *name;

   group = new osg::Group();

   panel = makePanel();
   group->addChild (panel);

   minLine = makeLine (minAngle);
   group->addChild (minLine);

   maxLine = makeLine (maxAngle);
   group->addChild (maxLine);

   midLine = makeLine ( (maxAngle+minAngle) *0.5);
   group->addChild (midLine);

//    minLabel = makeLabel(minAngle, minValue);
//    group->addChild(minLabel);
//
//    maxLabel = makeLabel(maxAngle, maxValue);
//    group->addChild(maxLabel);

   //name = makeTitle(paramName);
   //group->addChild(name);

   //sensableGeoset = panel->getGSet(0);

   return (group);

}


osg::Geode *
StarRegionScalarInteractor::makePanel()
{
   const osg::Vec4 *grey;
   osg::ShapeDrawable *shape;
   osg::Matrix  mat, scaleMat, rotMat;
   osg::Geode *geode;

   grey = new osg::Vec4 (0.5f, 0.5f, 0.5f, 1.0f);
   /*   grey  = (osg::Vec4 *) pfCalloc(1, sizeof(osg::Vec4), pfGetSharedArena());
      grey[0].set(0.5f, 0.5f, 0.5f, 1.0f);
   */

   // panel
   osg::ref_ptr<osg::Cylinder> circle;
   osg::ref_ptr<osg::TessellationHints> th = new osg::TessellationHints();
   th->setCreateFrontFace (true);
   th->setCreateNormals (true);
   th->setCreateBottom (true);
   th->setCreateTop (true);
   th->setCreateBody (true);

   circle = new osg::Cylinder();
   //circle->setCenter (osg::Vec3 (0.0f, 0.0f, -2.0f));
   circle->setRotation (osg::Quat (90.0, 1.0, 0.0, 0.0));
   circle->setRadius(1.505f);
   circle->setHeight(0.1f);

   shape = new osg::ShapeDrawable (circle.get(), th.get());
   shape->setColor (*grey);
   /*   circle = pfdNewCircle(60, pfGetSharedArena());
      circle->setAttr(PFGS_COLOR4, PFGS_OVERALL, grey, NULL);
      pfdGSetColor(circle, 0.5f, 0.5f, 0.5f, 1.0f);

      scaleMat.makeScale(1.505, 1.505, 1.505);
      pfdXformGSet(circle, scaleMat);
      rotMat.makeRotate(90.0, 1.0, 0.0, 0.0);
      pfdXformGSet(circle, rotMat);
   */
   shape->setStateSet (defGeoState.get());

   geode = new osg::Geode();
   /*   fixNode(geode); */

   //FIXME
   //geode->addDrawable (shape);

   return (geode);

}


osg::Geode *
StarRegionScalarInteractor::makeLine (float angle)
{
   osg::ref_ptr<osg::Vec3Array> lc;                                    // line coordinates
   osg::ref_ptr<osg::Vec4Array> yellow;
   osg::Geode *line;
   osg::Geometry *geoset;
   osg::Vec3 p0 (0.0, -0.05, 1.05), p1 (0.0, -0.05, 1.3);
   osg::Matrix m;

   m.makeRotate (angle, 0.0, 1.0, 0.0);
   /*   lc = (osg::Vec3 *) pfCalloc(2, sizeof(osg::Vec3), pfGetSharedArena()); */
   lc = new osg::Vec3Array (2);

   /*   lc[0].fullXformPt(p0, m);
      lc[1].fullXformPt(p1, m);
   */
   (*lc) [0] = m * p0;
   (*lc) [1] = m * p1;

   //fprintf(stderr, "StarRegionScalarInteractor::makeLine ANGLE=%f SIN=%f COS=%f\n", angle, fsin(angle*M_PI/180), fcos(angle*M_PI/180));

   /*   yellow  = (osg::Vec4 *) pfCalloc(1, sizeof(osg::Vec4), pfGetSharedArena());
      yellow[0].set(0.1, 0.1, 0.1, 1.0);
   */
   yellow = new osg::Vec4Array (1);
   (*yellow) [0] = osg::Vec4f (0.1, 0.1, 0.1, 1.0);

   //fprintf(stderr,"line start: %f %f\n", lc[0][0], lc[0][2]);

   //fprintf(stderr,"line end: %f %f\n", lc[1][0], lc[1][2]);

   geoset = new osg::Geometry();

   geoset->setVertexArray (lc.get());
   geoset->addPrimitiveSet (new osg::DrawArrays (osg::PrimitiveSet::LINES, 0, 2));

   geoset->setColorArray (yellow.get());
   geoset->setColorBinding (osg::Geometry::BIND_OVERALL);

   osg::ref_ptr<osg::StateSet> lineStateSet =
      dynamic_cast<osg::StateSet*> (unlightedGeoState->clone (osg::CopyOp (osg::CopyOp::DEEP_COPY_ALL)));

   line = new osg::Geode();
   line->setStateSet (lineStateSet.get());
   line->addDrawable (geoset);

   /*   fixNode(line); */

   return line;

}

/*
pfText *
StarRegionScalarInteractor::makeLabel(float angle, float value)
{
   //fprintf(stderr,"StarRegionScalarInteractor::makeLabel value=[%f]\n", value);
   char str[200];
   pfText *t = NULL;
   osg::Matrix scaleMat, mat;
   pfString *labelString;

   scaleMat.makeScale(0.2, 0.2, 0.2);

   sprintf(str, "%.2f", value);
   //fprintf(stderr,"str=[%s]\n", str);

   if (font != NULL)
   {
      labelString = new pfString();
      labelString->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
      labelString->setColor(0.1, 0.1, 0.1, 1.0);
      labelString->setGState(unlightedGeoState);
      labelString->setFont(font);
      mat.postTrans(scaleMat, 0.0, -0.05, 1.32);
      mat.postRot(mat, angle, 0.0, 1.0, 0.0);

      labelString->setMat(mat);
      labelString->setString(str);
      t = new pfText();
      fixNode(t);
      t->addString(labelString);

   }
   else
   {
      fprintf(stderr,"ERROR in StarRegionScalarInteractor::makeLabel: Cannot load font.\n");

   }

   return t;

}
*/

/*
pfText *
StarRegionScalarInteractor::makeTitle(char *name)
{

   //fprintf(stderr,"StarRegionScalarInteractor::makeTitle\n");

   pfText *t = NULL;
   osg::Matrix scaleMat, mat;
   pfString *string;

   scaleMat.makeScale(0.3, 0.3, 0.3);

   if (font != NULL)
   {
      string = new pfString();
      string->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
      string->setColor(1.0, 1.0, 1.0, 1.0);
      string->setGState(unlightedGeoState);
      string->setFont(font);
      mat.postTrans(scaleMat, 0.0, -0.05, 0.6);

      string->setMat(mat);
      string->setString(name);
      t = new pfText();
      fixNode(t);
      t->addString(string);

   }
   else
   {
      fprintf(stderr,"ERROR in StarRegionScalarInteractor::makeLabel: Cannot load font.\n");
   }
   return t;

}
*/

//       20
//      ...
//    .     .
//10 . --o   .
//    .     .
//      ...
//

osg::Geode *
StarRegionScalarInteractor::loadButton()
{
   //fprintf(stderr,"StarRegionScalarInteractor::loadButton\n");
   osg::Vec4 *color;
   osg::Matrix rotMat, transMat, scaleMat;
   osg::ref_ptr<osg::Cylinder> cylinder;
   osg::ref_ptr<osg::ShapeDrawable> geoset;
   osg::ref_ptr<osg::TessellationHints> th = new osg::TessellationHints();
   osg::Geode *geode;

   cylinder = new osg::Cylinder();
   cylinder->setCenter (osg::Vec3 (0.0, 0.0, 0.25));
   cylinder->setRotation (osg::Quat (90.0f, osg::X_AXIS));
   /*   geoset = pfdNewCylinder(60, pfGetSharedArena()); */

   // color red
   color = new osg::Vec4 (1.0f, 0.5f, 0.5f, 1.0f);
   /*   geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, color, NULL);
      pfdGSetColor(geoset, 1.0f, 0.5f, 0.5f, 1.0f);
   */

   th->setCreateFrontFace (true);
   th->setCreateNormals (true);
   th->setCreateBottom (true);
   th->setCreateTop (true);
   th->setCreateBody (true);

   geoset = new osg::ShapeDrawable (cylinder.get(), th.get());
   geoset->setColor (*color);

   // scale it to 1/4 size
   scaleMat.makeScale (1.0, 1.0, 0.25);
   /*   pfdXformGSet(geoset, scaleMat);

      // translate it to origin
      transMat.makeTranslate(0.0, 0.0, 0.25);
      pfdXformGSet(geoset, transMat);

      // rotate it to axis = yaxis
      rotMat.makeRotate(90.0, 1.0f, 0.0f, 0.0f);
      pfdXformGSet(geoset, rotMat);
   */
   geoset->setStateSet (defGeoState.get());

//   sensableGeoset = geoset;

   geode = new osg::Geode();
   /*   fixNode(geode); */
   geode->addDrawable (geoset.get());

   return (geode);

}


osg::Group *
StarRegionScalarInteractor::loadButtonMarker()
{
   //fprintf(stderr,"StarRegionScalarInteractor::loadButtonMarker\n");

   osg::Vec4 *color;
   osg::Matrix rotMat, transMat;
   osg::ref_ptr<osg::ShapeDrawable> geoset;
   osg::ref_ptr<osg::TessellationHints> th = new osg::TessellationHints();

   /*   geoset = pfdNewArrow(60, pfGetSharedArena()); */


   osg::ref_ptr<osg::ShapeDrawable>  coneGeoset, cyl1Geoset, cyl2Geoset;
   osg::ref_ptr<osg::Cylinder> cylinder;
   osg::ref_ptr<osg::Cone> cone;
   osg::ref_ptr<osg::Geode> coneGeode, cyl1, cyl2;
   osg::Group * group;
   osg::ref_ptr<osg::MatrixTransform> coneTransform, cyl1Transform, cyl2Transform;

   // color red
   color = new osg::Vec4 (1.0f, 0.1f, 0.1f, 1.0f);

   cone = new osg::Cone();
   transMat.makeTranslate(0.0f, 0.0f, -1.0f);

//    cone.get()->setCenter (osg::Vec3 (0.0f, 0.0f, -1.0f));
//    cone.get()->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));

   th->setCreateFrontFace (true);
   th->setCreateNormals (true);
   th->setCreateBottom (true);
   th->setCreateTop (true);
   th->setCreateBody (true);

   coneGeoset = new osg::ShapeDrawable (cone.get(), th.get());
   coneGeoset->setColor (*color);
   coneGeoset->setStateSet (defGeoState.get());

   cylinder = new osg::Cylinder();
   transMat.makeTranslate(0.0f, 0.0f, -2.0f);

   cylinder->setCenter (osg::Vec3 (0.0f, 0.0f, -2.0f));
   cylinder->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));

   cyl1Geoset = new osg::ShapeDrawable (cylinder.get(), th.get());
   cyl1Geoset->setColor (*color);
   cyl1Geoset->setStateSet (defGeoState.get());

   cylinder = new osg::Cylinder();
   cylinder->setCenter (osg::Vec3 (0.0f, 0.0f, -3.25f));
   cylinder->setRotation (osg::Quat (-90.0, 1.0, 0.0, 0.0));

   cyl2Geoset = new osg::ShapeDrawable (cylinder.get(), th.get());
   cyl2Geoset->setColor (*color);
   cyl2Geoset->setStateSet (defGeoState.get());

   //sensableGeoset = cyl2Geoset;

   coneGeode = new osg::Geode();
   coneGeode->addDrawable (coneGeoset.get());

   cyl1 = new osg::Geode();
   cyl1->addDrawable (cyl1Geoset.get());

   cyl2 = new osg::Geode();
   cyl2->addDrawable (cyl2Geoset.get());

   group = new osg::Group();
   //FIXME
//    group->addChild (coneGeode);
//    group->addChild (cyl1);
//    group->addChild (cyl2);
   return (group);

}

/*
pfText *
StarRegionScalarInteractor::makeCurrentLabel()
{
   char str[200];
   pfText *t = NULL;
   osg::Matrix scaleMat, mat;
   pfString *currentString;

   scaleMat.makeScale(0.2, 0.2, 0.2);

   sprintf(str, "%.2f", currentValue);

   if (font != NULL)
   {
      currentString = new pfString();
      currentString->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
      currentString->setColor(1.0, 0.0, 0.0, 1.0);
      currentString->setGState(unlightedGeoState);
      currentString->setFont(font);
      mat.postTrans(scaleMat, 0.0, -0.05, 1.32);

      currentString->setMat(mat);
      currentString->setString(str);
      t = new pfText();
      fixNode(t);
      t->addString(currentString);

   }
   else
   {
      fprintf(stderr,"ERROR in StarRegionScalarInteractor::makeLabel: Cannot load font.\n");
   }
   return t;

}
*/

/*
void
StarRegionScalarInteractor::updateLabels()
{
   pfString *curStr, *minStr, *maxStr;
   char str[200];

   sprintf(str, "%.2f", currentValue);

   curStr = currentLabel->getString(0);
   curStr->setString(str);

   sprintf(str, "%.2f", minValue);
   minStr = minLabel->getString(0);
   minStr->setString(str);

   sprintf(str, "%.2f", maxValue);
   maxStr = maxLabel->getString(0);
   maxStr->setString(str);

}
*/

void
StarRegionScalarInteractor::updateScale()
{
   osg::Matrix scaleMat, rotMat, offsetMat;
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());

   osg::Vec3 n;
   n = localMat * interNormal;

   rotMat.makeRotate (defAxis, n);
   scaleMat.makeScale (interScale/cover->getScale(), interScale/cover->getScale(), interScale/cover->getScale());
   offsetMat.mult (scaleMat, rotMat);

   osg::Vec3 p;
   p=interPos;
   float dr=interScale/cover->getScale();
   //p[2]+= -regionRadius - dr -4*dindex*dr;

   osg::Vec3 zaxis (0,0,1);
   zaxis = localMat * zaxis;

   p+=zaxis* (-regionRadius - dr -4*dindex*dr);

   /*   offsetMat.setRow(3, p); */
   for (int i = 0; i < 3; i++)
   {
      offsetMat (3, i) = p[i];
   }

   worldDCS->setMatrix (offsetMat);

}
