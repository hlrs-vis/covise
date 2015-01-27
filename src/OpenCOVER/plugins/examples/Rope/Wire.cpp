/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <device/VRTracker.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/PolygonMode>
#include "RopePlugin.h"
#include "Wire.h"
#include <osg/LineWidth>
#include <stdio.h>
using namespace osg;

//Constructor of Wire
Wire::Wire(rShared *daddy, int id, float posR, float posAngle, float wireR, coTUIFrame *frame, coTUIComboBox *box)
{
    (void)box;

    this->setLenFactor(1.0); // default
    this->daddy = daddy;
    this->initShared(daddy, (char *)"Wire", id, frame, NULL);
    this->setPosRadius(posR);
    this->setPosAngle(posAngle);
    this->setWireRadius(wireR);
    this->coverStuff();
}

void Wire::setWireRadius(float R)
{
    // im Notfall Default Wert setzen
    this->R = ((R > 0.0) ? R : 0.1);
}

void Wire::setColor(Vec4 color)
{
    globalmtl->setDiffuse(Material::FRONT_AND_BACK, color);
}

osg::Vec4 Wire::getColor(void)
{
    return globalmtl->getDiffuse(Material::FRONT_AND_BACK);
}

void Wire::createGeom(void)
{
    int h;
    int nL;
    int s;
    rShared *DA;
    float sHeight;
    int nSegments;

    // Beide Werte holen wir vom Seil-Objekt
    sHeight = this->getSegHeight();
    nSegments = this->getNumSegments();

    primitives->clear();
    indices->clear();
    normals->clear();
    vert->clear();
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    nL = (int)(this->getElemLength() / sHeight);
    for (h = 0; h < (nL + 1); h++)
    {
        for (s = 0; s < nSegments; s++)
        {
            float segAngle = 2 * M_PI / nSegments * s;
            Vec3 v;
            v.set(sin(segAngle) * this->R, cos(segAngle) * this->R, h * sHeight);
            //fprintf(stderr, "h=%d, s=%d, segAngle=%f, v=(%f,%f,%f)\n", h, s, segAngle, v[0], v[1], v[2]);
            vert->push_back(v);
            v[2] = 0;
            v.normalize();
            normals->push_back(v);
            //fprintf(stderr, "h=%d, s=%d, segAngle=%f, v=(%f,%f,%f)\n", h, s, segAngle, v[0], v[1], v[2]);
        }
    }
    for (h = 0; h < nL; h++)
    {
        for (s = 0; s < nSegments; s++)
        {
            if (s < (nSegments - 1))
            {
                indices->push_back((h + 1) * nSegments + s);
                indices->push_back((h + 1) * nSegments + s + 1);
                indices->push_back((h)*nSegments + s + 1);
                indices->push_back((h)*nSegments + s);
            }
            else
            {
                indices->push_back((h + 1) * nSegments + s);
                indices->push_back((h + 1) * nSegments);
                indices->push_back((h)*nSegments);
                indices->push_back((h)*nSegments + s);
            }
            primitives->push_back(4);
        }
    }

    // calcColors();
    geom->dirtyBound();
    geom->dirtyDisplayList();
    // Position und Schlaglaenge des Einzeldrahtes im Litzenverband
    DA = this->daddy;
    this->rotate(this->getPosRadius(), this->getPosAngle(), DA->getLengthOfTwist(), DA->getStateLengthOfTwist(), DA->getOrientation());
    // Position und Schlaglaenge der Litze im Seilverband
    DA = this->daddy->daddy;
    this->rotate(DA->getPosRadius(), DA->getPosAngle(), DA->daddy->getLengthOfTwist(), DA->daddy->getStateLengthOfTwist(), DA->daddy->getOrientation());
    // Position und "Schlaglaenge" des Seiles im Gesamtbild
    // Achtung: entgegen der Situation beim Draht und bei der Litze
    //          kennt das Seil seine Schlaglaenge und NICHT das Objekt drueber ...
    DA = this->daddy->daddy->daddy->daddy;
    this->rotate(DA->getPosRadius(), DA->getPosAngle(), DA->getLengthOfTwist(), DA->getStateLengthOfTwist(), DA->getOrientation());
    cutWire(osg::Vec3(0, 0, 1), 80.0);
}

void Wire::cutWire(osg::Vec3 normal, float dist)
{

    normal.normalize();

    DrawArrayLengths *newPrimitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    UShortArray *newIndices = new UShortArray();

    int startIndex = 0;

    int index = vert->getNumElements();
    for (unsigned int i = 0; i < primitives->getNumPrimitives(); i++)
    {
        float oldd = -1;
        int numInd = 0;
        for (int v = 0; v < (*primitives)[i]; v++)
        {
            osg::Vec3 p = (*vert)[indices->index(startIndex + v)];
            float d = (p - normal * dist) * normal;
            /* Grundidee:
			* wir laufen allen Punkte von jedem Polygon einmal
			* durch; liegt der Punkte im "Verbleib" Bereich, wird
			* wird er einfach kopiert, liegt er ausserhalb, dann
			* wird er weggelassen
			* !! Immer wenn ein Wechsel zwischen "verbleibt" und\
			* und "weglassen" erfolgt, dann muss mindestens ein
			* zusaetzlicher Punkt eingefuegt werden.
			*/
            if (v == 0 && d >= 0)
            {
                newIndices->push_back((*indices)[startIndex + v]);
                numInd++;
            }
            else
            {
                if (d < 0)
                {
                    if (oldd > 0)
                    {
                        osg::Vec3 b = (*vert)[(*indices)[startIndex + v - 1]];
                        osg::Vec3 r = p - b;
                        float t = (dist - normal * p) / (normal * r);
                        osg::Vec3 newP = p + r * t;
                        osg::Vec3 n1 = (*normals)[(*indices)[startIndex + v]];
                        osg::Vec3 n2 = (*normals)[(*indices)[startIndex + v - 1]];
                        osg::Vec3 newN = n1 + (n1 - n2) * t;
                        vert->push_back(newP);
                        normals->push_back(newN);
                        newIndices->push_back(index);
                        index++;
                        numInd++;
                    }
                    else
                    {
                    }
                }
                else
                {
                    if (oldd < 0)
                    {
                        osg::Vec3 b = (*vert)[(*indices)[startIndex + v - 1]];
                        osg::Vec3 r = p - b;
                        float t = (dist - normal * p) / (normal * r);
                        osg::Vec3 newP = p + r * t;
                        osg::Vec3 n1 = (*normals)[(*indices)[startIndex + v]];
                        osg::Vec3 n2 = (*normals)[(*indices)[startIndex + v - 1]];
                        osg::Vec3 newN = n1 + (n1 - n2) * t;
                        vert->push_back(newP);
                        normals->push_back(newN);
                        newIndices->push_back(index);
                        index++;
                        numInd++;
                    }

                    newIndices->push_back((*indices)[startIndex + v]);
                    numInd++;
                }
            }
            oldd = d;
        }

        osg::Vec3 p = (*vert)[indices->index(startIndex)];
        float d = (p - normal * dist) * normal;

        if (d < 0)
        {
            if (oldd > 0)
            {
                osg::Vec3 b = (*vert)[(*indices)[startIndex + (*primitives)[i] - 1]];
                osg::Vec3 r = p - b;
                float t = (dist - normal * p) / (normal * r);
                osg::Vec3 newP = p + r * t;
                osg::Vec3 n1 = (*normals)[(*indices)[startIndex]];
                osg::Vec3 n2 = (*normals)[(*indices)[startIndex + (*primitives)[i] - 1]];
                osg::Vec3 newN = n1 + (n1 - n2) * t;
                vert->push_back(newP);
                normals->push_back(newN);
                newIndices->push_back(index);
                index++;
                numInd++;
            }
        }
        else
        {
            if (oldd < 0)
            {
                osg::Vec3 b = (*vert)[(*indices)[startIndex + (*primitives)[i] - 1]];
                osg::Vec3 r = p - b;
                float t = (dist - normal * p) / (normal * r);
                osg::Vec3 newP = p + r * t;
                osg::Vec3 n1 = (*normals)[(*indices)[startIndex]];
                osg::Vec3 n2 = (*normals)[(*indices)[startIndex + (*primitives)[i] - 1]];
                osg::Vec3 newN = n1 + (n1 - n2) * t;
                vert->push_back(newP);
                normals->push_back(newN);
                newIndices->push_back(index);
                index++;
                numInd++;
            }
        }
        newPrimitives->push_back(numInd);
        startIndex += (*primitives)[i];
    }

    geom->setVertexIndices(newIndices);
    geom->setNormalIndices(newIndices);

    geom->removePrimitiveSet(0, geom->getNumPrimitiveSets());
    geom->addPrimitiveSet(newPrimitives);
    primitives = newPrimitives;
    indices = newIndices;

    // calcColors();
    geom->dirtyBound();
    geom->dirtyDisplayList();
}

void Wire::coverStuff(void)
{
    //start of design functions
    geom = new osg::Geometry();
    geode = new Geode();
    geode->setNodeMask(geode->getNodeMask() & ~(Isect::Walk | Isect::Intersection | Isect::Collision | Isect::Touch | Isect::Pick));
    geom->setUseDisplayList(true);
    geom->setUseVertexBufferObjects(false);
    geode->addDrawable(geom.get());
    vert = new Vec3Array;
    primitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    indices = new UShortArray();
    normals = new Vec3Array;
    //cindices = new UShortArray()   //colors = new Vec4Array();
    StateSet *geoState = geom->getOrCreateStateSet();

    this->createGeom();

    geom->setVertexArray(vert.get());
    geom->setVertexIndices(indices.get());
    geom->setNormalIndices(indices.get());
    geom->setNormalArray(normals.get());
    //geom->setColorIndices(cindices.get());
    //geom->setColorArray(colors.get());
    geom->addPrimitiveSet(primitives.get());

    geoState = geode->getOrCreateStateSet();
    if (globalmtl.get() == NULL)
    {
        globalmtl = new Material;
        globalmtl->ref();
        globalmtl->setColorMode(Material::OFF);
        globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(Material::FRONT_AND_BACK, 10.0f);
    }

    geoState->setRenderingHint(StateSet::OPAQUE_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::OFF);
    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);
    //char name[1000];
    //sprintf(name,"Wire %d", this->getID());
    coVRShader *SolidClipping = coVRShaderList::instance()->get("SolidClipping");
    if (SolidClipping)
        SolidClipping->apply(geode, geom);
    geode->setName(this->getName());
    this->coverGroup->addChild(geode.get());
    //end of design functions
}
//Wire Destructor delets all functions and selections, which have been made
Wire::~Wire()
{
    while (geode->getNumParents())
        geode->getParent(0)->removeChild(geode.get());
}

void Wire::rotate(float posR, float posAngle, float lengthOfTwist, bool stateLOT, osg::Vec3 orientation)
{
    int vNum = 0;
    int nL; // = (int) (numLengthSegments*length*strandlength);
    int h;
    int s;
    int nSegments;
    float sHeight;

    nSegments = this->getNumSegments();
    sHeight = this->getSegHeight();

    nL = (int)(this->getElemLength() / sHeight);
    osg::Matrix m, incRot, globTrans, globRot;
    globTrans.setTrans(posR, 0, 0);
    globRot.setRotate(osg::Quat(posAngle, orientation));
    m = globTrans * globRot;

    osg::Matrix twist;
    twist.makeIdentity();

    if (stateLOT)
    {
        double alpha = (2.0 * M_PI / (this->getRopeLength() / sHeight)) * (this->getRopeLength() / lengthOfTwist);
        incRot.setRotate(osg::Quat(alpha, orientation));
        float twistangle = -atan((2 * posR * sin(alpha / 2.0)) / sHeight);
        twist.makeRotate(twistangle, osg::Vec3(1, 0, 0));
    }
    else
    {
        incRot.makeIdentity();
    }
    osg::Vec3Array *v = vert.get();
    osg::Vec3Array *n = normals.get();
    for (h = 0; h < (nL + 1); h++)
    {
        for (s = 0; s < nSegments; s++)
        {
            osg::Matrix trans, itrans, summ;
            trans.makeTranslate(0, 0, h * sHeight);
            itrans.makeTranslate(0, 0, -h * sHeight);
            summ = itrans * twist * trans * m;
            (*v)[vNum] = (*v)[vNum] * summ;
            (*n)[vNum] = osg::Matrix::transform3x3((*n)[vNum], summ);
            vNum++;
        }
        if (stateLOT)
            m = m * incRot;
    }
}

xercesc::DOMElement *Wire::Save(xercesc::DOMDocument &document)
{
    char tmp[256];
    osg::Vec4 color;

    fprintf(stderr, "Wire->Save() ... %s\n", this->getName());

    xercesc::DOMElement *element = document.createElement(xercesc::XMLString::transcode(this->getName()));

    element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    element->setAttribute(xercesc::XMLString::transcode("depth"), xercesc::XMLString::transcode(this->getName()));
    sprintf(tmp, "%f", this->getPosRadius());
    element->setAttribute(xercesc::XMLString::transcode("posRadius"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->rad2grad(this->getPosAngle()));
    element->setAttribute(xercesc::XMLString::transcode("posAngle"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->R);
    element->setAttribute(xercesc::XMLString::transcode("Radius"), xercesc::XMLString::transcode(tmp));
    color = this->getColor();
    sprintf(tmp, "%f", color[0]);
    element->setAttribute(xercesc::XMLString::transcode("Color0"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", color[1]);
    element->setAttribute(xercesc::XMLString::transcode("Color1"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", color[2]);
    element->setAttribute(xercesc::XMLString::transcode("Color2"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", color[3]);
    element->setAttribute(xercesc::XMLString::transcode("Color3"), xercesc::XMLString::transcode(tmp));

    return element;
}

void Wire::Load(xercesc::DOMElement *node)
{
    // alle was der Konstruktor benoetogt, wird bereits in wiregroup geladen
    // wir machen hier nur den rest ...
    char *p;
    osg::Vec4 color = Vec4(0.9f, 0.9f, 0.9f, 1.0); // default

    if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("Color0")))) != NULL)
        color[0] = atof(p);
    if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("Color0")))) != NULL)
        color[1] = atof(p);
    if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("Color0")))) != NULL)
        color[2] = atof(p);
    if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("Color0")))) != NULL)
        color[3] = atof(p);
    this->setColor(color);
}
