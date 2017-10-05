/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 * File:   Variant.h
 * Author: Gottlieb A.
 *
 * Created on 14. August 2009, 09:45
 */

#ifndef _VARIANT_H
#define _VARIANT_H

#include <osg/Group>
#include <iostream>
#include <cover/coVRPlugin.h>
#include <cover/coVRLabel.h>
#include "VariantUI.h"
#include "coVRBoxOfInterest.h"
#include <QtCore>
#include <qdom.h>

namespace opencover {
namespace ui {
class Menu;
}
}

class Variant : public coVRPlugin, public coTUIListener
{
public:
    static Variant *variantClass;

    Variant(std::string var_Name, osg::Node *node, osg::Node::ParentList parents, ui::Menu *Variant_menu, coTUITab *VariantPluginTab, int numVar, QDomDocument *xmlfile, QDomElement *qDE_V, coVRBoxOfInterest *boi, bool default_state);
    ~Variant();
    //adding the Group-Node to the Scenegraph:
    void AddToScenegraph();
    void removeFromScenegraph(osg::Node *node);
    void attachNode(osg::Node *node);
    void releaseNode(osg::Node *node);
    void attachClippingPlane();
    void releaseClippingPlane();
    osg::Node *getNode();
    void printMatrix(osg::MatrixTransform *mt);

    std::string getVarname()
    {
        return varName;
    };
    int numNodes();
    int numParents();
    osg::Matrix getOriginMatrix();
    void createVRLabel();
    void showVRLabel();
    void hideVRLabel();
    //Events
#ifdef VRUI
    void menuEvent(coMenuItem *item);
#endif
    void tabletEvent(coTUIElement *item);
    void setOriginTransMatrix();
    void setOriginTransMatrix(osg::Vec3d vec);
    osg::Vec3d getTransVec();
    void setQDomElemTRANS(osg::Vec3d vec);
    std::list<osg::Node *> getAttachedNodes();

    VariantUI *ui;

private:
    osg::MatrixTransform *VarNode;
    std::string varName;
    osg::Node::ParentList parents;
    osg::Matrix origin_matrix;
    coVRLabel *VarLabel;

    QDomDocument *xmlfile;
    QDomElement *qDE_Variant;

    std::list<osg::Node *> attachedNodesList;

    coVRBoxOfInterest *myboi;
    osg::ClipNode *cn;
};

#endif /* _Variant_H */
