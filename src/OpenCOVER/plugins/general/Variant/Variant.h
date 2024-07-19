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
#include <set>
#ifdef USE_QT
#include <QtCore>
#include <qdom.h>
#else
class QDomDocument;
class QDomElement;
#endif

namespace opencover {
namespace ui {
class Menu;
}
}
class VariantPlugin;

class Variant: public coTUIListener
{
public:
    static Variant *variantClass;

    Variant(VariantPlugin *plugin, std::string var_Name, osg::Node *node, osg::Node::ParentList parents,
            ui::Menu *Variant_menu, coTUITab *VariantPluginTab, int numVar, QDomDocument *xmlfile, QDomElement *qDE_V,
            coVRBoxOfInterest *boi, bool default_state);
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
    void setParents(osg::Node::ParentList pa);
    osg::Matrix getOriginMatrix();
    void createVRLabel();
    void showVRLabel();
    void hideVRLabel();
    //Events
    void tabletEvent(coTUIElement *item);
    void setOriginTransMatrix();
    void setOriginTransMatrix(osg::Vec3d vec);
    osg::Vec3d getTransVec();
    void setQDomElemTRANS(osg::Vec3d vec);

    VariantUI *ui;
    bool defaultVisibilitySet = false;
    void setVisible(bool state);
    bool isVisible() const;

private:
    osg::ref_ptr<osg::MatrixTransform> VarNode;
    std::string varName;
    osg::Node::ParentList parents;
    osg::Matrix origin_matrix;
    coVRLabel *VarLabel;

#ifdef USE_QT
    QDomDocument *xmlfile;
    QDomElement *qDE_Variant;
#endif

    std::set<osg::Node *> attachedNodesList;

    coVRBoxOfInterest *myboi;
    osg::ref_ptr<osg::ClipNode> cn;
    VariantPlugin *plugin = nullptr;
    bool visible = false;
};

#endif /* _Variant_H */
