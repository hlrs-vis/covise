/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Bore_PLUGIN_H
#define _Bore_PLUGIN_H

#include <util/common.h>
#include <unordered_map>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <config/CoviseConfig.h>
#include <util/coTypes.h>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Array>
#include <osg/ShapeDrawable>
#include <osg/KdTree>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Action.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>

class BoreHolePos
{
public:
	BoreHolePos(std::string info);
	~BoreHolePos();
	std::string ID;
	double x;
	double y;
	double height;
	double depth= -1.0;
	double azimut;
	double angle;
};
class CoreInfo
{
public:
	CoreInfo(std::string info);
	~CoreInfo();
	std::string ID;
	double DepthTop;
	double DepthBase;
	std::string Annotation;
	std::string Stratigraphie;
	std::string color;
	std::string konsistenz;
	std::string kornbildung;
	std::string wassergehalt;
	double kerngewinn;
	std::string length;
	std::string verwitterung;
	std::string kluefte;
	std::string PETRO;
	std::string stratigraphie2;
	std::string Lithologie;
	std::string Vergrusung;
	std::string Verwitterungsgrad;
	std::string grus;
	std::string leicht_vergrust;
	std::string stark_vergrust;
	std::string komplett_vergrust;
	std::string total_vergrust;
	std::string w0;
	std::string w1;
	std::string w2;
	std::string w3;
	std::string w4;
	std::string w5;
	int verw_max;
};

class BoreHole
{
public:
	BoreHole(BoreHolePos *);
    ~BoreHole();
	void init();
	void regenerate(); // regenerate Geometry
	osg::Matrix position;
	std::string ID;
	BoreHolePos *boreHolePos;
	int numSides = 8;
	float radius = 0.4;

    osg::Cylinder *cylinder;
	osg::ref_ptr<osg::MatrixTransform> boreHoleTrans;
	osg::ref_ptr<osg::Geode> geode;
	osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
	std::vector<CoreInfo *> cores;
	osg::ref_ptr<osg::Geode> geodeVergrusung;
	osg::Geode *createGeometry();
};


class BorePlugin : public coVRPlugin, public ui::Owner
{
public:
    BorePlugin();
    ~BorePlugin();
    bool init();
	static BorePlugin *instance() {return plugin; };
	ui::Menu *BoreTab = nullptr;
	ui::Label *infoLabel = nullptr;
	ui::Button *BoreHolesVisible = nullptr;
	ui::Button *Interpolated = nullptr;
	ui::SelectionList *VisualizationType = nullptr;

    static int SloadBore(const char *filename, osg::Group *parent, const char *ck = "");
    int loadBore(std::string fileName, osg::Group *parent);
    static int SunloadBore(const char *filename, const char *ck = "");
    int unloadBore(std::string filename);

    bool update(); // return frue if we need a redraw

    osg::ref_ptr<osg::Group> BoreGroup;
	osg::Group *parent = nullptr;
	osg::Vec3 getProjectOffset();

	std::map<std::string, BoreHole *> Bore_map;
	std::map<std::string, BoreHolePos *> BoreHolePos_map;

	osg::ref_ptr<osg::KdTreeBuilder> d_kdtreeBuilder;

private:
	static BorePlugin *plugin;
};
#endif
