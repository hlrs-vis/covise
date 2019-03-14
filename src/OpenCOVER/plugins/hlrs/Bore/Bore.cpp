/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Bore.h"
#include <osg/LineWidth>
#include <osg/CullFace>
#include <osg/Version>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <boost/tokenizer.hpp>

BorePlugin *BorePlugin::plugin = NULL;

static const int NUM_HANDLERS = 1;
using namespace boost;

static const FileHandler handlers[] = {
    { NULL,
      BorePlugin::SloadBore,
      BorePlugin::SloadBore,
      BorePlugin::SunloadBore,
      "bcsv" },
};

BoreHolePos::BoreHolePos(std::string info)
{
	escaped_list_separator<char> sep('\\',';','\"');
	tokenizer<escaped_list_separator<char>> tokens(info, sep);
	auto it = tokens.begin();
	ID = *it++;
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		x = std::stod(s);
	}
	catch (...) { x = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		y = std::stod(s);
	}
	catch (...) { y = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		height = std::stod(s);
	}
	catch (...) { height = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		depth = std::stod(s);
	}
	catch (...) { depth = 1.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		azimut = std::stod(s);
	}
	catch (...) { azimut = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		angle = std::stod(s);
	}
	catch (...) { angle = 0.0; }
	if (it == tokens.end())
		return;
}
BoreHolePos::~BoreHolePos()
{

}
CoreInfo::CoreInfo(std::string info)
{
	escaped_list_separator<char> CSVSeparator('\\', ';', '\"');
	char_separator<char> kommaSeparator(".");
	tokenizer<escaped_list_separator<char>> tokens(info, CSVSeparator);
	verw_max = 0.0;
	auto it = tokens.begin();
	ID = *it++;
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		DepthTop = std::stod(s);
	}
	catch (...) { DepthTop = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		DepthBase = std::stod(s);
	}
	catch (...) { DepthBase = 0.0; }
	if (it == tokens.end())
		return;
	Annotation = *it++;
	if (it == tokens.end())
		return;
	Stratigraphie = *it++;
	if (it == tokens.end())
		return;
	color = *it++;
	if (it == tokens.end())
		return;
	std::string kkw = *it++;
	{
		tokenizer<char_separator<char>> tokens(kkw, kommaSeparator);
		auto it = tokens.begin();
		if (it != tokens.end())
		konsistenz = *it++;
		if (it != tokens.end())
		kornbildung = *it++;
		if (it != tokens.end())
		wassergehalt = *it++;
	}
	std::string kk = *it++;
	{
		tokenizer<char_separator<char>> tokens(kk, kommaSeparator);
		auto it = tokens.begin();
		if (it != tokens.end())
		try {
			kerngewinn = std::stoi(*it++)/100.0;
		}
		catch (...) { kerngewinn = 1.0; }
		if (it != tokens.end())
		length = *it++;
	}
	if (it != tokens.end())
	verwitterung = *it++;
	if (it != tokens.end())
	kluefte = *it++;
	if (it != tokens.end())
	PETRO = *it++;
	if (it != tokens.end())
	stratigraphie2 = *it++;
	if (it != tokens.end())
	Lithologie = *it++;
	if (it != tokens.end())
	Vergrusung = *it++;
	if (it != tokens.end())
	Verwitterungsgrad = *it++;
	if (it != tokens.end())
	grus = *it++;
	if (it != tokens.end())
	leicht_vergrust = *it++;
	if (it != tokens.end())
	stark_vergrust = *it++;
	if (it != tokens.end())
	komplett_vergrust = *it++;
	if (it != tokens.end())
	total_vergrust = *it++;
	if (it != tokens.end())
	w0 = *it++;
	if (it != tokens.end())
	w1 = *it++;
	if (it != tokens.end())
	w2 = *it++;
	if (it != tokens.end())
	w3 = *it++;
	if (it != tokens.end())
	w4 = *it++;
	if (it != tokens.end())
	w5 = *it++;
	if (it != tokens.end())
	{
		try {
			verw_max = std::stoi(*it++);
		}
		catch (...) { verw_max = 0.0; }
	}
}
CoreInfo::~CoreInfo()
{

}

BoreHole::BoreHole(BoreHolePos *bp)
{
    fprintf(stderr, "BoreHole::BoreHole\n");
	boreHolePos = bp;

    

}
void BoreHole::init()
{
	cylinder = new osg::Cylinder(osg::Vec3(0, 0, boreHolePos->depth / 2.0), 0.3, boreHolePos->depth);
	cylinderDrawable = new osg::ShapeDrawable(cylinder);
	geode = new osg::Geode();
	geode->setName(boreHolePos->ID);
	geode->addDrawable(cylinderDrawable);
	osg::StateSet *stateset = geode->getOrCreateStateSet();
	osg::Material *material = new osg::Material;
	material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f));
	material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
	material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
	material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
	stateset->setAttributeAndModes(material);
	stateset->setNestRenderBins(false);
	boreHoleTrans = new osg::MatrixTransform();
	geode->setName(boreHolePos->ID+"T");
	regenerate();
	osg::Matrix m;
	osg::Vec3f p(boreHolePos->x, boreHolePos->y, boreHolePos->height);
	m.makeTranslate(p - BorePlugin::instance()->getProjectOffset());
	osg::Matrix rot;
	rot.makeRotate((boreHolePos->angle + 180) / 180.0*M_PI, osg::Vec3(1, 0, 0), boreHolePos->azimut / 180.0*M_PI, osg::Vec3(0, 0, -1), 0.0, osg::Vec3(0, 1, 0));
	boreHoleTrans->setMatrix(rot*m);
	BorePlugin::instance()->BoreGroup->addChild(boreHoleTrans);
}

void BoreHole::regenerate()
{
	if (cores.size() > 0)
	{
		if (geodeVergrusung.get() != nullptr)
		{
			if (geodeVergrusung->getParent(0) != nullptr)
			{
				geodeVergrusung->getParent(0)->removeChild(geodeVergrusung.get());
			}
		}
		geodeVergrusung = createGeometry();
		boreHoleTrans->addChild(geodeVergrusung);
	}
	else
	{
		boreHoleTrans->addChild(geode);
	}
}




// this is called if the plugin is removed at runtime
BoreHole::~BoreHole()
{
    fprintf(stderr, "BorePlugin::~BorePlugin\n");
    if (boreHoleTrans->getNumParents())
		boreHoleTrans->getParent(0)->removeChild(boreHoleTrans);
}

osg::Geode *BoreHole::createGeometry()
{
	osg::Geode *geode = new osg::Geode();
	geode->setName(boreHolePos->ID);
	osg::Geometry *geom = new osg::Geometry();
	cover->setRenderStrategy(geom);
	osg::StateSet *stateset = geode->getOrCreateStateSet();
	osg::Material *material = new osg::Material;
	material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
	material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
	material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
	material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
	material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
	stateset->setAttributeAndModes(material);
	stateset->setNestRenderBins(false);
	bool interpolate = BorePlugin::instance()->Interpolated->state();

	int numSegments = cores.size();

	osg::Vec3Array *vert = new osg::Vec3Array;
	osg::Vec3Array *normal = new osg::Vec3Array;
	osg::Vec4Array *color = new osg::Vec4Array;
	osg::Vec4 colors[6];
	CoreInfo *ci = cores[0];
	colors[0] = osg::Vec4(1, 1, 1, 1);
	colors[1] = osg::Vec4(1, 0.8, 0.8, 1);
	colors[2] = osg::Vec4(1, 0.6, 0.6, 1);
	colors[3] = osg::Vec4(1, 0.4, 0.4, 1);
	colors[4] = osg::Vec4(1, 0.2, 0.2, 1);
	colors[5] = osg::Vec4(1, 0.0, 0.0, 1);
	/*colors[0] = osg::Vec4(1, 1, 1, 1);
	colors[1] = osg::Vec4(1, 0.9, 0.9, 1);
	colors[2] = osg::Vec4(1, 0.8, 0.8, 1);
	colors[3] = osg::Vec4(1, 0.7, 0.7, 1);
	colors[4] = osg::Vec4(1, 0.6, 0.6, 1);
	colors[5] = osg::Vec4(1, 0.5, 0.5, 1);*/
	if(interpolate)
	{
		for (int i = 0; i < numSides; i++)
		{
			float a = i * 2.0 * M_PI / numSides;
			vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, 0.0));
			normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
			color->push_back(colors[ci->verw_max]);
		}
	}
	for (int n = 0; n < numSegments; n++)
	{
		CoreInfo *ci = cores[n];
		for (int i = 0; i < numSides; i++)
		{
			float a = i * 2.0 * M_PI / numSides;
			vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, ci->DepthTop));
			normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
			color->push_back(colors[ci->verw_max]);
		}
		if (!interpolate)
		{
			for (int i = 0; i < numSides; i++)
			{
				float a = i * 2.0 * M_PI / numSides;
				vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, ci->DepthBase));
				normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
				color->push_back(colors[ci->verw_max]);
			}
		}
	}
	osg::DrawElementsUInt *primitives = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES);
	for (int n = 0; n < numSegments; n++)
	{
		int b0 = n * numSides;
		int b1 = (n + 1) * numSides;
		if(!interpolate)
		{
			b0 = (n*2) * numSides;
			b1 = ((n * 2) + 1) * numSides;
		}
		for (int i = 0; i < numSides; i++)
		{
			int ni = i + 1;
			if (i == numSides - 1)
				ni = 0;
			primitives->push_back(b0 + i);
			primitives->push_back(b1 + ni);
			primitives->push_back(b1 + i);
			primitives->push_back(b0 + i);
			primitives->push_back(b0 + ni);
			primitives->push_back(b1 + ni);
		}
	}
	geom->addPrimitiveSet(primitives);
	geom->setVertexArray(vert);;
	geom->setNormalArray(normal);
	geom->setColorArray(color);
	geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);


#if (OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
	BorePlugin::instance()->d_kdtreeBuilder->apply(*geom);
#endif

	geode->addDrawable(geom);
	osg::CullFace *cullFace = new osg::CullFace();
	cullFace->setMode(osg::CullFace::BACK);
	stateset->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
	return geode;
}

BorePlugin::BorePlugin() : ui::Owner("BorePlugin", cover->ui)
{
	plugin = this;
    fprintf(stderr, "BorePlugin::BorePlugin\n");

	d_kdtreeBuilder = new osg::KdTreeBuilder;

    BoreGroup = new osg::Group();
    BoreGroup->setName("Bore_Holes");
	BoreTab = new ui::Menu("BoreHoles", this);
	BoreTab->setText("Bore Holes");
	// infoLabel = new ui::Label("GPS Version 1.0", GPSTab);

	BoreHolesVisible = new ui::Button(BoreTab, "HolesOn");
	BoreHolesVisible->setText("Bore Holes");
	BoreHolesVisible->setCallback([this](bool state)
	{
		if (state)
		{
			if (parent)
			{
				parent->addChild(BoreGroup.get());
			}
		}
		else
		{
			if (parent)
			{
				parent->removeChild(BoreGroup.get());
			}
		}
	});

	Interpolated = new ui::Button(BoreTab, "Interpolated");
	Interpolated->setText("Interpolated");
	Interpolated->setCallback([this](bool state)
	{
		for (auto& b : Bore_map) {
			b.second->regenerate();
		}
	});

	VisualizationType = new ui::SelectionList(BoreTab, "VizType");
	VisualizationType->setText("VizType");
	VisualizationType->setCallback([this](int state)
	{
		for (auto& b : Bore_map) {
			b.second->regenerate();
		}
	});

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);
}

// this is called if the plugin is removed at runtime
BorePlugin::~BorePlugin()
{
    fprintf(stderr, "BorePlugin::~BorePlugin\n");
    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);

    
    Bore_map.clear();
	delete BoreHolesVisible;
    delete BoreTab;
	if(BoreGroup->getParent(0))
	{
		BoreGroup->getParent(0)->removeChild(BoreGroup);
	}
}

osg::Vec3 BorePlugin::getProjectOffset()
{
	return osg::Vec3(3449864.546988, 5392358.883212, 0);
}
bool BorePlugin::init()
{
    return true;
}

bool
BorePlugin::update()
{
	return false;
}

int BorePlugin::SunloadBore(const char *filename, const char *)
{
    return plugin->unloadBore(filename);
}

int BorePlugin::unloadBore(std::string name)
{
    
        for (auto it = Bore_map.begin(); it != Bore_map.end(); it++)
        {
            Bore_map.erase(it);
            break;
        }
    return 0;
}

int BorePlugin::SloadBore(const char *filename, osg::Group *parent, const char *)
{

    if (filename)
    {

        plugin->loadBore(filename, parent);
    }

    return 0;
}

int BorePlugin::loadBore(std::string fileName, osg::Group *p)
{
	parent = p;
	if (parent == nullptr)
		parent = cover->getObjectsRoot();
	parent->addChild(BoreGroup);

	FILE *fp = fopen(fileName.c_str(), "r");
	if (fp != NULL)
	{
		char buf[1000];
		fgets(buf, 1000, fp);
		while (fgets(buf, 1000, fp) != NULL)
		{
			BoreHolePos *bp = new BoreHolePos(buf);
			if(bp->depth > 0)
			{
				BoreHolePos_map[bp->ID] = bp;
				BoreHole *b = new BoreHole(bp);
				Bore_map[bp->ID] = b;
			}
		}

	}
	else
	{
    return 0;
	}
	fclose(fp);
	fp = fopen((fileName.substr(0,fileName.length()-5)+".csv").c_str(), "r");
	if (fp != NULL)
	{
		char buf[1000];
		fgets(buf, 1000, fp);
		while (fgets(buf, 1000, fp) != NULL)
		{
			CoreInfo *c = new CoreInfo(buf);
			auto b = Bore_map.find(c->ID);
			if (b != Bore_map.end())
			{
				b->second->cores.push_back(c);
			}
		}

	}
	else
	{
		return 0;
	}
	for (auto& b : Bore_map) {
		b.second->init();
	}
	return 1;
}


COVERPLUGIN(BorePlugin)
