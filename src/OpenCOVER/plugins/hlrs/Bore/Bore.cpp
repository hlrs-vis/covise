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
#include <osg/TexEnv>
#include <osg/TexGen>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <boost/tokenizer.hpp>
#include <osgDB/ReadFile>
#include <boost/filesystem.hpp>

BorePlugin *BorePlugin::plugin = NULL;


HoleSensor::HoleSensor(BoreHole *h, osg::Node *n): coPickSensor(n)
{
	hole = h;
}

HoleSensor::~HoleSensor()
{
	if (active)
		disactivate();
}

void HoleSensor::activate()
{
	hole->activate();
}

void HoleSensor::disactivate()
{
	hole->disactivate();
}

static const int NUM_HANDLERS = 1;
using namespace boost;

static const FileHandler handlers[] = {
    { NULL,
      BorePlugin::SloadBore,
      BorePlugin::SunloadBore,
      "bcsv" },
};

BoreHolePos::BoreHolePos(const std::string &info, BoreHole::DatabaseVersion dbv)
{
	escaped_list_separator<char> sep('\\',';','\"');
	tokenizer<escaped_list_separator<char>> tokens(info, sep);
	height = 0.0;
	depth = 40.0;
	type = dbv;
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
		if(type == BoreHole::enbw)
		    height = std::stod(s);
		else
			depth = std::stod(s);
	}
	catch (...) { height = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		if (type == BoreHole::enbw)
			depth = std::stod(s);
		else
			height = std::stod(s);
	}
	catch (...) {  }
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
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		angle2 = std::stod(s);
	}
	catch (...) { angle2 = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string ds = *it++;
		std::replace(ds.begin(), ds.end(), ',', '.');
		buildingDist = std::stod(ds);
	}
	catch (...) { buildingDist = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string ds = *it++;
		std::replace(ds.begin(), ds.end(), ',', '.');
		Bauwerksbereich = std::stod(ds);
	}
	catch (...) { Bauwerksbereich = 0.0; }
	if (it == tokens.end())
		return;
	try {
		description = *it++;
	}
	catch (...) {  }
	while (it != tokens.end())
	{
		textureInfo ti;
		ti.fileName = *it++;
		if (it == tokens.end())
			return;
		std::string s = *it++;
		if (it == tokens.end())
			return;
		std::replace(s.begin(), s.end(), ',', '.');
		ti.startDepth = std::stod(s);
		s = *it++;
		if (it == tokens.end())
			return;
		std::replace(s.begin(), s.end(), ',', '.');
		ti.endDepth = std::stod(s);
		textures.push_back(ti);
		fprintf(stderr, "BT: %s,%f,%f\n", ti.fileName.c_str(), ti.startDepth, ti.endDepth);
	}
}
BoreHolePos::~BoreHolePos()
{

}
CoreInfo::CoreInfo(const std::string &info)
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

BoreHole::BoreHole(BoreHolePos *bp, const std::string &p,DatabaseVersion dbv)
{
    fprintf(stderr, "BoreHole::BoreHole\n");
	type = dbv;
	boreHolePos = bp;
	if (p.length() == 0)
	{
		path = ".";
	}
	else
	{
		path = p;
	}
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
	holeSensor = new HoleSensor(this, boreHoleTrans);
	boreHoleTrans->setName(boreHolePos->ID+"_t");
	BorePlugin::instance()->BoreGroup->addChild(boreHoleTrans);
}
void BoreHole::activate()
{
	currentTex++;
	if (currentTex >= boreHolePos->textures.size())
	{
		currentTex = -1;
	}
	regenerate();
}
void BoreHole::disactivate()
{
	//regenerate();
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
		if (cleftGeode.get() != nullptr)
		{
			if (cleftGeode->getParent(0) != nullptr)
			{
				cleftGeode->getParent(0)->removeChild(cleftGeode.get());
			}
		}
		geodeVergrusung = createGeometry();
		boreHoleTrans->addChild(geodeVergrusung);
		cleftGeode = createCleftGeometry(1.5);
		boreHoleTrans->addChild(cleftGeode);
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
	delete holeSensor;
    if (boreHoleTrans->getNumParents())
		boreHoleTrans->getParent(0)->removeChild(boreHoleTrans);
}

void BoreHole::update()
{
	holeSensor->update();
}

osg::Geode *BoreHole::createGeometry()
{
	osg::Geode *geode = new osg::Geode();
	geode->setName(boreHolePos->ID);
	osg::Geometry *geom = new osg::Geometry();
	cover->setRenderStrategy(geom);
	bool interpolate = BorePlugin::instance()->Interpolated->state();

	int numSegments = cores.size();

	osg::Vec3Array *vert = new osg::Vec3Array;
	osg::Vec3Array *normal = new osg::Vec3Array;
	osg::Vec4Array *color = new osg::Vec4Array;
	osg::Vec2Array *texCoord = new osg::Vec2Array;
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
	float minT, maxT;
	bool texture = false;;
	osg::Image * image = NULL;
	if (currentTex >= 0 && currentTex < boreHolePos->textures.size())
	{
		minT = boreHolePos->textures[currentTex].startDepth;
		maxT = boreHolePos->textures[currentTex].endDepth;
		image = osgDB::readImageFile(path+"/"+boreHolePos->textures[currentTex].fileName);
		if (image == NULL)
		{

			osg::notify(osg::ALWAYS) << "Can't open image file" << path + "/" + boreHolePos->textures[currentTex].fileName << std::endl;
		}
		else
		{
			texture = true;
		}
	}

	osg::StateSet *stateset = geom->getOrCreateStateSet();
	osg::Material *material = new osg::Material;
	if (texture)
	{
		osg::Texture2D *texture = new osg::Texture2D(image);
		texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR_MIPMAP_LINEAR);
		texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
		texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
		texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_BORDER);
		texture->setBorderColor(osg::Vec4(1,1,1,1));
		texture->setResizeNonPowerOfTwoHint(false);
		stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
		material->setColorMode(osg::Material::OFF);
		osg::TexEnv *tEnv = new osg::TexEnv(osg::TexEnv::MODULATE);
		stateset->setTextureAttributeAndModes(0, tEnv, osg::StateAttribute::ON);
		osg::ref_ptr<osg::TexGen> texGen = new osg::TexGen();
		stateset->setTextureAttributeAndModes(0, texGen.get(), osg::StateAttribute::OFF);
	}
	else
	{
		material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
	}
	material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
	material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
	material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
	material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
	stateset->setAttributeAndModes(material);
	stateset->setNestRenderBins(false);
	
	if(interpolate)
	{
		for (int i = 0; i < numSides+1; i++)
		{
			float a = i * 2.0 * M_PI / numSides;
			vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, 0.0));
			if(texture)
			    texCoord->push_back(osg::Vec2(i/(float)numSides, 0.0));
			else
				color->push_back(colors[ci->verw_max]);
			normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
		}
	}
	for (int n = 0; n < numSegments; n++)
	{
		CoreInfo *ci = cores[n];
		for (int i = 0; i < numSides+1; i++)
		{
			float a = i * 2.0 * M_PI / numSides;
			vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, ci->DepthTop));
			if (texture)
				texCoord->push_back(osg::Vec2(i / (float)numSides, (ci->DepthTop - minT) / (maxT - minT)));
			else
				color->push_back(colors[ci->verw_max]);
			normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
		}
		if (!interpolate)
		{
			for (int i = 0; i < numSides+1; i++)
			{
				float a = i * 2.0 * M_PI / numSides;
				vert->push_back(osg::Vec3(cos(a)*radius, sin(a)*radius, ci->DepthBase));
				if (texture)
					texCoord->push_back(osg::Vec2(i / (float)numSides, (ci->DepthBase - minT) / (maxT - minT)));
				else
					color->push_back(colors[ci->verw_max]);
				normal->push_back(osg::Vec3(cos(a), sin(a), 0.0));
			}
		}
	}
	osg::DrawElementsUInt *primitives = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES);
	for (int n = 0; n < numSegments; n++)
	{
		int b0 = n * (numSides+1);
		int b1 = (n + 1) * (numSides+1);
		if(!interpolate)
		{
			b0 = (n*2) * (numSides+1);
			b1 = ((n * 2) + 1) * (numSides+1);
		}
		for (int i = 0; i < numSides; i++)
		{
			int ni = i + 1;
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
	if (texture)
	{
		geom->setTexCoordArray(0, texCoord);
	}
	else
	{
		geom->setColorArray(color);
		geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	}
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

osg::Geode *BoreHole::createCleftGeometry(float cleftRadius)
{
	osg::Geode *geode = new osg::Geode();
	geode->setName(boreHolePos->ID+"Cleft");
	osg::Geometry *geom = new osg::Geometry();
	cover->setRenderStrategy(geom);


	osg::Vec3Array *vert = new osg::Vec3Array;
	osg::Vec3Array *normal = new osg::Vec3Array;
	osg::Vec4Array *color = new osg::Vec4Array;
	osg::Vec4 colors[6];
	colors[0] = osg::Vec4(1, 1, 1, 1);
	colors[1] = osg::Vec4(1, 0, 0.1, 1);
	colors[2] = osg::Vec4(0.1, 1, 0.0, 1);
	colors[3] = osg::Vec4(1, 0.4, 0.4, 1);
	colors[4] = osg::Vec4(1, 0.2, 0.2, 1);
	colors[5] = osg::Vec4(1, 0.0, 0.0, 1);

	osg::StateSet *stateset = geom->getOrCreateStateSet();
	osg::Material *material = new osg::Material;
	material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
	material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
	material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
	material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
	material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
	stateset->setAttributeAndModes(material);
	stateset->setNestRenderBins(false);

	for (auto ci : clefts)
	{

		osg::Matrix m;
		osg::Vec3f p(0,0, ci->depth);
		m.makeTranslate(p);
		osg::Matrix rot;
		rot.makeRotate(ci->angle1 / 180.0*M_PI, osg::Vec3(0, 0, 1), (ci->angle2) / 180.0*M_PI, osg::Vec3(1, 0, 0), 0.0, osg::Vec3(0, 1, 0));
		m = rot*m;

		osg::Vec3 n(0, 0, 1);
		n = osg::Matrix::transform3x3(m, n);
		osg::Vec3 v = osg::Vec3(0, 0, 0);
		v = m.preMult(v);
		vert->push_back(v);
		normal->push_back(n);

		for (int i = 0; i < numSides; i++)
		{
			float a = i * 2.0 * M_PI / numSides;
			osg::Vec3 v = osg::Vec3(cos(a)*cleftRadius, sin(a)*cleftRadius, 0);
			v = m.preMult(v);
			vert->push_back(v);
			normal->push_back(n);

			color->push_back(colors[ci->type]);
		}
	}
	osg::DrawElementsUInt *primitives = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES);
	for (int n = 0; n < clefts.size(); n++)
	{
		int b0 = n * (numSides + 1);
		for (int i = 0; i < numSides; i++)
		{
			primitives->push_back(b0 );
			primitives->push_back(b0 + i+1);
			if (i == numSides - 1)
				primitives->push_back(b0 + 1);
			else
				primitives->push_back(b0 + i + 2);
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
	stateset->setAttributeAndModes(cullFace, osg::StateAttribute::OFF);
	return geode;
}

BorePlugin::BorePlugin() 
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("BorePlugin", cover->ui)
{
	plugin = this;
    fprintf(stderr, "BorePlugin::BorePlugin\n");

	d_kdtreeBuilder = new osg::KdTreeBuilder;

    BoreGroup = new osg::MatrixTransform();
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
	while(BoreGroup.get() && BoreGroup->getNumParents())
	{
		BoreGroup->getParent(0)->removeChild(BoreGroup.get());
	}
}

osg::Vec3 BorePlugin::getProjectOffset()
{
	return projectOffset;
}
bool BorePlugin::init()
{
    return true;
}

bool
BorePlugin::update()
{
	for (auto& b : Bore_map) {
		b.second->update();
	}
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

	boost::filesystem::path fpath(fileName);

	BoreHole::DatabaseVersion type = BoreHole::enbw;
	FILE *fp = fopen(fileName.c_str(), "r");
	if (fp != NULL)
	{
		char buf[1000];
		fgets(buf, 1000, fp);

		if (strncmp(buf, "Aufschluss", 10) == 0)
		{
			type = BoreHole::suedlink;
			projectOffset = osg::Vec3( 3524820, 5968280, 0);
			projectOrientation = 108.903055555;
			BoreGroup->setMatrix(osg::Matrix::rotate(projectOrientation*M_PI/180.0,osg::Vec3(0,0,1))*osg::Matrix::translate(64.7,-10.26,-1.2));
		}
		else
		{
			projectOffset = osg::Vec3(3449864.546988, 5392358.883212, 0);
			projectOrientation = 0.0;
		}
		while (fgets(buf, 1000, fp) != NULL)
		{
			BoreHolePos *bp = new BoreHolePos(buf, type);
			if(bp->depth > 0)
			{
				BoreHolePos_map[bp->ID] = bp;
				std::string path = fpath.parent_path().string();
				BoreHole *b = new BoreHole(bp,path,type);
				Bore_map[bp->ID] = b;
			}
		}

	}
	else
	{
    return 0;
	}
	fclose(fp);
	if (type == BoreHole::enbw)
	{
		std::string basename = fileName.substr(0, fileName.length() - 5);
        auto csvFile = coVRFileManager::instance()->findOrGetFile(basename + ".csv");
		fp = fopen(csvFile.c_str(), "r");
		if (fp)
		{
			char buf[1000];
			fgets(buf, 1000, fp);
			while (fgets(buf, 1000, fp) != NULL)
			{
				CoreInfo* c = new CoreInfo(buf);
				auto b = Bore_map.find(c->ID);
				if (b != Bore_map.end())
				{
					b->second->cores.push_back(c);
				}
			}

		    fclose(fp);
		}
        auto KluefteFile = coVRFileManager::instance()->findOrGetFile(basename + "Kluefte.csv");
		fp = fopen(KluefteFile.c_str(), "r");
		if (fp)
		{
			char buf[1000];
			fgets(buf, 1000, fp);
			while (fgets(buf, 1000, fp) != NULL)
			{
				Cleft* c = new Cleft(buf);
				auto b = Bore_map.find(c->ID);
				if (b != Bore_map.end())
				{
					b->second->clefts.push_back(c);
				}
			}
            fclose(fp);
		}
	}
	for (auto& b : Bore_map) {
		b.second->init();
	}
	return 1;
}


COVERPLUGIN(BorePlugin)

textureInfo::textureInfo(const textureInfo &t)
{
	fileName = t.fileName;
	startDepth = t.startDepth;
	endDepth = t.endDepth;
}

Cleft::Cleft(const std::string & info)
{
	escaped_list_separator<char> sep('\\', ';', '\"');
	tokenizer<escaped_list_separator<char>> tokens(info, sep);
	auto it = tokens.begin();
	ID = *it++;
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		number = std::stoi(s);
	}
	catch (...) { number = 0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		depth = std::stod(s);
	}
	catch (...) { depth = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		angle1 = std::stod(s);
	}
	catch (...) { angle1 = 0.0; }
	if (it == tokens.end())
		return;
	try {
		std::string s = *it++;
		std::replace(s.begin(), s.end(), ',', '.');
		angle2 = std::stod(s);
	}
	catch (...) { angle2 = 0.0; }
	if (it == tokens.end())
		return;
	std::string stype = *it++;
	if (stype == "Kluft")
		type = 0;
	else
		type = 1;
}

Cleft::~Cleft()
{
}
