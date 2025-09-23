#include "osg_util.h"
#include "gl_state.h"
#ifdef WIN32
#include <windows.h>
#endif
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <set>
#include <algorithm>
#include <limits>
#include <cover/coVRConfig.h>


void osg_util::dumpAllStateAttributes(const osg::StateSet* ss)
{
	if (!ss) {
		std::cout << "StateSet is NULL"; return;
	}
	std::cout << "=== StateAttributes ===\n";
	for (const auto& p : osg_util::ATTRIBUTE_TYPE_NAMES)
	{
		int code = p.first;
		const char* name = p.second;
		const auto* pair = ss->getAttributePair(static_cast<osg::StateAttribute::Type>(code), 0);
		std::cout << name << " (" << code << "): ";
		if (pair)
		{
			auto ov = pair->second;
			int ovInt = static_cast<int>(ov);
			const char* ovStr;
			switch (ovInt) {
			case osg_util::Values::OFF:       ovStr = "OFF"; break;
			case osg_util::Values::ON:        ovStr = "ON"; break;
			case osg_util::Values::OVERRIDE:  ovStr = "OVERRIDE"; break;
			case osg_util::Values::PROTECTED: ovStr = "PROTECTED"; break;
			case osg_util::Values::INHERIT:   ovStr = "INHERIT"; break;
			default:                          ovStr = "UNKNOWN"; break;
			}
			std::cout << "SET (Override=" << ovStr << ")\n";
		}
		else
		{
			std::cout << "NOT SET\n";
		}
	}
	std::cout << "=== End StateAttributes ===\n\n";
}


void osg_util::dumpAllModes(const osg::StateSet* ss)
{
    if (!ss) { std::cout << "StateSet: NULL\n"; return; }
    std::cout << "=== Dump StateSet @" << ss << " ===\n";

    // 1) Die Modi, die explizit im StateSet gesetzt sind
    std::cout << "-- explizit im StateSet gesetzte Modes ("
        << ss->getModeList().size() << ") --\n";
    for (auto& kv : ss->getModeList())
    {
        GLenum mode = kv.first;
        auto value = kv.second;
        std::cout << "  0x" << std::hex << mode << std::dec
            << " = "
            << (value == osg::StateAttribute::ON ? "ON"
                : value == osg::StateAttribute::OFF ? "OFF"
                : "INHERIT")
            << "  (im StateSet)\n";
    }

    // 2) Alle Modi, tatsächlich aktiviert in der aktuellen GL‑Context
    std::cout << "-- tatsächlich enabled im OpenGL-Context --\n";
    for (auto mode : osg_util::ALL_GL_MODES)
    {
        GLboolean isOn = glIsEnabled(mode);
        std::cout << "  Mode 0x" << std::hex << mode << std::dec
            << " = " << (isOn == GL_TRUE ? "ENABLED" : "DISABLED")
            << "\n";
    }
}


static const std::vector<std::pair<GLenum, std::string>> KNOWN_GL_MODES = {
    { GL_DEPTH_TEST,        "GL_DEPTH_TEST" },
    { GL_BLEND,             "GL_BLEND" },
    { GL_ALPHA_TEST,        "GL_ALPHA_TEST" },
    { GL_CULL_FACE,         "GL_CULL_FACE" },
    { GL_STENCIL_TEST,      "GL_STENCIL_TEST" },
    { GL_SCISSOR_TEST,      "GL_SCISSOR_TEST" },
    { GL_POLYGON_OFFSET_FILL, "GL_POLYGON_OFFSET_FILL" },
    { GL_RESCALE_NORMAL,    "GL_RESCALE_NORMAL" },
    { GL_NORMALIZE,         "GL_NORMALIZE" },
    { GL_COLOR_MATERIAL,    "GL_COLOR_MATERIAL" },
    { GL_LINE_SMOOTH,       "GL_LINE_SMOOTH" },
    { GL_POLYGON_SMOOTH,    "GL_POLYGON_SMOOTH" },
    { GL_TEXTURE_1D,        "GL_TEXTURE_1D" },
    { GL_TEXTURE_2D,        "GL_TEXTURE_2D" },
    { GL_POINT_SMOOTH,      "GL_POINT_SMOOTH" },
    { GL_MAP1_VERTEX_3,     "GL_MAP1_VERTEX_3" },
    { GL_MAP1_VERTEX_4,     "GL_MAP1_VERTEX_4" }
    // … bei Bedarf erweitern …
};


std::string osg_util::vec3ToString(const osg::Vec3& v) {
	std::ostringstream oss;
	oss << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
	return oss.str();
}


std::string osg_util::matrixdToString(const osg::Matrixd& m) {
	std::ostringstream oss;
	oss << "[";
	for (int i = 0; i < 4; ++i) {
		oss << "(";
		for (int j = 0; j < 4; ++j) {
			oss << m(i, j);
			if (j < 3) oss << ", ";
		}
		oss << ")";
		if (i < 3) oss << ", ";
	}
	oss << "]";
	return oss.str();
}


void osg_util::dumpStateSet(const osg::StateSet* ss)
{
    if (!ss)
    {
        std::cout << "StateSet: NULL\n";
        return;
    }

    std::cout << "=== Dump StateSet @" << ss << " ===\n";
    std::cout << "-- GL Modes (erbt oder global) --\n";
    for (auto& entry : KNOWN_GL_MODES)
    {
        GLenum mode = entry.first;
        const auto name = entry.second;
        auto val = ss->getMode(mode);
        std::cout << "  " << name
            << " = " << osg_util::modeValueToString(val)
            << "\n";
    }

    // --- Texture Modes per Name pro Unit ---
    std::cout << "-- Texture Modes (erbt oder global) --\n";
    unsigned int numUnits = ss->getNumTextureModeLists();
    for (unsigned int unit = 0; unit < numUnits; ++unit)
    {
        std::cout << " [Unit " << unit << "]\n";
        for (auto& entry : KNOWN_GL_MODES)
        {
            GLenum mode = entry.first;
            const auto name = entry.second;
            auto val = ss->getTextureMode(unit, mode);
            if (val != osg::StateAttribute::INHERIT)
            {
                // Nur die, die nicht vererbt werden (d.h. explizit ON/OFF)
                std::cout << "    " << name
                    << " = " << osg_util::modeValueToString(val)
                    << "\n";
            }
        }
    }

    const auto& attrList = ss->getAttributeList();
    std::cout << "-- Attributes (" << attrList.size() << ") --\n";
    for (auto& kv : attrList)
    {
        auto typeMember = kv.first;
        auto pair = kv.second;
        auto attrib = pair.first.get();
        auto ov = pair.second;
        std::cout
            << "  Type=" << static_cast<int>(typeMember.first)
            << ", Member=" << typeMember.second
            << ", Override=" << osg_util::overrideValueToString(ov)
            << ", Ptr=" << attrib
            << "\n";
    }

    std::cout << "=== End Dump ===\n";
}


void osg_util::dumpInheritedAttributes(const osg::StateSet* ss, int depth, std::set<const osg::StateSet*>& visited)
{
    if (!ss) return;

    if (visited.count(ss) > 0)
    {
        std::cout << std::string(depth * 2, ' ')
            << "[StateSet @" << ss << "] (bereits besucht, Breche Rekursion)\n";
        return;
    }
    // Merke es als besucht
    visited.insert(ss);

    std::string indent(depth * 2, ' ');

    // 1) Attribute in genau diesem StateSet
    const auto& attrList = ss->getAttributeList();
    if (!attrList.empty())
    {
        std::cout << indent << "[StateSet @" << ss << "] enthält "
            << attrList.size() << " Attribute:\n";
        for (const auto& kv : attrList)
        {
            auto typeMember = kv.first;
            auto pair = kv.second;
            auto attrib = pair.first.get();
            auto ov = pair.second;

            std::cout << indent << "  - Type=" << static_cast<int>(typeMember.first)
                << ", Member=" << typeMember.second
                << ", Override=" << osg_util::overrideValueToString(ov)
                << ", Ptr=" << attrib
                << "\n";
        }
    }
    else
    {
        std::cout << indent << "[StateSet @" << ss << "] hat keine expliziten Attribute\n";
    }

    // 2) Eltern abklappern (Nodes und Drawables)
    for (auto parentObj : ss->getParents())
    {
        // Node?
        if (auto parentNode = dynamic_cast<osg::Node*>(parentObj))
        {
            const osg::StateSet* parentSS = parentNode->getStateSet();
            std::cout << indent << "-> geerbt von Node @"
                << parentNode << " (StateSet @" << parentSS << "):\n";
            dumpInheritedAttributes(parentSS, depth + 1, visited);
        }
        // Drawable?
        else if (auto parentDraw = dynamic_cast<osg::Drawable*>(parentObj))
        {
            const osg::StateSet* parentSS = parentDraw->getStateSet();
            std::cout << indent << "-> geerbt von Drawable @"
                << parentDraw << " (StateSet @" << parentSS << "):\n";
            dumpInheritedAttributes(parentSS, depth + 1, visited);
        }
        else
        {
            std::cout << indent << "-> geerbt von unbekanntem Parent @"
                << parentObj << "\n";
        }
    }
}


void osg_util::dumpStateSetWithInheritance(const osg::StateSet* ss)
{
    std::cout << "=== Dump StateSet mit Vererbung ===\n";
    std::set<const osg::StateSet*> visited;
    dumpInheritedAttributes(ss, 0, visited);
    std::cout << "=== Ende Dump ===\n";
}


void osg_util::dumpStateSetOld(const osg::StateSet* ss)
{
    if (!ss) { std::cout << "StateSet: NULL\n"; return; }
    std::cout << "=== Dump StateSet @" << ss << " ===\n";
    // 1) GL‑Modes
    const auto& modeList = ss->getModeList();
    std::cout << "-- GL Modes (" << modeList.size() << ") --\n";
    for (const auto& kv : modeList)
    {
        GLenum mode = kv.first;
        auto  value = kv.second;
        std::cout
            << "  Mode 0x" << std::hex << mode << std::dec
            << " = "
            << (value == osg::StateAttribute::ON ? "ON"
                : value == osg::StateAttribute::OFF ? "OFF"
                : "INHERIT")
            << "\n";
    }
    // 2) Texture‑Modes
    const auto& texModeList = ss->getTextureModeList();
    std::cout << "-- Texture Modes ("
        << texModeList.size() << " Units) --\n";
    for (size_t unit = 0; unit < texModeList.size(); ++unit)
    {
        const auto& unitModes = texModeList[unit];
        if (unitModes.empty()) continue;

        std::cout << "  [Unit " << unit << "]\n";
        for (const auto& kv : unitModes)
        {
            GLenum mode = kv.first;
            auto  value = kv.second;
            std::cout
                << "    Mode 0x" << std::hex << mode << std::dec
                << " = "
                << (value == osg::StateAttribute::ON ? "ON"
                    : value == osg::StateAttribute::OFF ? "OFF"
                    : "INHERIT")
                << "\n";
        }
    }
    // 3) StateAttributes
    // --- StateAttributes ---
    const auto& attrList = ss->getAttributeList();
    std::cout << "-- Attributes (" << attrList.size() << ") --\n";
    for (auto& kv : attrList)
    {
        auto typeMember = kv.first;
        auto pair = kv.second;
        auto attrib = pair.first.get();
        auto ov = pair.second;
        std::cout
            << "  Type=" << static_cast<int>(typeMember.first)
            << ", Member=" << typeMember.second
            << ", Override=" << osg_util::overrideValueToString(ov)
            << ", Ptr=" << attrib
            << "\n";
    }
    // 4) Texture-Attributes
    std::cout << "-- Texture Attributes --\n";
    unsigned int numTexAttr = ss->getNumTextureAttributeLists();
    for (unsigned int unit = 0; unit < numTexAttr; ++unit)
    {
        const auto& tal = ss->getTextureAttributeList()[unit];
        if (tal.empty()) continue;
        std::cout << "  Texture Unit " << unit << ":\n";
        for (auto& kv : tal)
        {
            auto typeMember = kv.first;
            auto pair = kv.second;
            auto attribute = pair.first.get();
            auto ov = pair.second;
            std::cout
                << "    Type=" << static_cast<int>(typeMember.first)
                << ", Member=" << typeMember.second
                << ", Override=" << osg_util::overrideValueToString(ov)
                << ", Ptr=" << attribute << "\n";
        }
    }
    // 5) Uniforms
    const auto& uniList = ss->getUniformList();
    std::cout << "-- Uniforms (" << uniList.size() << ") --\n";
    for (const auto& kv : uniList)
    {
        const std::string& name = kv.first;
        const auto& pair = kv.second;
        auto ub = pair.first.get();
        auto ov = pair.second;

        std::cout << "  " << name
            << ", Override=" << osg_util::overrideValueToString(ov);

        // Falls es wirklich eine osg::Uniform ist, können wir deren getType() nutzen:
        if (auto u = dynamic_cast<osg::Uniform*>(ub))
        {
            std::cout << ", Type=" << u->getType()
                << ", Elements=" << u->getNumElements();
        }
        else
        {
            // Fallback: Klassenname via RTTI
            std::cout << ", Class=" << typeid(*ub).name();
        }

        std::cout << "\n";
    }
    // 6) Defines
    const auto& defList = ss->getDefineList();
    std::cout << "-- Defines --\n";
    for (auto& kv : defList)
    {
        const std::string& name = kv.first;
        auto val = kv.second;
        std::cout
            << "  " << name
            << " = " << val.first
            << ", Override=" << std::string(osg_util::overrideValueToString(val.second))
            << "\n";
    }
    // 7) RenderingHint und RenderBin
    std::cout
        << "-- RenderingHint: " << ss->getRenderingHint() << "\n"
        << "-- RenderBinMode: " << ss->getRenderBinMode()
        << ", BinNumber: " << ss->getBinNumber()
        << ", BinName: " << ss->getBinName() << "\n"
        << "-- NestRenderBins: " << (ss->getNestRenderBins() ? "true" : "false") << "\n";

    // 8) Eltern-Knoten (ParentList)
    std::cout << "-- Parents (" << ss->getNumParents() << ") --\n";
    for (unsigned int i = 0; i < ss->getNumParents(); ++i)
    {
        std::cout << "  [" << i << "] " << ss->getParent(i) << "\n";
    }
}


void osg_util::printGraphicsContextAttributes(const osg::GraphicsContext* gc)
{
	std::cerr << "---------------------" << std::endl;
	GLState::printCurrentContext();
	if (!gc)
	{
		std::cout << "GraphicsContext is null." << std::endl;
		return;
	}
	std::cout << "GraphicsContext Pointer: " << gc << std::endl;
	std::cout << "Default FBO ID: " << gc->getDefaultFboId() << std::endl;

	const osg::GraphicsContext::Traits* traits = gc->getTraits();
	if (!traits)
	{
		std::cout << "GraphicsContext Traits are null." << std::endl;
		return;
	}
	std::ostringstream oss;
	oss << "x: " << traits->x << std::endl;
	oss << "y: " << traits->y << std::endl;
	oss << "width: " << traits->width << std::endl;
	oss << "height: " << traits->height << std::endl;
	oss << "Window Decoration: " << (traits->windowDecoration ? "true" : "false") << std::endl;
	oss << "Supports Resize: " << (traits->supportsResize ? "true" : "false") << std::endl;
	oss << "Red Bits: " << traits->red << std::endl;
	oss << "Green Bits: " << traits->green << std::endl;
	oss << "Blue Bits: " << traits->blue << std::endl;
	oss << "Alpha Bits: " << traits->alpha << std::endl;
	oss << "Depth Bits: " << traits->depth << std::endl;
	oss << "Stencil Bits: " << traits->stencil << std::endl;
	oss << "Sample Buffers: " << traits->sampleBuffers << std::endl;
	oss << "Samples: " << traits->samples << std::endl;
	oss << "Pbuffer: " << (traits->pbuffer ? "true" : "false") << std::endl;
	oss << "Quad Buffer Stereo: " << (traits->quadBufferStereo ? "true" : "false") << std::endl;
	oss << "Double Buffer: " << (traits->doubleBuffer ? "true" : "false") << std::endl;
	oss << "VSync: " << (traits->vsync ? "true" : "false") << std::endl;
	oss << "Window Name: " << traits->windowName << std::endl;
	oss << "Windowing System Preference: " << traits->windowingSystemPreference << std::endl;

	if (traits->sharedContext.valid())
	{
		oss << "Shared Context Pointer: " << traits->sharedContext.get() << std::endl;
	}
	else
	{
		oss << "Shared Context Pointer: none" << std::endl;
	}
	std::cout << oss.str() << std::endl;
	std::cerr << "---------------------" << std::endl;
}


void osg_util::printScreenStruct(const opencover::screenStruct& s)
{
	std::ostringstream oss;
	oss << "hsize = " << s.hsize << std::endl
		<< "vsize = " << s.vsize << std::endl
		<< "configuredHsize = " << s.configuredHsize << std::endl
		<< "configuredVsize = " << s.configuredVsize << std::endl
		<< "xyz = " << osg_util::vec3ToString(s.xyz) << std::endl
		<< "hpr = " << osg_util::vec3ToString(s.hpr) << std::endl
		<< "name = \"" << s.name << "\"" << std::endl
		<< "render = " << (s.render ? "true" : "false") << std::endl
		<< "lTan = " << s.lTan << std::endl
		<< "rTan = " << s.rTan << std::endl
		<< "tTan = " << s.tTan << std::endl
		<< "bTan = " << s.bTan;
	std::cout << oss.str() << std::endl;
}


void osg_util::printChannelStruct(const opencover::channelStruct& ch)
{
	std::ostringstream oss;
	oss << "name = \"" << ch.name << "\"" << std::endl
		<< "PBONum = " << ch.PBONum << std::endl
		<< "viewportNum = " << ch.viewportNum << std::endl
		<< "screenNum = " << ch.screenNum << std::endl
		<< "camera = " << ch.camera.get() << std::endl
		<< "ds = " << ch.ds << std::endl
		<< "stereo = " << (ch.stereo ? "true" : "false") << std::endl
		<< "stereoMode = " << ch.stereoMode << std::endl
		<< "fixedViewer = " << (ch.fixedViewer ? "true" : "false") << std::endl
		<< "stereoOffset = " << ch.stereoOffset << std::endl
		<< "leftView = " << osg_util::matrixdToString(ch.leftView) << std::endl
		<< "rightView = " << osg_util::matrixdToString(ch.rightView) << std::endl
		<< "leftProj = " << osg_util::matrixdToString(ch.leftProj) << std::endl
		<< "rightProj = " << osg_util::matrixdToString(ch.rightProj);
	std::cout << oss.str() << std::endl;
}


void osg_util::printPBOStruct(const opencover::PBOStruct& pbo)
{
	std::ostringstream oss;
	oss << "PBOsx = " << pbo.PBOsx << std::endl
		<< "PBOsy = " << pbo.PBOsy << std::endl
		<< "windowNum = " << pbo.windowNum << std::endl
		<< "renderTargetTexture = " << pbo.renderTargetTexture.get();
	std::cout << oss.str() << std::endl;
}


void osg_util::printAngleStruct(const opencover::angleStruct& a)
{
	std::ostringstream oss;
	oss << "analogInput = " << a.analogInput << std::endl
		<< "cmin = " << a.cmin << std::endl
		<< "cmax = " << a.cmax << std::endl
		<< "minangle = " << a.minangle << std::endl
		<< "maxangle = " << a.maxangle << std::endl
		<< "screen = " << a.screen << std::endl
		<< "value = " << static_cast<const void*>(a.value) << std::endl
		<< "hpr = " << a.hpr;
	std::cout << oss.str() << std::endl;
}


void osg_util::printWindowStruct(const opencover::windowStruct& w)
{
	std::ostringstream oss;
	oss << "ox = " << w.ox << std::endl
		<< "oy = " << w.oy << std::endl
		<< "sx = " << w.sx << std::endl
		<< "sy = " << w.sy << std::endl
		<< "context = " << w.context.get() << std::endl
		<< "window = " << w.window.get() << std::endl
		<< "pipeNum = " << w.pipeNum << std::endl
		<< "name = \"" << w.name << "\"" << std::endl
		<< "decoration = " << (w.decoration ? "true" : "false") << std::endl
		<< "resize = " << (w.resize ? "true" : "false") << std::endl
		<< "stereo = " << (w.stereo ? "true" : "false") << std::endl
		<< "embedded = " << (w.embedded ? "true" : "false") << std::endl
		<< "pbuffer = " << (w.pbuffer ? "true" : "false") << std::endl
		<< "doublebuffer = " << (w.doublebuffer ? "true" : "false") << std::endl
		<< "swapGroup = " << w.swapGroup << std::endl
		<< "swapBarrier = " << w.swapBarrier << std::endl
		<< "screenNum = " << w.screenNum << std::endl
		<< "type = \"" << w.type << "\"" << std::endl
		<< "windowPlugin = " << w.windowPlugin;
	std::cout << oss.str() << std::endl;
}


void osg_util::printViewportStruct(const opencover::viewportStruct& vp)
{
	std::ostringstream oss;
	oss << "mode = ";
	switch (vp.mode)
	{
	case opencover::viewportStruct::Channel: oss << "Channel"; break;
	case opencover::viewportStruct::PBO: oss << "PBO"; break;
	case opencover::viewportStruct::TridelityML: oss << "TridelityML"; break;
	case opencover::viewportStruct::TridelityMV: oss << "TridelityMV"; break;
	default: oss << "Unknown"; break;
	}
	oss << std::endl
		<< "window = " << vp.window << std::endl
		<< "PBOnum = " << vp.PBOnum << std::endl
		<< "sourceXMin = " << vp.sourceXMin << std::endl
		<< "sourceYMin = " << vp.sourceYMin << std::endl
		<< "sourceXMax = " << vp.sourceXMax << std::endl
		<< "sourceYMax = " << vp.sourceYMax << std::endl
		<< "viewportXMin = " << vp.viewportXMin << std::endl
		<< "viewportYMin = " << vp.viewportYMin << std::endl
		<< "viewportXMax = " << vp.viewportXMax << std::endl
		<< "viewportYMax = " << vp.viewportYMax << std::endl
		<< "distortMeshName = \"" << vp.distortMeshName << "\"" << std::endl
		<< "blendingTextureName = \"" << vp.blendingTextureName << "\"" << std::endl
		<< "pbos = [";
	for (size_t i = 0; i < vp.pbos.size(); ++i)
	{
		oss << vp.pbos[i];
		if (i < vp.pbos.size() - 1)
			oss << ", ";
	}
	oss << "]";
	std::cout << oss.str() << std::endl;
}


void osg_util::printBlendingTextureStruct(const opencover::blendingTextureStruct& bt)
{
	std::ostringstream oss;
	oss << "window = " << bt.window << std::endl
		<< "viewportXMin = " << bt.viewportXMin << std::endl
		<< "viewportYMin = " << bt.viewportYMin << std::endl
		<< "viewportXMax = " << bt.viewportXMax << std::endl
		<< "viewportYMax = " << bt.viewportYMax << std::endl
		<< "blendingTextureName = \"" << bt.blendingTextureName << "\"";
	std::cout << oss.str() << std::endl;
}

void osg_util::printPipeStruct(const opencover::pipeStruct& p)
{
	std::ostringstream oss;
	oss << "x11DisplayNum = " << p.x11DisplayNum << std::endl
		<< "x11ScreenNum = " << p.x11ScreenNum << std::endl
		<< "x11DisplayHost = \"" << p.x11DisplayHost << "\"" << std::endl
		<< "useDISPLAY = " << (p.useDISPLAY ? "true" : "false");
	std::cout << oss.str() << std::endl;
}

void osg_util::printCoVRConfigOverview()
{
	opencover::coVRConfig* config = opencover::coVRConfig::instance();
	if (!config)
	{
		std::cerr << "coVRConfig::instance() is null." << std::endl;
		return;
	}

	std::cout << "Configured with "
		<< config->numScreens() << " screens, " << std::endl
		<< config->numWindows() << " windows, " << std::endl
		<< config->numChannels() << " channels, " << std::endl
		<< config->numViewports() << " viewports, " << std::endl
		<< config->numBlendingTextures() << " blending textures, " << std::endl
		<< config->numPBOs() << " PBOs." << std::endl << std::endl;

	std::cout << "=== Screens ===" << std::endl;
	for (size_t i = 0; i < config->screens.size(); ++i)
	{
		std::cout << "Screen[" << i << "]: ";
		osg_util::printScreenStruct(config->screens[i]);
	}
	std::cout << std::endl;

	std::cout << "=== Windows ===" << std::endl;
	for (size_t i = 0; i < config->windows.size(); ++i)
	{
		std::cout << "Window[" << i << "]: ";
		osg_util::printWindowStruct(config->windows[i]);
	}
	std::cout << std::endl;

	std::cout << "=== Channels ===" << std::endl;
	for (size_t i = 0; i < config->channels.size(); ++i)
	{
		std::cout << "Channel[" << i << "]: ";
		osg_util::printChannelStruct(config->channels[i]);
	}
	std::cout << std::endl;

	std::cout << "=== Viewports ===" << std::endl;
	for (size_t i = 0; i < config->viewports.size(); ++i)
	{
		std::cout << "Viewport[" << i << "]: ";
		osg_util::printViewportStruct(config->viewports[i]);
	}
	std::cout << std::endl;

	std::cout << "=== Blending Textures ===" << std::endl;
	for (size_t i = 0; i < config->blendingTextures.size(); ++i)
	{
		std::cout << "BlendingTexture[" << i << "]: ";
		osg_util::printBlendingTextureStruct(config->blendingTextures[i]);
	}
	std::cout << std::endl;

	std::cout << "=== PBOs ===" << std::endl;
	for (size_t i = 0; i < config->PBOs.size(); ++i)
	{
		std::cout << "PBO[" << i << "]: ";
		osg_util::printPBOStruct(config->PBOs[i]);
	}
	std::cout << std::endl;

	std::cout << "=== Pipes ===" << std::endl;
	for (size_t i = 0; i < config->pipes.size(); ++i)
	{
		std::cout << "Pipe[" << i << "]: ";
		osg_util::printPipeStruct(config->pipes[i]);
	}
	std::cout << std::endl;
}


void osg_util::printAllGraphicsContextsAndWindows()
{
	osg::GraphicsContext::GraphicsContexts contexts = osg::GraphicsContext::getAllRegisteredGraphicsContexts();
	std::cout << "=== Registered GraphicsContexts (" << contexts.size() << ") ===" << std::endl;
	for (size_t i = 0; i < contexts.size(); ++i)
	{
		std::cout << "Context[" << i << "]:" << std::endl;
		osg_util::printGraphicsContextAttributes(contexts[i]);
		std::cout << std::endl;
	}

	std::cout << "=== Registered Windows ===" << std::endl;
	if (opencover::coVRConfig::instance())
	{
		const auto& windows = opencover::coVRConfig::instance()->windows; // Angenommen, windows ist ein std::vector<WindowInfo>
		std::cout << "Anzahl Fenster: " << windows.size() << std::endl;
		for (size_t i = 0; i < windows.size(); ++i)
		{
			std::cout << "Window[" << i << "]:" << std::endl;
			std::cout << "  Name: " << windows[i].name << std::endl;
			std::cout << "  Context Pointer: " << windows[i].context << std::endl;
			//std::cout << "  Resolution: " << windows[i].width << "x" << windows[i].height << std::endl;
			std::cout << std::endl;
		}
	}
	else
	{
		std::cout << "coVRConfig ist nicht verfügbar." << std::endl;
	}
}


void osg_util::printCandidateVAO(GLuint candidate)
{
	if (!glIsVertexArray(candidate))
		return;

	std::cout << "VAO Handle: " << candidate << std::endl;

	GLint oldVAO = 0;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO);

	glBindVertexArray(candidate);

	GLint currentEAB = 0;
	glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &currentEAB);
	//std::cout << "  Element Array Buffer Binding: " << currentEAB << std::endl;

	GLint maxAttribs = 0;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxAttribs);

	int activeAttribCount = 0;
	for (GLint i = 0; i < maxAttribs; ++i)
	{
		GLint enabled = 0;
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
		if (enabled)
			activeAttribCount++;
	}
	std::cout << "  Active Vertex Attributes: " << activeAttribCount << "/" << maxAttribs << std::endl;

	for (GLint i = 0; i < maxAttribs; ++i)
	{
		GLint enabled = 0;
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
		if (enabled)
		{
			GLint size = 0, type = 0, stride = 0, bufferBinding = 0;
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size);
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type);
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &stride);
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &bufferBinding);
			void* pointer = nullptr;
			glGetVertexAttribPointerv(i, GL_VERTEX_ATTRIB_ARRAY_POINTER, &pointer);

			std::ostringstream oss;
			oss << "  Attribute " << i << ":" << std::endl;
			oss << "    Size: " << size << std::endl;
			oss << "    Type: " << type << std::endl;
			oss << "    Stride: " << stride << std::endl;
			oss << "    Offset Pointer: " << pointer << std::endl;
			oss << "    Element Array Buffer Binding: " << currentEAB << std::endl;
			oss << "    Array Buffer Binding: " << bufferBinding << std::endl;
			std::cout << oss.str();
		}
	}
	glBindVertexArray(oldVAO);
	std::cout << std::endl;
}


void osg_util::printAllExistingVAOs(GLuint maxID)
{
	std::cout << "=== Overview of all VAOs ===" << std::endl;
	GLState::printCurrentContext();
	GLint activeVAO = 0;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &activeVAO);
	std::cout << "Active VAO: " << activeVAO << std::endl << std::endl;
	for (GLuint i = 1; i <= maxID; ++i)
	{
		if (glIsVertexArray(i))
		{
			printCandidateVAO(i);
		}
	}
}


void osg_util::waitForOpenGLContext() {
	unsigned int max_wait_time_ms = 10000;
	auto start_time = std::chrono::steady_clock::now();
	bool context_ready = false;
	while (!context_ready) {
#ifdef WIN32
		if (wglGetCurrentContext() != nullptr) {
#else
		if (glXGetCurrentContext() != nullptr) {
#endif
			context_ready = true;
			break;
		}
		std::cout << "Warte auf OpenGL-Kontext..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Kürzere Wartezeit für bessere Reaktionsfähigkeit
		// Prüfen, ob die maximale Wartezeit überschritten wurde
		auto current_time = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
		if (elapsed_ms >= max_wait_time_ms) {
			std::cerr << "Fehler: OpenGL-Kontext nicht verfügbar nach " << max_wait_time_ms << " Millisekunden." << std::endl;
			return;
		}
	}

	std::cout << "OpenGL-Kontext ist jetzt bereit!" << std::endl;
	// Stellen Sie sicher, dass ein gültiger OpenGL-Kontext vorhanden ist
	const GLubyte* version = glGetString(GL_VERSION);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	const GLubyte* shadingLanguageVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	// Überprüfen, ob die Rückgabewerte gültig sind
	if (!version || !renderer || !vendor || !shadingLanguageVersion) {
		std::cerr << "Fehler beim Abrufen von OpenGL-Informationen. Ist der OpenGL-Kontext aktiv?" << std::endl;
		return;
	}

	// Profilmaske abrufen
	GLint profileMask = 0;
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profileMask);

	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		std::cerr << "OpenGL-Fehler beim Abrufen der Profilmaske: " << err << std::endl;
	}

	// Ausgabe der Informationen
	std::cout << "OpenGL-Version: " << version << std::endl;
	std::cout << "Renderer: " << renderer << std::endl;
	std::cout << "Anbieter: " << vendor << std::endl;
	std::cout << "GLSL-Version: " << shadingLanguageVersion << std::endl;

	// Profiltyp bestimmen
	std::cout << "OpenGL-Profil: ";
	if (profileMask & GL_CONTEXT_CORE_PROFILE_BIT)
		std::cout << "Core Profile" << std::endl;
	else if (profileMask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
		std::cout << "Compatibility Profile" << std::endl;
	else
		std::cout << "Unbekanntes Profil" << std::endl;
}
