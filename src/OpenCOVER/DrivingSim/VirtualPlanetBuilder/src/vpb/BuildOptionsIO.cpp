/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/BuildOptions>
#include <vpb/Serializer>

#include <iostream>
#include <string>
#include <map>
#include <set>

#include <osg/Version>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/io_utils>

#include <osgDB/ReadFile>
#include <osgDB/Registry>

#if OSG_MIN_VERSION_REQUIRED(3, 1, 1)
#define IS_BEGIN_BRACKET is.BEGIN_BRACKET
#define IS_END_BRACKET is.END_BRACKET
#define OS_BEGIN_BRACKET os.BEGIN_BRACKET
#define OS_END_BRACKET os.END_BRACKET
#else
#define IS_BEGIN_BRACKET osgDB::BEGIN_BRACKET
#define IS_END_BRACKET osgDB::END_BRACKET
#define OS_BEGIN_BRACKET osgDB::BEGIN_BRACKET
#define OS_END_BRACKET osgDB::END_BRACKET
#endif

using namespace vpb;

template <typename C>
class GeospatialExtentsSerializer : public vpb::Serializer
{
public:
    typedef GeospatialExtents V;
    typedef const V &P;
    typedef P (C::*GetterFunctionType)() const;
    typedef void (C::*SetterFunctionType)(P);

    GeospatialExtentsSerializer(const char *fieldName, P defaultValue, GetterFunctionType getter, SetterFunctionType setter)
        : _fieldName(fieldName)
        , _default(defaultValue)
        , _getter(getter)
        , _setter(setter)
    {
    }

    bool write(osgDB::Output &fw, const osg::Object &obj)
    {
        const C &object = static_cast<const C &>(obj);
        if (fw.getWriteOutDefaultValues() || _default != (object.*_getter)())
        {
            P value = (object.*_getter)();
            fw.indent() << _fieldName << " " << value._min[0] << " " << value._min[1] << " " << value._max[0] << " " << value._max[1] << std::endl;
        }

        return true;
    }

    bool read(osgDB::Input &fr, osg::Object &obj, bool &itrAdvanced)
    {
        C &object = static_cast<C &>(obj);
        V value;
        if (fr.read(_fieldName.c_str(), value._min[0], value._min[1], value._max[0], value._max[1]))
        {
            (object.*_setter)(value);
            itrAdvanced = true;
        }

        return true;
    }

    std::string _fieldName;
    V _default;
    GetterFunctionType _getter;
    SetterFunctionType _setter;
};

template <typename C, typename T, typename Itr>
class SetSerializer : public vpb::Serializer
{
public:
    typedef const T &(C::*GetterFunctionType)() const;
    typedef void (C::*SetterFunctionType)(const T &);

    SetSerializer(const char *fieldName, const T &defaultValue, GetterFunctionType getter, SetterFunctionType setter)
        : _fieldName(fieldName)
        , _default(defaultValue)
        , _getter(getter)
        , _setter(setter)
    {
    }

    bool write(osgDB::Output &fw, const osg::Object &obj)
    {
        const C &object = static_cast<const C &>(obj);
        if (fw.getWriteOutDefaultValues() || _default != (object.*_getter)())
        {
            const T &value = (object.*_getter)();
            if (!value.empty())
            {
                fw.indent() << _fieldName << " {" << std::endl;
                fw.moveIn();

                for (Itr itr = value.begin();
                     itr != value.end();
                     ++itr)
                {
                    fw.indent() << *itr << std::endl;
                }
                fw.moveOut();
                fw.indent() << "}" << std::endl;
            }
        }

        return true;
    }

    bool read(osgDB::Input &fr, osg::Object &obj, bool &itrAdvanced)
    {
        C &object = static_cast<C &>(obj);
        if (fr[0].matchWord(_fieldName.c_str()) && fr[1].isOpenBracket())
        {
            T value;

            int entry = fr[0].getNoNestedBrackets();

            fr += 2;

            while (!fr.eof() && fr[0].getNoNestedBrackets() > entry)
            {
                if (fr[0].isWord())
                    value.insert(fr[0].getStr());
                ++fr;
            }

            ++fr;

            (object.*_setter)(value);
            itrAdvanced = true;
        }

        return true;
    }

    std::string _fieldName;
    T _default;
    GetterFunctionType _getter;
    SetterFunctionType _setter;
};

#define VPB_CREATE_ENUM_SERIALIZER(CLASS, PROPERTY, PROTOTYPE)        \
    typedef vpb::EnumSerializer<CLASS, CLASS::PROPERTY> MySerializer; \
    osg::ref_ptr<MySerializer> serializer = new MySerializer(         \
        #PROPERTY,                                                    \
        PROTOTYPE.get##PROPERTY(),                                    \
        &CLASS::get##PROPERTY,                                        \
        &CLASS::set##PROPERTY)

#define VPB_CREATE_ENUM_SERIALIZER2(CLASS, NAME, PROPERTY, PROTOTYPE) \
    typedef vpb::EnumSerializer<CLASS, CLASS::PROPERTY> MySerializer; \
    osg::ref_ptr<MySerializer> serializer = new MySerializer(         \
        #NAME,                                                        \
        PROTOTYPE.get##NAME(),                                        \
        &CLASS::get##NAME,                                            \
        &CLASS::set##NAME)

#define VPB_ADD_ENUM_PROPERTY(PROPERTY)                            \
    VPB_CREATE_ENUM_SERIALIZER(BuildOptions, PROPERTY, prototype); \
    _serializerList.push_back(serializer.get())

#define VPB_ADD_ENUM_PROPERTY2(NAME, PROPERTY)                            \
    VPB_CREATE_ENUM_SERIALIZER2(BuildOptions, NAME, PROPERTY, prototype); \
    _serializerList.push_back(serializer.get())

#define VPB_ADD_STRING_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_STRING_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_UINT_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_UINT_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_INT_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_INT_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_FLOAT_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_FLOAT_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_DOUBLE_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_DOUBLE_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_VEC4_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_VEC4_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_BOOL_PROPERTY(PROPERTY) _serializerList.push_back(VPB_CREATE_BOOL_SERIALIZER(BuildOptions, PROPERTY, prototype))

#define VPB_ADD_ENUM_VALUE(VALUE) serializer->add(BuildOptions::VALUE, #VALUE)

#define VPB_ADD_ENUM_PROPERTY_TWO_VALUES(PROPERTY, VALUE1, VALUE2) \
    {                                                              \
        VPB_ADD_ENUM_PROPERTY(PROPERTY);                           \
        VPB_ADD_ENUM_VALUE(VALUE1);                                \
        VPB_ADD_ENUM_VALUE(VALUE2);                                \
    }

#define VPB_ADD_ENUM_PROPERTY_TWO_VALUES(PROPERTY, VALUE1, VALUE2) \
    {                                                              \
        VPB_ADD_ENUM_PROPERTY(PROPERTY);                           \
        VPB_ADD_ENUM_VALUE(VALUE1);                                \
        VPB_ADD_ENUM_VALUE(VALUE2);                                \
    }

#define VPB_ADD_ENUM_PROPERTY_THREE_VALUES(PROPERTY, VALUE1, VALUE2, VALUE3) \
    {                                                                        \
        VPB_ADD_ENUM_PROPERTY(PROPERTY);                                     \
        VPB_ADD_ENUM_VALUE(VALUE1);                                          \
        VPB_ADD_ENUM_VALUE(VALUE2);                                          \
        VPB_ADD_ENUM_VALUE(VALUE3);                                          \
    }

#define VPB_AEV VPB_ADD_ENUM_VALUE
#define VPB_AEP VPB_ADD_ENUM_PROPERTY
#define VPB_AEP2 VPB_ADD_ENUM_PROPERTY2

#define VPB_SCOPED_AEV(SCOPE, VALUE) serializer->add(SCOPE::VALUE, #VALUE)

class BuildOptionsLookUps
{
public:
    typedef std::list<osg::ref_ptr<vpb::Serializer> > SerializerList;
    SerializerList _serializerList;

    BuildOptionsLookUps()
    {
        BuildOptions prototype;

        VPB_ADD_STRING_PROPERTY(Directory);
        VPB_ADD_BOOL_PROPERTY(OutputTaskDirectories);
        VPB_ADD_STRING_PROPERTY(DestinationTileBaseName);
        VPB_ADD_STRING_PROPERTY(DestinationTileExtension);
        VPB_ADD_STRING_PROPERTY(DestinationImageExtension);
        VPB_ADD_BOOL_PROPERTY(PowerOfTwoImages);
        VPB_ADD_STRING_PROPERTY(ArchiveName);
        VPB_ADD_STRING_PROPERTY(IntermediateBuildName);
        VPB_ADD_STRING_PROPERTY(LogFileName);
        VPB_ADD_STRING_PROPERTY(TaskFileName);
        VPB_ADD_STRING_PROPERTY(CommentString);

        VPB_ADD_ENUM_PROPERTY_TWO_VALUES(DatabaseType, LOD_DATABASE, PagedLOD_DATABASE)

        VPB_ADD_ENUM_PROPERTY_THREE_VALUES(GeometryType, HEIGHT_FIELD, POLYGONAL, TERRAIN)
        VPB_ADD_ENUM_PROPERTY_THREE_VALUES(MipMappingMode, NO_MIP_MAPPING, MIP_MAPPING_HARDWARE, MIP_MAPPING_IMAGERY)

        {
            VPB_AEP(TextureType);
            VPB_AEV(RGB_24);
            VPB_AEV(RGBA);
            VPB_AEV(RGB_16);
            VPB_AEV(RGBA_16);
            VPB_AEV(RGB_S3TC_DXT1);
            VPB_AEV(RGBA_S3TC_DXT1);
            VPB_AEV(RGBA_S3TC_DXT3);
            VPB_AEV(RGBA_S3TC_DXT5);
            VPB_AEV(ARB_COMPRESSED);
            VPB_AEV(COMPRESSED_TEXTURE);
            VPB_AEV(COMPRESSED_RGBA_TEXTURE);
            VPB_AEV(RGB32F);
            VPB_AEV(RGBA32F);
        }

        VPB_ADD_UINT_PROPERTY(MaximumTileImageSize);
        VPB_ADD_UINT_PROPERTY(MaximumTileTerrainSize);

        VPB_ADD_FLOAT_PROPERTY(MaximumVisibleDistanceOfTopLevel);
        VPB_ADD_FLOAT_PROPERTY(RadiusToMaxVisibleDistanceRatio);
        VPB_ADD_FLOAT_PROPERTY(VerticalScale);
        VPB_ADD_FLOAT_PROPERTY(SkirtRatio);
        VPB_ADD_UINT_PROPERTY(ImageryQuantization);
        VPB_ADD_BOOL_PROPERTY(ImageryErrorDiffusion);
        VPB_ADD_FLOAT_PROPERTY(MaxAnisotropy);

        VPB_ADD_BOOL_PROPERTY(BuildOverlays);
        VPB_ADD_BOOL_PROPERTY(ReprojectSources);
        VPB_ADD_BOOL_PROPERTY(GenerateTiles);
        VPB_ADD_BOOL_PROPERTY(ConvertFromGeographicToGeocentric);
        VPB_ADD_BOOL_PROPERTY(UseLocalTileTransform);
        VPB_ADD_BOOL_PROPERTY(SimplifyTerrain);
        VPB_ADD_BOOL_PROPERTY(DecorateGeneratedSceneGraphWithCoordinateSystemNode);
        VPB_ADD_BOOL_PROPERTY(DecorateGeneratedSceneGraphWithMultiTextureControl);
        VPB_ADD_BOOL_PROPERTY(WriteNodeBeforeSimplification);

        VPB_ADD_VEC4_PROPERTY(DefaultColor);

        VPB_ADD_BOOL_PROPERTY(UseInterpolatedImagerySampling);
        VPB_ADD_BOOL_PROPERTY(UseInterpolatedTerrainSampling);

        VPB_ADD_STRING_PROPERTY(DestinationCoordinateSystem);
        VPB_ADD_STRING_PROPERTY(DestinationCoordinateSystemFormat);
        VPB_ADD_DOUBLE_PROPERTY(RadiusPolar);
        VPB_ADD_DOUBLE_PROPERTY(RadiusEquator);

        _serializerList.push_back(new GeospatialExtentsSerializer<BuildOptions>(
            "DestinationExtents",
            prototype.getDestinationExtents(),
            &BuildOptions::getDestinationExtents,
            &BuildOptions::setDestinationExtents));

        VPB_ADD_UINT_PROPERTY(MaximumNumOfLevels);

        VPB_ADD_UINT_PROPERTY(DistributedBuildSplitLevel);
        VPB_ADD_UINT_PROPERTY(DistributedBuildSecondarySplitLevel);
        VPB_ADD_BOOL_PROPERTY(RecordSubtileFileNamesOnLeafTile);
        VPB_ADD_BOOL_PROPERTY(GenerateSubtile);
        VPB_ADD_UINT_PROPERTY(SubtileLevel);
        VPB_ADD_UINT_PROPERTY(SubtileX);
        VPB_ADD_UINT_PROPERTY(SubtileY);

        {
            VPB_AEP(NotifyLevel);
            VPB_AEV(ALWAYS);
            VPB_AEV(FATAL);
            VPB_AEV(WARN);
            VPB_AEV(NOTICE);
            VPB_AEV(INFO);
            VPB_AEV(DEBUG_INFO);
            VPB_AEV(DEBUG_FP);
        }

        VPB_ADD_BOOL_PROPERTY(DisableWrites);

        VPB_ADD_FLOAT_PROPERTY(NumReadThreadsToCoresRatio);
        VPB_ADD_FLOAT_PROPERTY(NumWriteThreadsToCoresRatio);

        VPB_ADD_STRING_PROPERTY(BuildOptionsString);
        VPB_ADD_STRING_PROPERTY(WriteOptionsString);

        {
            VPB_AEP(LayerInheritance);
            VPB_AEV(INHERIT_LOWEST_AVAILABLE);
            VPB_AEV(INHERIT_NEAREST_AVAILABLE);
            VPB_AEV(NO_INHERITANCE);
        }

        VPB_ADD_BOOL_PROPERTY(AbortTaskOnError);
        VPB_ADD_BOOL_PROPERTY(AbortRunOnError);

        {
            VPB_AEP2(DefaultImageLayerOutputPolicy, LayerOutputPolicy);
            VPB_AEV(INLINE);
            VPB_AEV(EXTERNAL_LOCAL_DIRECTORY);
            VPB_AEV(EXTERNAL_SET_DIRECTORY);
        }
        {
            VPB_AEP2(DefaultElevationLayerOutputPolicy, LayerOutputPolicy);
            VPB_AEV(INLINE);
            VPB_AEV(EXTERNAL_LOCAL_DIRECTORY);
            VPB_AEV(EXTERNAL_SET_DIRECTORY);
        }

        {
            VPB_AEP2(OptionalImageLayerOutputPolicy, LayerOutputPolicy);
            VPB_AEV(INLINE);
            VPB_AEV(EXTERNAL_LOCAL_DIRECTORY);
            VPB_AEV(EXTERNAL_SET_DIRECTORY);
        }
        {
            VPB_AEP2(OptionalElevationLayerOutputPolicy, LayerOutputPolicy);
            VPB_AEV(INLINE);
            VPB_AEV(EXTERNAL_LOCAL_DIRECTORY);
            VPB_AEV(EXTERNAL_SET_DIRECTORY);
        }

        _serializerList.push_back(new SetSerializer<BuildOptions, BuildOptions::OptionalLayerSet, BuildOptions::OptionalLayerSet::const_iterator>(
            "OptionalLayerSet",
            prototype.getOptionalLayerSet(),
            &BuildOptions::getOptionalLayerSet,
            &BuildOptions::setOptionalLayerSet));

        VPB_ADD_UINT_PROPERTY(RevisionNumber);

        {
            VPB_AEP(BlendingPolicy);
            VPB_AEV(INHERIT);
            VPB_AEV(DO_NOT_SET_BLENDING);
            VPB_AEV(ENABLE_BLENDING);
            VPB_AEV(ENABLE_BLENDING_WHEN_ALPHA_PRESENT);
        }

        VPB_ADD_ENUM_PROPERTY_THREE_VALUES(CompressionMethod, GL_DRIVER, NVTT, NVTT_NOCUDA)
    }

    bool read(osgDB::Input &fr, BuildOptions &db, bool &itrAdvanced)
    {
        for (SerializerList::iterator itr = _serializerList.begin();
             itr != _serializerList.end();
             ++itr)
        {
            (*itr)->read(fr, db, itrAdvanced);
        }

        return true;
    }

    bool write(osgDB::Output &fw, const BuildOptions &db)
    {
        bool result = false;
        for (SerializerList::iterator itr = _serializerList.begin();
             itr != _serializerList.end();
             ++itr)
        {
            if ((*itr)->write(fw, db))
                result = true;
        }
        return result;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  BuildOptions IO support
//

bool BuildOptions_readLocalData(osg::Object &obj, osgDB::Input &fr);
bool BuildOptions_writeLocalData(const osg::Object &obj, osgDB::Output &fw);

osgDB::TemplateRegisterDotOsgWrapperProxy<BuildOptionsLookUps> BuildOptions_Proxy(
    new vpb::BuildOptions,
    "BuildOptions",
    "BuildOptions Object",
    BuildOptions_readLocalData,
    BuildOptions_writeLocalData);

bool BuildOptions_readLocalData(osg::Object &obj, osgDB::Input &fr)
{
    vpb::BuildOptions &gt = static_cast<vpb::BuildOptions &>(obj);
    bool itrAdvanced = false;

    BuildOptions_Proxy.read(fr, gt, itrAdvanced);

    return itrAdvanced;
}

bool BuildOptions_writeLocalData(const osg::Object &obj, osgDB::Output &fw)
{
    const vpb::BuildOptions &db = static_cast<const vpb::BuildOptions &>(obj);

    BuildOptions_Proxy.write(fw, db);

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
// New serializers ImageOptions
//
#include <osgDB/ObjectWrapper>
#include <osgDB/InputStream>
#include <osgDB/OutputStream>

namespace ImageOptionsIO
{

REGISTER_OBJECT_WRAPPER(ImageOptions,
                        new vpb::ImageOptions,
                        vpb::ImageOptions,
                        "osg::Object vpb::ImageOptions")
{
    ADD_STRING_SERIALIZER(DestinationImageExtension, ".dds");
    ADD_UINT_SERIALIZER(MaximumTileImageSize, 256);
    ADD_VEC4_SERIALIZER(DefaultColor, osg::Vec4(0.5f, 0.5f, 1.0f, 1.0f));
    ADD_BOOL_SERIALIZER(PowerOfTwoImages, true);
    ADD_BOOL_SERIALIZER(UseInterpolatedImagerySampling, true);

    BEGIN_ENUM_SERIALIZER(TextureType, COMPRESSED_TEXTURE);
    ADD_ENUM_VALUE(RGB_24);
    ADD_ENUM_VALUE(RGBA);
    ADD_ENUM_VALUE(RGB_16);
    ADD_ENUM_VALUE(RGBA_16);
    ADD_ENUM_VALUE(RGB_S3TC_DXT1);
    ADD_ENUM_VALUE(RGBA_S3TC_DXT1);
    ADD_ENUM_VALUE(RGBA_S3TC_DXT3);
    ADD_ENUM_VALUE(RGBA_S3TC_DXT5);
    ADD_ENUM_VALUE(ARB_COMPRESSED);
    ADD_ENUM_VALUE(COMPRESSED_TEXTURE);
    ADD_ENUM_VALUE(COMPRESSED_RGBA_TEXTURE);
    ADD_ENUM_VALUE(RGB32F);
    ADD_ENUM_VALUE(RGBA32F);
    END_ENUM_SERIALIZER();

    ADD_UINT_SERIALIZER(ImageryQuantization, 0);
    ADD_BOOL_SERIALIZER(ImageryErrorDiffusion, false);
    ADD_FLOAT_SERIALIZER(MaxAnisotropy, 1.0);

    BEGIN_ENUM_SERIALIZER(MipMappingMode, MIP_MAPPING_IMAGERY);
    ADD_ENUM_VALUE(NO_MIP_MAPPING);
    ADD_ENUM_VALUE(MIP_MAPPING_HARDWARE);
    ADD_ENUM_VALUE(MIP_MAPPING_IMAGERY);
    END_ENUM_SERIALIZER();
}
}

namespace BuildOptionsIO
{

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
// New serializers BuildOptions
//
static bool checkOptionalLayerSet(const vpb::BuildOptions &bo)
{
    return !(bo.getOptionalLayerSet().empty());
}

static bool readOptionalLayerSet(osgDB::InputStream &is, vpb::BuildOptions &bo)
{
    vpb::BuildOptions::OptionalLayerSet &ols = bo.getOptionalLayerSet();
    unsigned int size = 0;
    is >> size >> IS_BEGIN_BRACKET;
    for (unsigned int i = 0; i < size; ++i)
    {
        std::string value;
        is.readWrappedString(value);
        ols.insert(value);
    }
    is >> IS_END_BRACKET;
    return true;
}

static bool writeOptionalLayerSet(osgDB::OutputStream &os, const vpb::BuildOptions &bo)
{
    const vpb::BuildOptions::OptionalLayerSet &ols = bo.getOptionalLayerSet();
    unsigned int size = ols.size();
    os << size << OS_BEGIN_BRACKET << std::endl;
    for (vpb::BuildOptions::OptionalLayerSet::const_iterator itr = ols.begin(); itr != ols.end(); ++itr)
    {
        os.writeWrappedString(*itr);
        os << std::endl;
    }
    os << OS_END_BRACKET << std::endl;
    return true;
}

static bool checkLayerImageOptions(const vpb::BuildOptions &bo)
{
    return bo.getNumLayerImageOptions() > 0;
}

static bool readLayerImageOptions(osgDB::InputStream &is, vpb::BuildOptions &bo)
{
    unsigned int size = 0;
    is >> size >> IS_BEGIN_BRACKET;
    for (unsigned int i = 0; i < size; ++i)
    {
        vpb::ImageOptions *imageOptions = dynamic_cast<vpb::ImageOptions *>(is.readObject());
        if (imageOptions)
            bo.setLayerImageOptions(i, imageOptions);
    }
    is >> IS_END_BRACKET;
    return true;
}

static bool writeLayerImageOptions(osgDB::OutputStream &os, const vpb::BuildOptions &bo)
{
    unsigned int size = bo.getNumLayerImageOptions();
    os << size << OS_BEGIN_BRACKET << std::endl;
    for (unsigned int i = 0; i < size; ++i)
    {
        OSG_NOTICE << "Writing out ImageOptions " << bo.getLayerImageOptions(i) << " " << bo.getLayerImageOptions(i)->className() << std::endl;
        os.writeObject(bo.getLayerImageOptions(i));
    }
    os << OS_END_BRACKET << std::endl;
    return true;
}

static bool checkDestinationExtents(const vpb::BuildOptions &bo)
{
    return bo.getDestinationExtents().valid();
}

static bool readDestinationExtents(osgDB::InputStream &is, vpb::BuildOptions &bo)
{
    double xMin, xMax, yMin, yMax;
    bool isGeo;
    is >> IS_BEGIN_BRACKET;
    is >> xMin;
    is >> yMin;
    is >> xMax;
    is >> yMax;
    is >> isGeo;
    is >> IS_END_BRACKET;
    GeospatialExtents ext(xMin, yMin, xMax, yMax, isGeo);
    bo.setDestinationExtents(ext);
    return true;
}

static bool writeDestinationExtents(osgDB::OutputStream &os, const vpb::BuildOptions &bo)
{
    GeospatialExtents ext(bo.getDestinationExtents());
    os << OS_BEGIN_BRACKET << std::endl;
    os << ext.xMin();
    os << ext.yMin();
    os << ext.xMax();
    os << ext.yMax();
    os << ext._isGeographic;
    os << std::endl;

    os << OS_END_BRACKET << std::endl;
    return true;
}

REGISTER_OBJECT_WRAPPER(BuildOptions,
                        new vpb::BuildOptions,
                        vpb::BuildOptions,
                        "osg::Object vpb::ImageOptions vpb::BuildOptions")
{
    ADD_STRING_SERIALIZER(Directory, "");
    ADD_STRING_SERIALIZER(DestinationTileBaseName, "output");
    ADD_STRING_SERIALIZER(DestinationTileExtension, ".osgb");
    ADD_BOOL_SERIALIZER(OutputTaskDirectories, true);
    ADD_STRING_SERIALIZER(ArchiveName, "");
    ADD_STRING_SERIALIZER(IntermediateBuildName, "");
    ADD_STRING_SERIALIZER(LogFileName, "");
    ADD_STRING_SERIALIZER(TaskFileName, "");
    ADD_STRING_SERIALIZER(CommentString, "");

    ADD_UINT_SERIALIZER(MaximumTileTerrainSize, 64);
    ADD_FLOAT_SERIALIZER(MaximumVisibleDistanceOfTopLevel, 1e10);
    ADD_FLOAT_SERIALIZER(RadiusToMaxVisibleDistanceRatio, 7.0f);
    ADD_FLOAT_SERIALIZER(VerticalScale, 1.0f);
    ADD_FLOAT_SERIALIZER(SkirtRatio, 0.02f);

    ADD_BOOL_SERIALIZER(UseInterpolatedTerrainSampling, true);
    ADD_BOOL_SERIALIZER(BuildOverlays, false);
    ADD_BOOL_SERIALIZER(ReprojectSources, true);
    ADD_BOOL_SERIALIZER(GenerateTiles, true);
    ADD_BOOL_SERIALIZER(ConvertFromGeographicToGeocentric, false);
    ADD_BOOL_SERIALIZER(UseLocalTileTransform, true);
    ADD_BOOL_SERIALIZER(SimplifyTerrain, true);

    ADD_BOOL_SERIALIZER(DecorateGeneratedSceneGraphWithCoordinateSystemNode, true);
    ADD_BOOL_SERIALIZER(DecorateGeneratedSceneGraphWithMultiTextureControl, true);
    ADD_BOOL_SERIALIZER(WriteNodeBeforeSimplification, false);

    BEGIN_ENUM_SERIALIZER(DatabaseType, PagedLOD_DATABASE);
    ADD_ENUM_VALUE(LOD_DATABASE);
    ADD_ENUM_VALUE(PagedLOD_DATABASE);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER(GeometryType, TERRAIN);
    ADD_ENUM_VALUE(HEIGHT_FIELD);
    ADD_ENUM_VALUE(POLYGONAL);
    ADD_ENUM_VALUE(TERRAIN);
    END_ENUM_SERIALIZER();

    ADD_STRING_SERIALIZER(DestinationCoordinateSystem, "");
    ADD_DOUBLE_SERIALIZER(RadiusEquator, osg::WGS_84_RADIUS_EQUATOR);
    ADD_DOUBLE_SERIALIZER(RadiusPolar, osg::WGS_84_RADIUS_POLAR);

    ADD_UINT_SERIALIZER(MaximumNumOfLevels, 30);
    ADD_UINT_SERIALIZER(DistributedBuildSplitLevel, 0);
    ADD_UINT_SERIALIZER(DistributedBuildSecondarySplitLevel, 0);
    ADD_BOOL_SERIALIZER(RecordSubtileFileNamesOnLeafTile, false);
    ADD_BOOL_SERIALIZER(GenerateSubtile, false);
    ADD_UINT_SERIALIZER(SubtileLevel, 0);
    ADD_UINT_SERIALIZER(SubtileX, 0);
    ADD_UINT_SERIALIZER(SubtileY, 0);

    BEGIN_ENUM_SERIALIZER(NotifyLevel, NOTICE);
    ADD_ENUM_VALUE(ALWAYS);
    ADD_ENUM_VALUE(FATAL);
    ADD_ENUM_VALUE(WARN);
    ADD_ENUM_VALUE(NOTICE);
    ADD_ENUM_VALUE(INFO);
    ADD_ENUM_VALUE(DEBUG_INFO);
    ADD_ENUM_VALUE(DEBUG_FP);
    END_ENUM_SERIALIZER();

    ADD_BOOL_SERIALIZER(DisableWrites, false);
    ADD_FLOAT_SERIALIZER(NumReadThreadsToCoresRatio, 0.0f);
    ADD_FLOAT_SERIALIZER(NumWriteThreadsToCoresRatio, 0.0f);

    ADD_STRING_SERIALIZER(BuildOptionsString, "");
    ADD_STRING_SERIALIZER(WriteOptionsString, "");

    BEGIN_ENUM_SERIALIZER(LayerInheritance, INHERIT_NEAREST_AVAILABLE);
    ADD_ENUM_VALUE(INHERIT_LOWEST_AVAILABLE);
    ADD_ENUM_VALUE(INHERIT_NEAREST_AVAILABLE);
    ADD_ENUM_VALUE(NO_INHERITANCE);
    END_ENUM_SERIALIZER();

    ADD_BOOL_SERIALIZER(AbortTaskOnError, true);
    ADD_BOOL_SERIALIZER(AbortRunOnError, false);

    BEGIN_ENUM_SERIALIZER2(DefaultImageLayerOutputPolicy, vpb::BuildOptions::LayerOutputPolicy, INLINE);
    ADD_ENUM_VALUE(INLINE);
    ADD_ENUM_VALUE(EXTERNAL_LOCAL_DIRECTORY);
    ADD_ENUM_VALUE(EXTERNAL_SET_DIRECTORY);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER2(DefaultElevationLayerOutputPolicy, vpb::BuildOptions::LayerOutputPolicy, INLINE);
    ADD_ENUM_VALUE(INLINE);
    ADD_ENUM_VALUE(EXTERNAL_LOCAL_DIRECTORY);
    ADD_ENUM_VALUE(EXTERNAL_SET_DIRECTORY);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER2(OptionalImageLayerOutputPolicy, vpb::BuildOptions::LayerOutputPolicy, EXTERNAL_SET_DIRECTORY);
    ADD_ENUM_VALUE(INLINE);
    ADD_ENUM_VALUE(EXTERNAL_LOCAL_DIRECTORY);
    ADD_ENUM_VALUE(EXTERNAL_SET_DIRECTORY);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER2(OptionalElevationLayerOutputPolicy, vpb::BuildOptions::LayerOutputPolicy, EXTERNAL_SET_DIRECTORY);
    ADD_ENUM_VALUE(INLINE);
    ADD_ENUM_VALUE(EXTERNAL_LOCAL_DIRECTORY);
    ADD_ENUM_VALUE(EXTERNAL_SET_DIRECTORY);
    END_ENUM_SERIALIZER();

    ADD_USER_SERIALIZER(OptionalLayerSet);

    ADD_UINT_SERIALIZER(RevisionNumber, 0);

    BEGIN_ENUM_SERIALIZER(BlendingPolicy, INHERIT);
    ADD_ENUM_VALUE(INHERIT);
    ADD_ENUM_VALUE(DO_NOT_SET_BLENDING);
    ADD_ENUM_VALUE(ENABLE_BLENDING);
    ADD_ENUM_VALUE(ENABLE_BLENDING_WHEN_ALPHA_PRESENT);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER(CompressionMethod, GL_DRIVER);
    ADD_ENUM_VALUE(GL_DRIVER);
    ADD_ENUM_VALUE(NVTT);
    ADD_ENUM_VALUE(NVTT_NOCUDA);
    END_ENUM_SERIALIZER();

    BEGIN_ENUM_SERIALIZER(CompressionQuality, FASTEST);
    ADD_ENUM_VALUE(FASTEST);
    ADD_ENUM_VALUE(NORMAL);
    ADD_ENUM_VALUE(PRODUCTION);
    ADD_ENUM_VALUE(HIGHEST);
    END_ENUM_SERIALIZER();

    ADD_USER_SERIALIZER(DestinationExtents);

    ADD_USER_SERIALIZER(LayerImageOptions);
}
}
