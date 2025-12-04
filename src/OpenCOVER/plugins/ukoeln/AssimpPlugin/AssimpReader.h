/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ASSIMP_READER_H
#define ASSIMP_READER_H

#include <osg/Node>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Material>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>
#include <osgUtil/SmoothingVisitor>
#include <OpenThreads/Mutex>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <unordered_map>
#include <sstream>

#include <config/CoviseConfig.h>

/**
 * Simple convenience construct to make another type "lockable"
 * as long as it has a default constructor
 */
template<typename T>
struct Lockable : public T {
    inline void lock() const {
        _lockable_mutex.lock();
    }
    inline void unlock() const {
        _lockable_mutex.unlock();
    }
    inline OpenThreads::Mutex& mutex() const {
        return _lockable_mutex;
    }
private:
    mutable OpenThreads::Mutex _lockable_mutex;
};

class AssimpReader
{
public:
    typedef Lockable<
        std::unordered_map<std::string, osg::ref_ptr<osg::Texture2D> >
    > TextureCache;

    struct Env
    {
        Env(const std::string& loc, const osgDB::Options* opt)
            : referrer(loc), readOptions(opt) { }
        const std::string referrer;
        const osgDB::Options* readOptions;
    };

public:
    mutable TextureCache* _texCache;

    AssimpReader() : _texCache(NULL)
    {
        //NOP
    }

    void setTextureCache(TextureCache* cache) const
    {
        _texCache = cache;
    }

    osgDB::ReaderWriter::ReadResult read(const std::string& location,
                                         const osgDB::Options* readOptions) const
    {
        OSG_NOTICE << "AssimpReader: Loading file: " << location << std::endl;

        struct FlagEntry {
            const char* configKey;
            bool defaultValue;
            unsigned int flag;
        };

        static const FlagEntry flagTable[] = {
            { "COVER.Plugin.AssimpPlugin.Triangulate",            true, aiProcess_Triangulate },
            { "COVER.Plugin.AssimpPlugin.GenNormals",             true, aiProcess_GenNormals },
            { "COVER.Plugin.AssimpPlugin.GenSmoothNormals",       true, aiProcess_GenSmoothNormals },
            { "COVER.Plugin.AssimpPlugin.JoinIdenticalVertices",  true, aiProcess_JoinIdenticalVertices },
            { "COVER.Plugin.AssimpPlugin.ImproveCacheLocality",   true, aiProcess_ImproveCacheLocality },
            { "COVER.Plugin.AssimpPlugin.RemoveRedundantMaterials", true, aiProcess_RemoveRedundantMaterials },
            { "COVER.Plugin.AssimpPlugin.SortByPType",            true, aiProcess_SortByPType },
            { "COVER.Plugin.AssimpPlugin.FindInvalidData",        true, aiProcess_FindInvalidData },
            { "COVER.Plugin.AssimpPlugin.GenUVCoords",            true, aiProcess_GenUVCoords },
            { "COVER.Plugin.AssimpPlugin.OptimizeMeshes",         true, aiProcess_OptimizeMeshes }
        };

        unsigned int flags = 0;

        // Track the two specific flags
        bool requestedGenNormals = false;
        bool requestedGenSmoothNormals = false;

        for (const auto& entry : flagTable) {
            bool value = coCoviseConfig::isOn(entry.configKey, entry.defaultValue);

            OSG_NOTICE << "AssimpReader: " << entry.configKey
                    << " is " << (value ? "ON" : "OFF") << std::endl;

            if (value) {
                flags |= entry.flag;

                // Detect which one was requested
                if (entry.flag == aiProcess_GenNormals)
                    requestedGenNormals = true;

                if (entry.flag == aiProcess_GenSmoothNormals)
                    requestedGenSmoothNormals = true;
            }
        }

        // Resolve conflict: prefer GenSmoothNormals
        if (requestedGenNormals && requestedGenSmoothNormals) {
            OSG_WARN << "AssimpReader: GenNormals and GenSmoothNormals both enabled — "
                        "using GenSmoothNormals only." << std::endl;

            // Remove the GenNormals bit
            flags &= ~aiProcess_GenNormals;
        }

        OSG_NOTICE << "AssimpReader: Using flags: " << flags << std::endl;
        
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(location, flags);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            OSG_WARN << "Assimp Error loading " << location << ": "
                     << importer.GetErrorString() << std::endl;
            return osgDB::ReaderWriter::ReadResult::ERROR_IN_READING_FILE;
        }

        OSG_NOTICE << "  Scene loaded successfully:" << std::endl;
        OSG_NOTICE << "    Meshes: " << scene->mNumMeshes << std::endl;
        OSG_NOTICE << "    Materials: " << scene->mNumMaterials << std::endl;
        OSG_NOTICE << "    Textures: " << scene->mNumTextures << std::endl;
        OSG_NOTICE << "    Animations: " << scene->mNumAnimations << std::endl;

        Env env(location, readOptions);
        return makeNodeFromScene(scene, env);
    }

    osg::Node* makeNodeFromScene(const aiScene* scene, const Env& env) const
    {
        OSG_NOTICE << "  Converting scene to OSG nodes..." << std::endl;

        // Create root transform with Y-up to Z-up conversion
        osg::MatrixTransform* transform = new osg::MatrixTransform;
        transform->setMatrix(osg::Matrixd::rotate(osg::Vec3d(0.0, 1.0, 0.0), osg::Vec3d(0.0, 0.0, 1.0)));

        // Process root node
        if (scene->mRootNode)
        {
            OSG_NOTICE << "  Processing root node: " << scene->mRootNode->mName.C_Str() << std::endl;
            osg::Node* rootNode = createNode(scene, scene->mRootNode, env);
            if (rootNode)
            {
                transform->addChild(rootNode);
            }
        }

        return transform;
    }

    osg::Node* createNode(const aiScene* scene, const aiNode* node, const Env& env) const
    {
        OSG_NOTICE << "    Creating node: " << node->mName.C_Str()
                 << " (meshes: " << node->mNumMeshes
                 << ", children: " << node->mNumChildren << ")" << std::endl;

        osg::MatrixTransform* mt = new osg::MatrixTransform;
        mt->setName(node->mName.C_Str());

        const aiMatrix4x4& aiMat = node->mTransformation;

        // TRANSPOSED: Convert Assimp matrix to OSG matrix
        osg::Matrixd osgMat(
            aiMat.a1, aiMat.b1, aiMat.c1, aiMat.d1,  // column 0 (was row 0)
            aiMat.a2, aiMat.b2, aiMat.c2, aiMat.d2,  // column 1 (was row 1)
            aiMat.a3, aiMat.b3, aiMat.c3, aiMat.d3,  // column 2 (was row 2)
            aiMat.a4, aiMat.b4, aiMat.c4, aiMat.d4   // column 3 (was row 3)
        );

        mt->setMatrix(osgMat);

        // Process meshes attached to this node
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            osg::Node* geomNode = createMesh(scene, mesh, env);
            if (geomNode)
            {
                mt->addChild(geomNode);
            }
        }

        // Process child nodes recursively
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            osg::Node* child = createNode(scene, node->mChildren[i], env);
            if (child)
            {
                mt->addChild(child);
            }
        }

        return mt;
    }

    // Check if color values are in byte range (0-255) and normalize to float (0-1)
    void normalizeColors(osg::Vec4Array* colors) const
    {
        if (!colors || colors->empty()) return;

        // Sample first few colors to detect range
        float maxVal = 0.0f;
        int sampleSize = std::min(10, (int)colors->size());
        for (int i = 0; i < sampleSize; i++)
        {
            const osg::Vec4& c = (*colors)[i];
            maxVal = std::max(maxVal, std::max(c.r(), std::max(c.g(), c.b())));
        }

        // If any value > 1.0, assume byte format (0-255)
        if (maxVal > 1.0f)
        {
            OSG_NOTICE << "  Normalizing vertex colors from byte range (0-255) to float (0-1)" << std::endl;
            OSG_NOTICE << "  Max value detected: " << maxVal << std::endl;

            for (unsigned int i = 0; i < colors->size(); i++)
            {
                osg::Vec4& c = (*colors)[i];
                c.r() = c.r() / 255.0f;
                c.g() = c.g() / 255.0f;
                c.b() = c.b() / 255.0f;
                c.a() = c.a() > 1.0f ? c.a() / 255.0f : c.a();
            }

            // Log first color after normalization
            if (colors->size() > 0)
            {
                const osg::Vec4& first = (*colors)[0];
                OSG_NOTICE << "  First color after normalization: ("
                          << first.r() << ", " << first.g() << ", "
                          << first.b() << ", " << first.a() << ")" << std::endl;
            }
        }
        else
        {
            OSG_NOTICE << "  Colors already in float range (0-1), max: " << maxVal << std::endl;
        }
    }

    osg::Node* createMesh(const aiScene* scene, const aiMesh* mesh, const Env& env) const
    {
        OSG_NOTICE << "Creating mesh with " << mesh->mNumVertices << " vertices, "
                  << mesh->mNumFaces << " faces" << std::endl;
        OSG_NOTICE << "  Has positions: " << mesh->HasPositions() << std::endl;
        OSG_NOTICE << "  Has normals: " << mesh->HasNormals() << std::endl;
        OSG_NOTICE << "  Has texcoords: " << mesh->HasTextureCoords(0) << std::endl;
        OSG_NOTICE << "  Has vertex colors: " << mesh->HasVertexColors(0) << std::endl;

        osg::Geode* geode = new osg::Geode();
        osg::Geometry* geometry = new osg::Geometry();
        geometry->setUseVertexBufferObjects(true);

        // Vertices
        if (mesh->HasPositions())
        {
            osg::Vec3Array* vertices = new osg::Vec3Array();
            vertices->reserve(mesh->mNumVertices);
            for (unsigned int i = 0; i < mesh->mNumVertices; i++)
            {
                const aiVector3D& v = mesh->mVertices[i];
                vertices->push_back(osg::Vec3(v.x, v.y, v.z));
            }
            geometry->setVertexArray(vertices);
        }

        // Normals
        if (mesh->HasNormals())
        {
            osg::Vec3Array* normals = new osg::Vec3Array();
            normals->reserve(mesh->mNumVertices);
            for (unsigned int i = 0; i < mesh->mNumVertices; i++)
            {
                const aiVector3D& n = mesh->mNormals[i];
                normals->push_back(osg::Vec3(n.x, n.y, n.z));
            }
            geometry->setNormalArray(normals);
            geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }

        // Texture coordinates (UV)
        if (mesh->HasTextureCoords(0))
        {
            osg::Vec2Array* texcoords = new osg::Vec2Array();
            texcoords->reserve(mesh->mNumVertices);
            for (unsigned int i = 0; i < mesh->mNumVertices; i++)
            {
                const aiVector3D& tc = mesh->mTextureCoords[0][i];
                texcoords->push_back(osg::Vec2(tc.x, tc.y));
            }
            geometry->setTexCoordArray(0, texcoords);
        }

        // Vertex colors
        if (mesh->HasVertexColors(0))
        {
            OSG_NOTICE << "  Loading vertex colors: " << mesh->mNumVertices << " vertices" << std::endl;

            osg::Vec4Array* colors = new osg::Vec4Array();
            colors->reserve(mesh->mNumVertices);
            for (unsigned int i = 0; i < mesh->mNumVertices; i++)
            {
                const aiColor4D& c = mesh->mColors[0][i];
                colors->push_back(osg::Vec4(c.r, c.g, c.b, c.a));
            }

            // Log first few colors before normalization
            if (colors->size() > 0)
            {
                const osg::Vec4& first = (*colors)[0];
                OSG_NOTICE << "  First color (raw): ("
                          << first.r() << ", " << first.g() << ", "
                          << first.b() << ", " << first.a() << ")" << std::endl;
            }

            // Normalize if needed (0-255 -> 0-1)
            normalizeColors(colors);

            geometry->setColorArray(colors);
            geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

            OSG_NOTICE << "  Vertex colors set with BIND_PER_VERTEX" << std::endl;
        }

        // Indices - convert faces to triangles
        osg::DrawElementsUInt* indices = new osg::DrawElementsUInt(GL_TRIANGLES);
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            const aiFace& face = mesh->mFaces[i];
            // Should be triangles after aiProcess_Triangulate
            if (face.mNumIndices == 3)
            {
                indices->push_back(face.mIndices[0]);
                indices->push_back(face.mIndices[1]);
                indices->push_back(face.mIndices[2]);
            }
        }
        geometry->addPrimitiveSet(indices);

        // Generate normals if missing
        if (!mesh->HasNormals())
        {
            osgUtil::SmoothingVisitor sv;
            geometry->accept(sv);
        }

        // Apply material
        if (mesh->mMaterialIndex < scene->mNumMaterials)
        {
            const aiMaterial* aiMat = scene->mMaterials[mesh->mMaterialIndex];
            applyMaterial(scene, aiMat, geometry, env);
        }

        geode->addDrawable(geometry);
        return geode;
    }

    void applyMaterial(const aiScene* scene, const aiMaterial* aiMat, osg::Geometry* geometry, const Env& env) const
    {
        osg::ref_ptr<osg::Material> osgMat = new osg::Material();

        // Diffuse color
        aiColor4D diffuse(0.8f, 0.8f, 0.8f, 1.0f);
        if (AI_SUCCESS == aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
        {
            osgMat->setDiffuse(osg::Material::FRONT_AND_BACK,
                              osg::Vec4(diffuse.r, diffuse.g, diffuse.b, diffuse.a));
        }

        // Ambient color
        aiColor4D ambient(0.2f, 0.2f, 0.2f, 1.0f);
        if (AI_SUCCESS == aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_AMBIENT, &ambient))
        {
            osgMat->setAmbient(osg::Material::FRONT_AND_BACK,
                              osg::Vec4(ambient.r, ambient.g, ambient.b, ambient.a));
        }

        // Specular color
        aiColor4D specular(0.0f, 0.0f, 0.0f, 1.0f);
        if (AI_SUCCESS == aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_SPECULAR, &specular))
        {
            osgMat->setSpecular(osg::Material::FRONT_AND_BACK,
                               osg::Vec4(specular.r, specular.g, specular.b, specular.a));
        }

        // Shininess
        float shininess = 0.0f;
        if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_SHININESS, &shininess))
        {
            osgMat->setShininess(osg::Material::FRONT_AND_BACK, shininess);
        }

        // Opacity
        float opacity = 1.0f;
        if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_OPACITY, &opacity))
        {
            osg::Vec4 diff = osgMat->getDiffuse(osg::Material::FRONT_AND_BACK);
            diff.a() = opacity;
            osgMat->setDiffuse(osg::Material::FRONT_AND_BACK, diff);

            if (opacity < 1.0f)
            {
                geometry->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
                geometry->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            }
        }

        // Vertex colors
        if (geometry->getColorArray())
        {
            OSG_NOTICE << "  Mesh has vertex colors, enabling color material mode" << std::endl;
            osgMat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        }

        geometry->getOrCreateStateSet()->setAttributeAndModes(osgMat.get(), osg::StateAttribute::ON);

        // Diffuse texture
        if (aiMat->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            aiString texPath;
            if (AI_SUCCESS == aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath))
            {
                // Load texture (handles both embedded "*N" and external paths)
                osg::ref_ptr<osg::Texture2D> tex = loadTexture(scene, texPath.C_Str(), env);
                if (tex.valid())
                {
                    geometry->getOrCreateStateSet()->setTextureAttributeAndModes(
                        0, tex.get(), osg::StateAttribute::ON);
                }
            }
        }
    }

    std::string resolveTexturePath(const std::string& texPath, const std::string& modelPath) const
    {
        if (osgDB::isAbsolutePath(texPath))
            return texPath;

        // Try relative to model file
        std::string fullPath = osgDB::concatPaths(osgDB::getFilePath(modelPath), texPath);
        if (osgDB::fileExists(fullPath))
            return fullPath;

        // Try just the filename in model directory
        std::string filename = osgDB::getSimpleFileName(texPath);
        fullPath = osgDB::concatPaths(osgDB::getFilePath(modelPath), filename);
        if (osgDB::fileExists(fullPath))
            return fullPath;

        return texPath; // Return original, let OSG try to find it
    }

    osg::Texture2D* loadTexture(const aiScene* scene, const std::string& path, const Env& env) const
    {
        osg::ref_ptr<osg::Texture2D> tex;
        osg::ref_ptr<osg::Texture2D>* cachedTex = NULL;

        // Check cache
        if (_texCache)
        {
            _texCache->lock();
            cachedTex = &(*_texCache)[path];
            tex = cachedTex->get();
        }

        if (!tex.valid())
        {
            OSG_NOTICE << "  Loading texture: " << path << std::endl;

            osg::ref_ptr<osg::Image> img;

            // Check if this is an embedded texture (format: "*0", "*1", etc.)
            if (path.length() > 0 && path[0] == '*')
            {
                // Parse embedded texture index
                int texIndex = std::atoi(path.c_str() + 1);

                if (texIndex >= 0 && texIndex < (int)scene->mNumTextures)
                {
                    OSG_NOTICE << "    Loading embedded texture at index " << texIndex << std::endl;

                    const aiTexture* aiTex = scene->mTextures[texIndex];

                    if (aiTex->mHeight == 0)
                    {
                        // Compressed texture (PNG, JPG, etc.)
                        OSG_NOTICE << "    Compressed format hint: " << aiTex->achFormatHint << std::endl;
                        OSG_NOTICE << "    Data size: " << aiTex->mWidth << " bytes" << std::endl;

                        // Get the appropriate ReaderWriter for this format
                        std::string format = aiTex->achFormatHint;
                        osgDB::ReaderWriter* rw = osgDB::Registry::instance()->getReaderWriterForExtension(format);

                        if (rw)
                        {
                            // Create stream with compressed data
                            std::stringstream ss(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
                            ss.write(reinterpret_cast<const char*>(aiTex->pcData), aiTex->mWidth);
                            ss.seekg(0);

                            // Read image from stream
                            osgDB::ReaderWriter::ReadResult rr = rw->readImage(ss);
                            if (rr.success())
                            {
                                img = rr.takeImage();
                                OSG_NOTICE << "    Successfully decoded embedded texture: "
                                          << img->s() << "x" << img->t() << std::endl;
                            }
                            else
                            {
                                OSG_WARN << "    Failed to decode embedded texture" << std::endl;
                            }
                        }
                        else
                        {
                            OSG_WARN << "    No ReaderWriter found for format: " << format << std::endl;
                        }
                    }
                    else
                    {
                        // Uncompressed RGBA8888 texture
                        OSG_NOTICE << "    Uncompressed RGBA texture: "
                                  << aiTex->mWidth << "x" << aiTex->mHeight << std::endl;

                        img = new osg::Image();

                        // Allocate and copy pixel data
                        unsigned char* data = new unsigned char[aiTex->mWidth * aiTex->mHeight * 4];
                        for (unsigned int i = 0; i < aiTex->mWidth * aiTex->mHeight; i++)
                        {
                            data[i*4 + 0] = aiTex->pcData[i].r;
                            data[i*4 + 1] = aiTex->pcData[i].g;
                            data[i*4 + 2] = aiTex->pcData[i].b;
                            data[i*4 + 3] = aiTex->pcData[i].a;
                        }

                        img->setImage(aiTex->mWidth, aiTex->mHeight, 1,
                                     GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                                     data, osg::Image::USE_NEW_DELETE);
                    }
                }
                else
                {
                    OSG_WARN << "    Invalid embedded texture index: " << texIndex << std::endl;
                }
            }
            else
            {
                // External texture file - resolve path and load
                std::string fullPath = resolveTexturePath(path, env.referrer);
                img = osgDB::readImageFile(fullPath, env.readOptions);
            }

            // Create texture from image
            if (img.valid())
            {
                tex = new osg::Texture2D(img.get());
                tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
                tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
                tex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
                tex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
                tex->setUnRefImageDataAfterApply(true);
                tex->setResizeNonPowerOfTwoHint(false);
                tex->setDataVariance(osg::Object::STATIC);

                // Store in cache
                if (cachedTex && !cachedTex->valid())
                {
                    (*cachedTex) = tex.get();
                }
            }
            else
            {
                OSG_WARN << "  Failed to load texture: " << path << std::endl;
            }
        }

        if (_texCache)
        {
            _texCache->unlock();
        }

        return tex.release();
    }

private:
};

#endif // ASSIMP_READER_H
