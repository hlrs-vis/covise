#pragma once

#include <array>
#include <memory>
#include <anari/anari_cpp.hpp>
#include <asg/asg.h>

namespace asg {
    // Error code thrown or raised (not returned!) by asg functions
    using Error             = ASGError_t;
    using MatrixFormat      = ASGMatrixFormat_t;
    using FreeFunc          = ASGFreeFunc;

    class Param
    {
        ASGParam parm;
    public:
        Param(const char* name, int v1);
        Param(const char* name, int v1, int v2);
        Param(const char* name, int v1, int v2, int v3);
        Param(const char* name, int v1, int v2, int v3, int v4);
        Param(const char* name, std::array<int,2> v);
        Param(const char* name, std::array<int,3> v);
        Param(const char* name, std::array<int,4> v);

        Param(const char* name, float v1);
        Param(const char* name, float v1, float v2);
        Param(const char* name, float v1, float v2, float v3);
        Param(const char* name, float v1, float v2, float v3, float v4);
        Param(const char* name, std::array<float,2> v);
        Param(const char* name, std::array<float,3> v);
        Param(const char* name, std::array<float,4> v);

        operator ASGParam() const;
    };

    namespace detail {

        class Object
        {
        protected:
            ASGObject handle;
            Object(ASGObject obj);

        public:
            static std::shared_ptr<Object> create();

            operator ASGObject();

            void release();

            void retain();

            template <typename T>
            void addChild(const std::shared_ptr<T>& child);

            std::shared_ptr<Object> getChild(int childID);
        };

        class Geometry : public Object
        {
        protected:
            Geometry(ASGGeometry obj);

        public:
            operator ASGGeometry();
        };

        class TriangleGeometry : public Geometry
        {
            TriangleGeometry(ASGGeometry obj);
        public:
            static std::shared_ptr<TriangleGeometry> create(float* vertices,
                                                            float* vertexNormals,
                                                            float* vertexColors,
                                                            uint32_t numVertices,
                                                            uint32_t* indices,
                                                            uint32_t numIndices,
                                                            FreeFunc freeVertices,
                                                            FreeFunc freeNormals,
                                                            FreeFunc freeColors,
                                                            FreeFunc freeIndices);

            operator ASGTriangleGeometry();
        };

        class SphereGeometry;

        class CylinderGeometry : public Geometry
        {
            CylinderGeometry(ASGGeometry obj);
        public:
            static std::shared_ptr<CylinderGeometry> create(float* vertices, float* radii,
                                                            float* vertexColors,
                                                            uint8_t* caps,
                                                            uint32_t numVertices,
                                                            uint32_t* indices,
                                                            uint32_t numIndices,
                                                            float defaultRadius,
                                                            FreeFunc freeVertices,
                                                            FreeFunc freeRadii,
                                                            FreeFunc freeColors,
                                                            FreeFunc freeCaps,
                                                            FreeFunc freeIndices);

            operator ASGCylinderGeometry();
        };

        class Material : public Object
        {
            Material(ASGMaterial obj);
        public:
            static std::shared_ptr<Material> create(const char* materialType);

            operator ASGMaterial();

            void setParam(Param parm);
        };

        class Light;

        class Surface : public Object
        {
            Surface(ASGSurface obj);
        public:
            static std::shared_ptr<Surface> create(ASGGeometry geom, ASGMaterial mat);
        };

        class Sampler2D;
        class LookupTable1D;
        class StructuredVolume;

        class Transform : public Object
        {
            Transform(ASGTransform obj);
        public:
            static std::shared_ptr<Transform> create(float initialMatrix[12],
                    MatrixFormat format = ASG_MATRIX_FORMAT_COL_MAJOR);

            void translate(float x, float y, float z);

            void translate(float xyz[3]);

        };

        class Select;
        class Camera;
    } // detail

    using Object            = std::shared_ptr<detail::Object>;
    using Geometry          = std::shared_ptr<detail::Geometry>;
    using TriangleGeometry  = std::shared_ptr<detail::TriangleGeometry>;
    using SphereGeometry    = std::shared_ptr<detail::SphereGeometry>;
    using CylinderGeometry  = std::shared_ptr<detail::CylinderGeometry>;
    using Material          = std::shared_ptr<detail::Material>;
    using Light             = std::shared_ptr<detail::Light>;
    using Surface           = std::shared_ptr<detail::Surface>;
    using Sampler2D         = std::shared_ptr<detail::Sampler2D>;
    using LookupTable1D     = std::shared_ptr<detail::LookupTable1D>;
    using StructuredVolume  = std::shared_ptr<detail::StructuredVolume>;
    using Transform         = std::shared_ptr<detail::Transform>;
    using Select            = std::shared_ptr<detail::Select>;
    using Camera            = std::shared_ptr<detail::Camera>;

    inline Object newObject()
    {
        return detail::Object::create();
    }

    inline TriangleGeometry newTriangleGeometry(float* vertices,
                                                float* vertexNormals,
                                                float* vertexColors,
                                                uint32_t numVertices,
                                                uint32_t* indices,
                                                uint32_t numIndices,
                                                FreeFunc freeVertices = nullptr,
                                                FreeFunc freeNormals = nullptr,
                                                FreeFunc freeColors = nullptr,
                                                FreeFunc freeIndices = nullptr)
    {
        return detail::TriangleGeometry::create(vertices,vertexNormals,vertexColors,
                                                numVertices,indices,numIndices,
                                                freeVertices,freeNormals,freeColors,
                                                freeIndices);
    }

    inline CylinderGeometry newCylinderGeometry(float* vertices, float* radii,
                                                float* vertexColors, uint8_t* caps,
                                                uint32_t numVertices, uint32_t* indices,
                                                uint32_t numIndices,
                                                float defaultRadius = 1.f,
                                                FreeFunc freeVertices = nullptr,
                                                FreeFunc freeRadii = nullptr,
                                                FreeFunc freeColors = nullptr,
                                                FreeFunc freeCaps = nullptr,
                                                FreeFunc freeIndices = nullptr)
    {
        return detail::CylinderGeometry::create(vertices,radii,vertexColors,caps,
                                                numVertices,indices,numIndices,
                                                defaultRadius,freeVertices,freeRadii,
                                                freeColors,freeCaps,freeIndices);
    }

    inline Material newMaterial(const char* materialType)
    {
        return detail::Material::create(materialType);
    }

    inline Surface newSurface(Geometry geom, Material mat = nullptr)
    {
        return detail::Surface::create((ASGGeometry)*geom,mat?(ASGMaterial)*mat:nullptr);
    }

    inline Surface newSurface(TriangleGeometry geom, Material mat)
    {
        return detail::Surface::create((ASGGeometry)*geom,(ASGMaterial)*mat);
    }

    inline Transform newTransform(float initialMatrix[12] = nullptr,
                                  MatrixFormat format = ASG_MATRIX_FORMAT_COL_MAJOR)
    {
        return detail::Transform::create(initialMatrix,format);
    }

} // ::asg

#include "detail/asg.inl"

