// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_CAMERA_H
#define LAMURE_CAMERA_H

#include <lamure/prov/meta_data.h>
#include <lamure/prov/common.h>

#include <memory>

#include <fstream>
#include <map>
#include <vector>

namespace lamure {
namespace prov
{
class Camera
{
  public:
    uint16_t MAX_LENGTH_FILE_PATH;

    Camera() {}
    Camera(uint16_t _index, const string &_im_file_name, const quatf &_orientation, const vec3f &_translation, const vec<uint8_t> &_metadata)
        : _index(_index), _im_file_name(_im_file_name), _orientation(_orientation), _translation(_translation)
    {
        prepare("");
    }
    ~Camera() {}
    void prepare(const std::string& fotos_directory)
    {
        try
        {
            read_image(fotos_directory);
        }
        catch(const std::runtime_error e)
        {
            printf("\nFailed to read image: %s", e.what());
        }
    }

    int get_index() { return _index; }
    quatf &get_orientation() { return _orientation; }
    string &get_file_name() { return _im_file_name; }
    vec3f &get_translation() { return _translation; }
    float get_focal_length() { return _focal_length; }
    const std::string& get_image_file() const { return _im_file_name; }
    uint32_t get_image_width() { return _im_width; }
    uint32_t get_image_height() { return _im_height; }
    
    friend ifstream &operator>>(ifstream &is, Camera &camera)
    {
        is.read(reinterpret_cast<char *>(&camera._index), 2);
        camera._index = swap(camera._index, true);

        // if(DEBUG)
        // printf("\nIndex: %i", camera._index);

        is.read(reinterpret_cast<char *>(&camera._focal_length), 4);
        camera._focal_length = swap(camera._focal_length, true);

        // if(DEBUG)
        // printf("\nFocal length: %f", camera._focal_length);

        float w, x, y, z;
        is.read(reinterpret_cast<char *>(&w), 4);
        is.read(reinterpret_cast<char *>(&x), 4);
        is.read(reinterpret_cast<char *>(&y), 4);
        is.read(reinterpret_cast<char *>(&z), 4);
        w = swap(w, true);
        x = swap(x, true);
        y = swap(y, true);
        z = swap(z, true);

        // if(DEBUG)
        // printf("\nWXYZ: %f %f %f %f", w, x, y, z);

        quatf quat_tmp = quatf(w, x, y, z);
        scm::math::quat<float> new_orientation = scm::math::quat<float>::from_axis(180, scm::math::vec3f(1.0, 0.0, 0.0));
        quat_tmp = scm::math::normalize(quat_tmp);
        camera._orientation = scm::math::quat<float>::from_matrix(camera.set_quaternion_rotation(quat_tmp)) * new_orientation;
        // camera._orientation = quatd(w, x, y, z);

        is.read(reinterpret_cast<char *>(&x), 4);
        is.read(reinterpret_cast<char *>(&y), 4);
        is.read(reinterpret_cast<char *>(&z), 4);
        x = swap(x, true);
        y = swap(y, true);
        z = swap(z, true);

        // if(DEBUG)
        // printf("\nXYZ: %f %f %f", x, y, z);

        camera._translation = vec3f(x, y, z);

        char *byte_buffer = new char[camera.MAX_LENGTH_FILE_PATH];
        is.read(byte_buffer, camera.MAX_LENGTH_FILE_PATH);
        camera._im_file_name = string(byte_buffer);
        camera._im_file_name = trim(camera._im_file_name);
        delete[] byte_buffer;

        // if(DEBUG)
        // printf("\nFile path: \'%s\'\n", camera._im_file_name.c_str());

        //        camera.read_metadata(is);

        return is;
    }

  private:
    scm::math::mat3f set_quaternion_rotation(const scm::math::quat<float> q)
    {
        scm::math::mat3f m = scm::math::mat3f::identity();
        float qw = q.w;
        float qx = q.i;
        float qy = q.j;
        float qz = q.k;
        m[0] = (qw * qw + qx * qx - qz * qz - qy * qy);
        m[1] = (2 * qx * qy - 2 * qz * qw);
        m[2] = (2 * qy * qw + 2 * qz * qx);
        m[3] = (2 * qx * qy + 2 * qw * qz);
        m[4] = (qy * qy + qw * qw - qz * qz - qx * qx);
        m[5] = (2 * qz * qy - 2 * qx * qw);
        m[6] = (2 * qx * qz - 2 * qy * qw);
        m[7] = (2 * qy * qz + 2 * qw * qx);
        m[8] = (qz * qz + qw * qw - qy * qy - qx * qx);
        return m;
    }

    void read_image(const std::string& fotos_directory)
    {
        std::cout << "Reading image " << fotos_directory+_im_file_name << std::endl;

        FILE *fp = fopen((fotos_directory+_im_file_name).c_str(), "rb");
        if(!fp)
        {
            std::stringstream sstr;
            sstr << "Can't open file: \'" << _im_file_name << '\'';
            throw std::runtime_error(sstr.str());
        }
        fseek(fp, 0, SEEK_END);
        size_t fsize = (size_t)ftell(fp);
        rewind(fp);
        unsigned char *buf = new unsigned char[fsize];
        if(fread(buf, 1, fsize, fp) != fsize)
        {
            delete[] buf;
            std::stringstream sstr;
            sstr << "Can't read file: \'" << _im_file_name << '\'';
            throw std::runtime_error(sstr.str());
        }
        fclose(fp);

        easyexif::EXIFInfo result;
        result.parseFrom(buf, (unsigned int)fsize);

        _im_height = result.ImageHeight;
        _im_width = result.ImageWidth;
        _focal_length = result.FocalLength * 0.001;
        _fp_resolution_x = result.LensInfo.FocalPlaneResolutionUnit == 2 ? result.LensInfo.FocalPlaneXResolution / 0.0254 : result.LensInfo.FocalPlaneXResolution / 0.01;
        _fp_resolution_y = result.LensInfo.FocalPlaneResolutionUnit == 2 ? result.LensInfo.FocalPlaneYResolution / 0.0254 : result.LensInfo.FocalPlaneYResolution / 0.01;

        // std::cout << result.FocalLength << std::endl;
        // std::cout << result.LensInfo.FocalPlaneXResolution << std::endl;
        if(_fp_resolution_x == 0 && _fp_resolution_y == 0)
        {
            // _fp_resolution_x = 5715.545755 / 0.0254;
            // _fp_resolution_x = (2976 / 0.384615385) / 0.0254;
            // _fp_resolution_y = (2976 / 0.384615385) / 0.0254;
            // how many pixels are inside 1m = 1m / (1.12nm / 1'000'000)
            _fp_resolution_x = 1.0f / (1.12 / 1000000);
            _fp_resolution_y = 1.0f / (1.12 / 1000000);
        }

        // if(DEBUG)
        // printf("Focal length: %f, FP Resolution X: %f, Y: %f\n", _focal_length, _fp_resolution_x, _fp_resolution_y);
    }

  protected:
    uint16_t _index;
    float _focal_length;
    string _im_file_name;
    quatf _orientation;
    vec3f _translation;

    int _im_height;
    int _im_width;
    float _fp_resolution_x;
    float _fp_resolution_y;
};
}
}

#endif // LAMURE_CAMERA_H
