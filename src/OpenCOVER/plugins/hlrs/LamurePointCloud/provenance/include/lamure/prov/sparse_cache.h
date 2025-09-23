// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_SPARSE_DATA_H
#define LAMURE_SPARSE_DATA_H

#include <lamure/prov/common.h>
#include <lamure/prov/sparse_point.h>
#include <lamure/prov/cacheable.h>

namespace lamure {
namespace prov
{
class SparseCache : public Cacheable<SparsePoint, MetaData>
{
  public:
    SparseCache(ifstream &is_prov, ifstream &is_meta) : Cacheable(is_prov, is_meta)
    {
        _cameras = vec<Camera>();
        _cameras_metadata = vec<MetaData>();
    };

    void cache(bool load_images, const std::string& fotos_directory) {
        Cacheable::cache();
        cache_cameras(load_images, fotos_directory);
    }

    void cache() override {
        Cacheable::cache();
        cache_cameras(true, "");
    }


    const vec<prov::Camera> get_cameras() const { return _cameras; }
    const vec<prov::MetaData> get_cameras_metadata() const { return _cameras_metadata; }

  protected:
    void cache_cameras(bool load_images, const std::string& fotos_directory)
    {
        uint16_t cameras_length;
        (*is_prov).read(reinterpret_cast<char *>(&cameras_length), 2);
        cameras_length = swap(cameras_length, true);

        // if(DEBUG)
        // printf("\nCameras length: %i", cameras_length);

        uint32_t meta_data_length;
        (*is_prov).read(reinterpret_cast<char *>(&meta_data_length), 4);
        meta_data_length = swap(meta_data_length, true);

        // if(DEBUG)
        // printf("\nCamera meta data length: %i ", meta_data_length);

        uint16_t max_len_fpath;
        (*is_prov).read(reinterpret_cast<char *>(&max_len_fpath), 2);
        max_len_fpath = swap(max_len_fpath, true);

        // if(DEBUG)
        // printf("\nMax file path length: %i ", max_len_fpath);

        for(uint16_t i = 0; i < cameras_length; i++)
        {
            Camera camera = Camera();
            camera.MAX_LENGTH_FILE_PATH = max_len_fpath;
            (*is_prov) >> camera;
            if (load_images) {
              camera.prepare(fotos_directory);
            }
            _cameras.push_back(camera);

            MetaData meta_container;
            meta_container.read_metadata((*is_meta), meta_data_length);
            _cameras_metadata.push_back(meta_container);
        }
    }

    vec<Camera> _cameras;
    vec<MetaData> _cameras_metadata;
};
}
}

#endif // LAMURE_SPARSE_DATA_H