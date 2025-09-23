// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_VTCONFIG_H
#define LAMURE_VTCONFIG_H

#include <lamure/vt/common.h>

#include <fstream>
#include <iostream>

#include <lamure/vt/ext/SimpleIni.h>

namespace vt
{
class VTConfig
{
  public:
    enum FORMAT_TEXTURE
    {
        RGBA8,
        RGB8,
        R8
    };

    static std::string CONFIG_PATH;

    static VTConfig &get_instance()
    {
        static VTConfig instance(CONFIG_PATH.c_str());
        return instance;
    }

    VTConfig(VTConfig const &) = delete;
    void operator=(VTConfig const &) = delete;

    static const FORMAT_TEXTURE which_texture_format(const char *_texture_format);

    uint16_t get_byte_stride() const;

    /** Use define_size_physical_texture(max_tex_layers, max_tex_px_width_gl) with the outputs of the following GL calls:
     *
     * GLint max_tex_layers;
     * glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &max_tex_layers);
     *
     * GLint max_tex_px_width_gl;
     * glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex_px_width_gl);
     *
     * */
    void define_size_physical_texture(uint32_t max_tex_layers, uint32_t max_tex_px_width_gl);

    uint16_t get_size_tile() const;
    uint16_t get_size_padding() const;
    uint32_t get_size_physical_update_throughput() const;

    uint32_t get_size_ram_cache() const;

    FORMAT_TEXTURE get_format_texture() const;
    bool is_verbose() const;

    uint32_t get_phys_tex_px_width() const;
    uint32_t get_phys_tex_tile_width() const;
    uint16_t get_phys_tex_layers() const;

  private:
    VTConfig(const char *path_config) { read_config(path_config); }

    // Sections
    static constexpr const char *TEXTURE_MANAGEMENT = "TEXTURE_MANAGEMENT";
    static constexpr const char *DEBUG = "DEBUG";

    // Texture management fields
    static constexpr const char *TILE_SIZE = "TILE_SIZE";
    static constexpr const char *TILE_PADDING = "TILE_PADDING";
    static constexpr const char *PHYSICAL_SIZE_MB = "PHYSICAL_SIZE_MB";
    static constexpr const char *PHYSICAL_UPDATE_THROUGHPUT_MB = "PHYSICAL_UPDATE_THROUGHPUT_MB";
    static constexpr const char *RAM_CACHE_SIZE_MB = "RAM_CACHE_SIZE_MB";

    static constexpr const char *TEXTURE_FORMAT = "TEXTURE_FORMAT";
    static constexpr const char *TEXTURE_FORMAT_RGBA8 = "RGBA8";
    static constexpr const char *TEXTURE_FORMAT_RGB8 = "RGB8";
    static constexpr const char *TEXTURE_FORMAT_R8 = "R8";

    // Debug fields
    static constexpr const char *VERBOSE = "VERBOSE";

    static constexpr const char *UNDEF = "UNDEF";

  private:
    void read_config(const char *path_config);

    uint16_t _size_tile;
    uint16_t _size_padding;
    uint32_t _size_physical_texture;
    uint32_t _size_physical_update_throughput;
    uint32_t _size_ram_cache;

    VTConfig::FORMAT_TEXTURE _format_texture;
    bool _verbose;

    uint32_t _phys_tex_px_width;
    uint32_t _phys_tex_tile_width;
    uint16_t _phys_tex_layers;
};
}

#endif // LAMURE_VTCONFIG_H
