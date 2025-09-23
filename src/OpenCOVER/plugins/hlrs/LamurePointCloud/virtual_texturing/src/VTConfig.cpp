// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/QuadTree.h>
#include <lamure/vt/VTConfig.h>

namespace vt
{
std::string vt::VTConfig::CONFIG_PATH;

void VTConfig::read_config(const char *path_config)
{
    CSimpleIniA *ini_config = new CSimpleIniA(true, false, false);

    if(ini_config->LoadFile(path_config) < 0)
    {
        throw std::runtime_error("Configuration file parsing error");
    }

    _size_tile = (uint16_t)atoi(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::TILE_SIZE, VTConfig::UNDEF));
    _size_padding = (uint16_t)atoi(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::TILE_PADDING, VTConfig::UNDEF));
    _size_physical_texture = (uint32_t)atoi(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::PHYSICAL_SIZE_MB, VTConfig::UNDEF));
    _size_physical_update_throughput = (uint32_t)atoi(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::PHYSICAL_UPDATE_THROUGHPUT_MB, VTConfig::UNDEF));
    _size_ram_cache = (uint32_t)atoi(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::RAM_CACHE_SIZE_MB, VTConfig::UNDEF));
    _format_texture = VTConfig::which_texture_format(ini_config->GetValue(VTConfig::TEXTURE_MANAGEMENT, VTConfig::TEXTURE_FORMAT, VTConfig::UNDEF));
    _verbose = atoi(ini_config->GetValue(VTConfig::DEBUG, VTConfig::VERBOSE, VTConfig::UNDEF)) == 1;
}
void VTConfig::define_size_physical_texture(uint32_t max_tex_layers, uint32_t max_tex_px_width_gl)
{
    size_t max_tex_byte_size = (size_t)_size_physical_texture * 1024 * 1024;

    size_t max_tex_px_width_custom = (size_t)std::pow(2, (size_t)std::log2(sqrt(max_tex_byte_size / get_byte_stride())));
    size_t max_tex_px_width = ((size_t)max_tex_px_width_gl < max_tex_px_width_custom ? (size_t)max_tex_px_width_gl : (size_t)max_tex_px_width_custom);

    size_t tex_tile_width = max_tex_px_width / _size_tile;
    size_t tex_px_width = tex_tile_width * _size_tile;
    size_t tex_byte_size = tex_px_width * tex_px_width * get_byte_stride();
    size_t layers = max_tex_byte_size / tex_byte_size;

    _phys_tex_px_width = (uint32_t)tex_px_width;
    _phys_tex_tile_width = (uint32_t)tex_tile_width;
    _phys_tex_layers = (uint16_t)layers < (uint16_t)max_tex_layers ? (uint16_t)layers : (uint16_t)max_tex_layers;
}
uint16_t VTConfig::get_size_tile() const { return _size_tile; }
uint16_t VTConfig::get_size_padding() const { return _size_padding; }
VTConfig::FORMAT_TEXTURE VTConfig::get_format_texture() const { return _format_texture; }
bool VTConfig::is_verbose() const { return _verbose; }
uint32_t VTConfig::get_size_physical_update_throughput() const { return _size_physical_update_throughput; }
uint32_t VTConfig::get_phys_tex_px_width() const { return _phys_tex_px_width; }
uint32_t VTConfig::get_phys_tex_tile_width() const { return _phys_tex_tile_width; }
uint16_t VTConfig::get_phys_tex_layers() const { return _phys_tex_layers; }
uint16_t VTConfig::get_byte_stride() const
{
    uint8_t _byte_stride = 0;
    switch(_format_texture)
    {
    case VTConfig::FORMAT_TEXTURE::RGBA8:
        _byte_stride = 4;
        break;
    case VTConfig::FORMAT_TEXTURE::RGB8:
        _byte_stride = 3;
        break;
    case VTConfig::FORMAT_TEXTURE::R8:
        _byte_stride = 1;
        break;
    }

    return _byte_stride;
}
const VTConfig::FORMAT_TEXTURE VTConfig::which_texture_format(const char *_texture_format)
{
    if(strcmp(_texture_format, TEXTURE_FORMAT_RGBA8) == 0)
    {
        return RGBA8;
    }
    else if(strcmp(_texture_format, TEXTURE_FORMAT_RGB8) == 0)
    {
        return RGB8;
    }
    else if(strcmp(_texture_format, TEXTURE_FORMAT_R8) == 0)
    {
        return R8;
    }
    throw std::runtime_error("Unknown texture format");
}

    uint32_t VTConfig::get_size_ram_cache() const {
        return _size_ram_cache;
    }
}