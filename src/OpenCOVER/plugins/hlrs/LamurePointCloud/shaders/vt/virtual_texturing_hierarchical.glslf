// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core

uniform uint max_level;
uniform uvec2 physical_texture_dim;

uniform vec2 tile_size;
uniform vec2 tile_padding;

uniform bool enable_hierarchy;
uniform int toggle_visualization;

layout(binding = 0) uniform usampler2D hierarchical_idx_textures[16];
layout(binding = 17) uniform sampler2DArray physical_texture_array;

layout(std430, binding = 0) buffer out_lod_feedback { int out_lod_feedback_values[]; };
layout(std430, binding = 1) buffer out_count_feedback { uint out_count_feedback_values[]; };

layout(location = 0) out vec4 out_color;

in vec2 texture_coord;

/*
in vertex_data {
    vec3 position;
    vec3 normal;
    vec2 coord;
} vertex_in;
*/

struct idx_tex_positions
{
    int parent_lvl;
    uvec4 parent_idx;
    int child_lvl;
    uvec4 child_idx;
};

/*
* Estimation of the requiered Level-of-Detail.
*/
float dxdy()
{
    vec2 c = texture_coord * tile_size;

    float dFdxCx = dFdx(c.x);
    float dFdxCy = dFdx(c.y);

    float dFdyCx = dFdy(c.x);
    float dFdyCy = dFdy(c.y);

    float rho = max( sqrt( dFdxCx*dFdxCx + dFdxCy*dFdxCy ),
                     sqrt( dFdyCx*dFdyCx + dFdyCy*dFdyCy ));

    float lambda = log2(rho);

    return lambda;
}

/*
* Physical texture lookup
*/
vec4 get_physical_texture_color(uvec4 index_quadruple, vec2 texture_sampling_coordinates, uint current_level)
{
    // exponent for calculating the occupied pixels in our index texture, based on which level the tile is in
    uint tile_occupation_exponent = max_level - current_level;

    // 2^tile_occupation_exponent defines how many pixel (of the index texture) are used by the given tile
    uint occupied_index_pixel_per_dimension = uint(1 << tile_occupation_exponent);

    // offset represented as tiles is divided by total num tiles per axis
    // (replace max_width_tiles later by correct uniform)
    // extracting x,y from index texture
    uvec2 base_xy_offset = index_quadruple.xy;

    // base x,y coordinates * number of tiles / number of used index texture pixel
    // taking the factional part by modf
    vec2 physical_tile_ratio_xy = fract(texture_sampling_coordinates * (1 << max_level) / vec2(occupied_index_pixel_per_dimension));

    // Use only tile_size - 2*tile_padding pixels to render scene
    // Therefore, scale reduced tile size to full size and translate it
    vec2 padding_scale = 1 - 2 * tile_padding / tile_size;
    vec2 padding_offset = tile_padding / tile_size;

    // adding the ratio for every texel to our base offset to get the right pixel in our tile
    // and dividing it by the dimension of the phy. tex.
    vec2 physical_texture_coordinates = (base_xy_offset.xy + physical_tile_ratio_xy * padding_scale + padding_offset) / physical_texture_dim;

    // outputting the calculated coordinate from our physical texture
    vec4 c = texture(physical_texture_array, vec3(physical_texture_coordinates, index_quadruple.z));

    return c;
}

/*
* Fill the feedback buffer with the feedback value for a given tile.
* Here, we use the maximum lod required from the rendered tile.
*/
void update_feedback(int feedback_value, uvec4 base_offset)
{
    uint one_d_feedback_ssbo_index = base_offset.x + base_offset.y * physical_texture_dim.x + base_offset.z * physical_texture_dim.x * physical_texture_dim.y;

    //atomicMax(out_lod_feedback_values[one_d_feedback_ssbo_index], feedback_value);
    int prev = out_lod_feedback_values[one_d_feedback_ssbo_index];
    out_lod_feedback_values[one_d_feedback_ssbo_index] = max(prev, feedback_value);
    //atomicAdd(out_count_feedback_values[one_d_feedback_ssbo_index], 1);
}

/*
* Interpolate linearly between the parent and child physical texels.
*/
vec4 mix_colors(idx_tex_positions positions, int desired_level, vec2 texture_coordinates, float mix_ratio)
{
    vec4 child_color = get_physical_texture_color(positions.child_idx, texture_coordinates, positions.child_lvl);
    vec4 parent_color = get_physical_texture_color(positions.parent_idx, texture_coordinates, positions.parent_lvl);

    return enable_hierarchy == true ?
        mix(parent_color, child_color, mix_ratio) : child_color;
}

vec4 illustrate_level(float lambda, idx_tex_positions positions) {
    float mix_ratio = fract(lambda);
    int desired_level = int(ceil(lambda));

    vec4 child_color = vec4(0,0,0,1);
    vec4 parent_color = vec4(0,0,0,1);

    if (toggle_visualization == 1) {
        vec4 c0 = vec4(0,0,0,1); // black   - level 0 and below
        vec4 c1 = vec4(0,0,1,1); // blue    - level 1, 8, 15
        vec4 c2 = vec4(0,1,0,1); // green   - level 2, 9, 16
        vec4 c3 = vec4(0,1,1,1); // cyan    - level 3, 10
        vec4 c4 = vec4(1,0,0,1); // red     - level 4, 11
        vec4 c5 = vec4(1,0,1,1); // magenta - level 5, 12
        vec4 c6 = vec4(1,1,0,1); // yellow  - level 6, 13
        vec4 c7 = vec4(1,1,1,1); // white   - level 7, 14

        switch(desired_level) {
            case -2:
            case -1:
            case 0:
                parent_color = c0;
                child_color = c0;
                break;
            case 1:
                parent_color = c0;
                child_color = c1;
                break;
            case 2:
            case 9:
            case 16:
                parent_color = c1;
                child_color = c2;
                break;
            case 3:
            case 10:
                parent_color = c2;
                child_color = c3;
                break;
            case 4:
            case 11:
                parent_color = c3;
                child_color = c4;
                break;
            case 5:
            case 12:
                parent_color = c4;
                child_color = c5;
                break;
            case 6:
            case 13:
                parent_color = c5;
                child_color = c6;
                break;
            case 7:
            case 14:
                parent_color = c6;
                child_color = c7;
                break;
            case 8:
            case 15:
                parent_color = c7;
                child_color = c1;
        }

        //return vec4(lambda / 16.0f, 0.0f, 0.0f, 1.0f);
    } else {
        vec4  child_idx = positions.child_idx;
        float child_lvl = positions.child_lvl;
        vec4  parent_idx = positions.parent_idx;
        float parent_lvl = positions.parent_lvl;

        float array_size = textureSize(physical_texture_array, 0).z;

        child_color = vec4(clamp((child_lvl/8.0) , 0.0, 1.0),
                            (float(child_idx.x + child_idx.y + child_idx.z)/float(physical_texture_dim.x + physical_texture_dim.y + array_size)),
                            clamp(((child_lvl - 8.0)/8.0), 0.0, 1.0),
                            1);

        parent_color = vec4(clamp((parent_lvl/8.0), 0.0, 1.0),
                            (float(parent_idx.x + parent_idx.y + parent_idx.z)/float(physical_texture_dim.x + physical_texture_dim.y + array_size)),
                            clamp(((parent_lvl - 8.0)/8.0), 0.0, 1.0),
                            1);

    }


    return enable_hierarchy == true ?
        mix(parent_color, child_color, mix_ratio) : child_color;
    //color = vec3(child_idx.x, child_idx.y, 0.5);
}

/*
* Traverse the index hierarchy textures to find the best representation for the given texture coordinates
* based on the optimal representation calculated from the dxdy and the tiles loaded up until this frame.
*
* Example:
* lambda = 3.7
* Look at the hierarchy levels 3 (parent) and 4 (child). If the child is loaded (child_idx.w == 1) then save the
* index positions from the parent and the child node; otherwise, check the levels above.
*/
vec4 traverse_idx_hierarchy(float lambda, vec2 texture_coordinates)
{
    float mix_ratio = fract(lambda);
    int desired_level = int(ceil(lambda))*3;

    idx_tex_positions positions;

    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
 
    // Desired level can be negative when the dxdy-fct requests a coarser representation as of the root tile size
    if(desired_level <= 0)
    {
        uvec4 idx_pos = texture(hierarchical_idx_textures[0], texture_coordinates).rgba;
        positions = idx_tex_positions(0, idx_pos, 0, idx_pos);
    }
    else
    {
        // Go from desired tree level downwards to root until a loaded tile is found
        for(int i = desired_level; i >= 0; --i)
        {
            c += vec4(0.1, 0.1, 0.1, 0.0);
            uvec4 idx_child_pos = texture(hierarchical_idx_textures[i], texture_coordinates).rgba;

            // check if the requested tile is loaded and if we are not at the root level
            // enables to mix (linearly interpolate) between hierarchy levels
            if(idx_child_pos.w == 1 && i >= 1)
            {
                uvec4 idx_parent_pos = texture(hierarchical_idx_textures[i-1], texture_coordinates).rgba;
                positions = idx_tex_positions(i-1, idx_parent_pos, i, idx_child_pos);
                break;
            }

            // we are down to the root level: we cannot take the parent node from the root node;
            // therefore, we use the root node as child as well as parent for mixing
            else if(idx_child_pos.w == 1 && i == 0)
            {
                positions = idx_tex_positions(0, idx_child_pos, 0, idx_child_pos);
                break;
            }
        }
    }

    
    if (toggle_visualization == 1 || toggle_visualization == 2) {
        //c = illustrate_level(lambda, positions);
        //c = vec4(1.0, lambda, lambda, 1.0);
    } else {
        c = mix_colors(positions, desired_level, texture_coordinates, mix_ratio);
    }

    int feedback_value = desired_level;
    update_feedback(feedback_value, positions.child_idx);

    return c;
}

void main()
{
    // swap y axis
    vec2 texture_coordinates = vec2(texture_coord);
    texture_coordinates.y = 1.0 - texture_coordinates.y;
    vec4 c;

    float lambda = -dxdy();
    c = traverse_idx_hierarchy(lambda, texture_coordinates);

    out_color = c;
    //out_color = vec4(vertex_in.normal*0.5 + 0.5, 1.0);
}
