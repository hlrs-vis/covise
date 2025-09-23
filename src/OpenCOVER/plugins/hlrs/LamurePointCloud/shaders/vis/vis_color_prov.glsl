// vis_color_prov.glsl

INCLUDE ../common/heatmapping/wavelength_to_rainbow.glsl
INCLUDE ../common/heatmapping/colormap.glsl

uniform bool  show_normals;
uniform bool  show_accuracy;
uniform bool  show_radius_deviation;
uniform bool  show_output_sensitivity;

uniform float accuracy;
uniform float average_radius;

uniform int   channel;
uniform bool  heatmap;
uniform float heatmap_min;
uniform float heatmap_max;
uniform vec3  heatmap_min_color;
uniform vec3  heatmap_max_color;


// ---- Hilfsfunktionen ----
vec3 quick_interp(vec3 c1, vec3 c2, float v) {
    return c1 + (c2 - c1) * clamp(v, 0.0, 1.0);
}

vec3 prov_to_color(float prov_value) {
    float value = (prov_value - heatmap_min) / (heatmap_max - heatmap_min);
    if (heatmap) {
        return quick_interp(heatmap_min_color, heatmap_max_color, value);
    } else {
        init_colormap();
        return get_colormap_value(value);
    }
}

vec3 get_output_sensitivity_color(float screen_size) {
    float min_size = 0.0;
    float max_size = 10.0;
    float val = clamp(screen_size, min_size, max_size);
    return data_value_to_rainbow(val, min_size, max_size);
}

vec3 get_color(in vec3 position,
               in vec3 normal,
               in vec3 base_color,
               in float radius,
               in float screen_size,
               in float prov1, 
               in float prov2, 
               in float prov3,
               in float prov4, 
               in float prov5, 
               in float prov6) {

    vec3  view_color = vec3(0.0);

    if (show_normals) {
        vec3 n = normal;
        if (n.z < 0.0) n *= -1.0;
        view_color = n * 0.5 + 0.5;
    }
    else if (show_output_sensitivity) {
        view_color = get_output_sensitivity_color(screen_size);
    }
    else if (show_radius_deviation) {
        float max_fac  = 2.0;
        float safe_avg = max(1e-8, average_radius);
        view_color = vec3(min(max_fac, radius / safe_avg) / max_fac);
    }
    else if (channel == 0) {
        view_color = base_color;
    }
    else {
        float pv = 0.0;
        if      (channel == 1) pv = prov1;
        else if (channel == 2) pv = prov2;
        else if (channel == 3) pv = prov3;
        else if (channel == 4) pv = prov4;
        else if (channel == 5) pv = prov5;
        else if (channel == 6) pv = prov6;

        view_color = prov_to_color(pv);
    }

    if (show_accuracy) {
        view_color += vec3(accuracy, 0.0, 0.0);
    }

    return view_color;
}