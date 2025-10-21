// vis_color_prov.glsl

INCLUDE ../common/heatmapping/wavelength_to_rainbow.glsl
INCLUDE ../common/heatmapping/colormap.glsl

struct ColorProvDebugParams {
    bool  show_normals;
    bool  show_accuracy;
    bool  show_radius_deviation;
    bool  show_output_sensitivity;
    float accuracy;
    float average_radius;
    int   channel;
    bool  heatmap;
    float heatmap_min;
    float heatmap_max;
    vec3  heatmap_min_color;
    vec3  heatmap_max_color;
};

// ---- Hilfsfunktionen ----
vec3 quick_interp(vec3 c1, vec3 c2, float v) {
    return c1 + (c2 - c1) * clamp(v, 0.0, 1.0);
}

vec3 prov_to_color(float prov_value, in ColorProvDebugParams dbg) {
    float range = max(dbg.heatmap_max - dbg.heatmap_min, 1e-8);
    float value = (prov_value - dbg.heatmap_min) / range;
    if (dbg.heatmap) {
        return quick_interp(dbg.heatmap_min_color, dbg.heatmap_max_color, value);
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
               in float prov6,
               in ColorProvDebugParams dbg) {

    vec3 view_color = vec3(0.0);

    if (dbg.show_normals) {
        vec3 n = normal;
        if (n.z < 0.0) n *= -1.0;
        view_color = n * 0.5 + 0.5;
    }
    else if (dbg.show_output_sensitivity) {
        view_color = get_output_sensitivity_color(screen_size);
    }
    else if (dbg.show_radius_deviation) {
        float max_fac  = 2.0;
        float safe_avg = max(1e-8, dbg.average_radius);
        view_color = vec3(min(max_fac, radius / safe_avg) / max_fac);
    }
    else if (dbg.channel == 0) {
        view_color = base_color;
    }
    else {
        float pv = 0.0;
        if      (dbg.channel == 1) pv = prov1;
        else if (dbg.channel == 2) pv = prov2;
        else if (dbg.channel == 3) pv = prov3;
        else if (dbg.channel == 4) pv = prov4;
        else if (dbg.channel == 5) pv = prov5;
        else if (dbg.channel == 6) pv = prov6;

        view_color = prov_to_color(pv, dbg);
    }

    if (dbg.show_accuracy) {
        view_color += vec3(dbg.accuracy, 0.0, 0.0);
    }

    return view_color;
}
