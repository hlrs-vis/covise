// vis_color_no_prov.glsl

INCLUDE ../common/heatmapping/wavelength_to_rainbow.glsl
INCLUDE ../common/heatmapping/colormap.glsl

struct ColorDebugParams {
    bool  show_normals;
    bool  show_accuracy;
    bool  show_radius_deviation;
    bool  show_output_sensitivity;
    float accuracy;
    float average_radius;
};

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
               in ColorDebugParams dbg) {

    vec3 view_color = base_color;

    if (dbg.show_normals) {
        vec3 n = normal;
        if (n.z < 0.0) n *= -1.0;
        view_color = n * 0.5 + 0.5;
    }
    else if (dbg.show_output_sensitivity) {
        view_color = get_output_sensitivity_color(screen_size);
    }
    else if (dbg.show_radius_deviation) {
        // const float max_fac        = 2.0;   // symmetrischer Kappfaktor (r/avg in [1/max_fac, max_fac])
        // const float gray_min       = 0.25;  // untere Graugrenze (vermeide "druckschwarz")
        // const float gray_max       = 0.90;  // obere Graugrenze (vermeide "papierweiß")
        // const float tint_alpha_max = 0.25;  // maximale Tönungsstärke (dezent!)
        // const float tint_gamma     = 0.6;   // nichtlinear: kleine Abweichungen schon leicht sichtbar
        // const vec3  tint_red       = vec3(0.85, 0.40, 0.40); // gedecktes Rot
        // const vec3  tint_blue      = vec3(0.38, 0.53, 0.95); // gedecktes Blau
        // float safe_avg = max(1e-8, dbg.average_radius);
        // float ratio    = max(1e-8, radius / safe_avg);
        // float max_dev  = log2(max_fac);
        // float dev      = log2(ratio);                       // (-8,8)
        // float dev_n    = clamp(dev / max_dev, -1.0, 1.0);   // normiert auf [-1,1]
        // float gray  = mix(gray_min, gray_max, 0.5 * (dev_n + 1.0));
        // vec3  base  = vec3(gray);
        // vec3  tint  = (dev_n >= 0.0) ? tint_red : tint_blue;
        // float a     = tint_alpha_max * pow(abs(dev_n), tint_gamma);
        // view_color = mix(base, tint, a);

        float safe_avg = max(1e-8, dbg.average_radius);
        float rel      = radius / safe_avg;
        float max_fac  = 2.0;
        view_color     = vec3(min(max_fac, rel) / max_fac);
    }
    else {
        view_color += vec3(dbg.accuracy, 0.0, 0.0);
    }

    return view_color;
}
