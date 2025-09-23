// Hochstabile Tangentenvektorberechnung mit externer Skalierung
void compute_stable_tangent_vectors(in vec3 normal, in float radius, in float scale_factor, out vec3 tangent_u, out vec3 tangent_v) {
    
    // Robuste Normalisierung mit Fallback
    vec3 n = normal;
    float normal_length = length(n);
    
    // Degenerate normale abfangen
    if (normal_length < 1e-6) {
        // Fallback für degenerierte Normalen
        n = vec3(0.0, 0.0, 1.0);
    } else {
        n = n / normal_length;
    }
    
    // Finde die Komponente mit dem kleinsten Absolutwert
    // für maximale numerische Stabilität
    vec3 abs_n = abs(n);
    vec3 axis;
    
    if (abs_n.x <= abs_n.y && abs_n.x <= abs_n.z) {
        // x ist kleinste Komponente
        axis = vec3(1.0, 0.0, 0.0);
    } else if (abs_n.y <= abs_n.z) {
        // y ist kleinste Komponente  
        axis = vec3(0.0, 1.0, 0.0);
    } else {
        // z ist kleinste Komponente
        axis = vec3(0.0, 0.0, 1.0);
    }
    
    // Gram-Schmidt Orthogonalisierung für maximale Stabilität
    vec3 u_unnormalized = axis - dot(axis, n) * n;
    float u_length = length(u_unnormalized);
    
    // Sicherheitscheck - sollte nie auftreten, aber zur Sicherheit
    if (u_length < 1e-6) {
        // Notfall-Fallback
        if (abs(n.z) < 0.9) {
            u_unnormalized = vec3(0.0, 0.0, 1.0) - dot(vec3(0.0, 0.0, 1.0), n) * n;
        } else {
            u_unnormalized = vec3(1.0, 0.0, 0.0) - dot(vec3(1.0, 0.0, 0.0), n) * n;
        }
        u_length = length(u_unnormalized);
    }
    
    vec3 u = u_unnormalized / u_length;
    vec3 v = cross(n, u);  // Bereits normalisiert da n und u orthonormal sind
    
    // Einfache Radiusskalierung - externe Kontrolle
    float final_radius = radius * scale_factor;
    
    tangent_u = u * final_radius;
    tangent_v = v * final_radius;
}

// Alternative: Ultra-stabile Frisvad-Methode (2012) mit externer Skalierung
// Optimal für GPUs und extrem numerisch stabil
void compute_frisvad_tangents(in vec3 normal, in float radius, in float scale_factor, out vec3 tangent_u, out vec3 tangent_v) {
    
    vec3 n = normal;
    float normal_length = length(n);
    
    if (normal_length < 1e-6) {
        n = vec3(0.0, 0.0, 1.0);
    } else {
        n = n / normal_length;
    }
    
    // Frisvad's method - extrem stabil und effizient
    vec3 u, v;
    
    if (n.z < -0.9999999) { // Handle n = (0,0,-1)
        u = vec3(0.0, -1.0, 0.0);
        v = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = vec3(1.0 - n.x * n.x * a, b, -n.x);
        v = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    
    // Einfache Radiusskalierung - externe Kontrolle
    float final_radius = radius * scale_factor;
    
    tangent_u = u * final_radius;
    tangent_v = v * final_radius;
}

// Überladene Versionen mit mehreren Skalierungsfaktoren für erweiterte Kontrolle
void compute_stable_tangent_vectors_multi_scale(in vec3 normal, in float radius, 
                                               in float global_scale, in float local_scale, 
                                               out vec3 tangent_u, out vec3 tangent_v) {
    
    vec3 n = normal;
    float normal_length = length(n);
    
    if (normal_length < 1e-6) {
        n = vec3(0.0, 0.0, 1.0);
    } else {
        n = n / normal_length;
    }
    
    vec3 abs_n = abs(n);
    vec3 axis;
    
    if (abs_n.x <= abs_n.y && abs_n.x <= abs_n.z) {
        axis = vec3(1.0, 0.0, 0.0);
    } else if (abs_n.y <= abs_n.z) {
        axis = vec3(0.0, 1.0, 0.0);
    } else {
        axis = vec3(0.0, 0.0, 1.0);
    }
    
    vec3 u_unnormalized = axis - dot(axis, n) * n;
    float u_length = length(u_unnormalized);
    
    if (u_length < 1e-6) {
        if (abs(n.z) < 0.9) {
            u_unnormalized = vec3(0.0, 0.0, 1.0) - dot(vec3(0.0, 0.0, 1.0), n) * n;
        } else {
            u_unnormalized = vec3(1.0, 0.0, 0.0) - dot(vec3(1.0, 0.0, 0.0), n) * n;
        }
        u_length = length(u_unnormalized);
    }
    
    vec3 u = u_unnormalized / u_length;
    vec3 v = cross(n, u);
    
    // Mehrstufige Skalierung
    float final_radius = radius * global_scale * local_scale;
    
    tangent_u = u * final_radius;
    tangent_v = v * final_radius;
}

// Utility-Funktion für komplexe Skalierungslogik
float compute_adaptive_scale(float base_radius, float distance_factor, float quality_factor) {
    // Beispiel für adaptive Skalierung basierend auf verschiedenen Faktoren
    return base_radius * distance_factor * quality_factor;
}
