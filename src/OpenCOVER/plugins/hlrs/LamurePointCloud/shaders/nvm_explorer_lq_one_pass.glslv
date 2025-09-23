// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 model_to_screen_matrix;
uniform mat4 inv_mv_matrix;
uniform float model_radius_scale;
uniform float point_size_factor;
uniform int render_provenance;
uniform float average_radius;
uniform float accuracy;
uniform float radius_sphere;
uniform vec3 position_sphere;
uniform bool render_normals;
uniform bool state_lense;
uniform int mode_prov_data;
uniform float heatmap_min;
uniform float heatmap_max;
uniform vec3 heatmap_min_color;
uniform vec3 heatmap_max_color;

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;
layout(location = 7) in float prov_float;
layout(location = 8) in float prov_float1;
layout(location = 9) in float prov_float2;
layout(location = 10) in float prov_float3;
layout(location = 11) in float prov_float4;
layout(location = 12) in float prov_float5;

vec3 colormap[255];

const vec4 GREEN = vec4(0.0, 1.0, 0.0, 1.0);
const vec4 WHITE = vec4(1.0, 1.0, 1.0, 1.0);
const vec4 RED = vec4(1.0, 0.0, 0.0, 1.0);

out VertexData
{
    // output to geometry shader
    vec3 pass_ms_u;
    vec3 pass_ms_v;

    vec3 pass_point_color;
    vec3 pass_normal;
}
VertexOut;

vec3 get_color(float value)
{
    if(value == -100 || value == -0.5)
    {
        return vec3(0.2, 0, 0);
    }
    int index = int(value * 255);
    index = clamp(index, 0, 254);
    return colormap[index];
}

vec3 quick_interp(vec3 color1, vec3 color2, float value) { return color1 + (color2 - color1) * clamp(value, 0, 1); }

bool is_in_sphere()
{
    return length(in_position - position_sphere) < radius_sphere;
    // return length((inv_mv_matrix * vec4(in_position, 1.0)).xyz - position_sphere) < radius_sphere;
}

float remap(float minval, float maxval, float curval) { return (curval - minval) / (maxval - minval); }

void main()
{
    colormap[0] = vec3(0.267004, 0.004874, 0.329415);
    colormap[1] = vec3(0.268510, 0.009605, 0.335427);
    colormap[2] = vec3(0.269944, 0.014625, 0.341379);
    colormap[3] = vec3(0.271305, 0.019942, 0.347269);
    colormap[4] = vec3(0.272594, 0.025563, 0.353093);
    colormap[5] = vec3(0.273809, 0.031497, 0.358853);
    colormap[6] = vec3(0.274952, 0.037752, 0.364543);
    colormap[7] = vec3(0.276022, 0.044167, 0.370164);
    colormap[8] = vec3(0.277018, 0.050344, 0.375715);
    colormap[9] = vec3(0.277941, 0.056324, 0.381191);
    colormap[10] = vec3(0.278791, 0.062145, 0.386592);
    colormap[11] = vec3(0.279566, 0.067836, 0.391917);
    colormap[12] = vec3(0.280267, 0.073417, 0.397163);
    colormap[13] = vec3(0.280894, 0.078907, 0.402329);
    colormap[14] = vec3(0.281446, 0.084320, 0.407414);
    colormap[15] = vec3(0.281924, 0.089666, 0.412415);
    colormap[16] = vec3(0.282327, 0.094955, 0.417331);
    colormap[17] = vec3(0.282656, 0.100196, 0.422160);
    colormap[18] = vec3(0.282910, 0.105393, 0.426902);
    colormap[19] = vec3(0.283091, 0.110553, 0.431554);
    colormap[20] = vec3(0.283197, 0.115680, 0.436115);
    colormap[21] = vec3(0.283229, 0.120777, 0.440584);
    colormap[22] = vec3(0.283187, 0.125848, 0.444960);
    colormap[23] = vec3(0.283072, 0.130895, 0.449241);
    colormap[24] = vec3(0.282884, 0.135920, 0.453427);
    colormap[25] = vec3(0.282623, 0.140926, 0.457517);
    colormap[26] = vec3(0.282290, 0.145912, 0.461510);
    colormap[27] = vec3(0.281887, 0.150881, 0.465405);
    colormap[28] = vec3(0.281412, 0.155834, 0.469201);
    colormap[29] = vec3(0.280868, 0.160771, 0.472899);
    colormap[30] = vec3(0.280255, 0.165693, 0.476498);
    colormap[31] = vec3(0.279574, 0.170599, 0.479997);
    colormap[32] = vec3(0.278826, 0.175490, 0.483397);
    colormap[33] = vec3(0.278012, 0.180367, 0.486697);
    colormap[34] = vec3(0.277134, 0.185228, 0.489898);
    colormap[35] = vec3(0.276194, 0.190074, 0.493001);
    colormap[36] = vec3(0.275191, 0.194905, 0.496005);
    colormap[37] = vec3(0.274128, 0.199721, 0.498911);
    colormap[38] = vec3(0.273006, 0.204520, 0.501721);
    colormap[39] = vec3(0.271828, 0.209303, 0.504434);
    colormap[40] = vec3(0.270595, 0.214069, 0.507052);
    colormap[41] = vec3(0.269308, 0.218818, 0.509577);
    colormap[42] = vec3(0.267968, 0.223549, 0.512008);
    colormap[43] = vec3(0.266580, 0.228262, 0.514349);
    colormap[44] = vec3(0.265145, 0.232956, 0.516599);
    colormap[45] = vec3(0.263663, 0.237631, 0.518762);
    colormap[46] = vec3(0.262138, 0.242286, 0.520837);
    colormap[47] = vec3(0.260571, 0.246922, 0.522828);
    colormap[48] = vec3(0.258965, 0.251537, 0.524736);
    colormap[49] = vec3(0.257322, 0.256130, 0.526563);
    colormap[50] = vec3(0.255645, 0.260703, 0.528312);
    colormap[51] = vec3(0.253935, 0.265254, 0.529983);
    colormap[52] = vec3(0.252194, 0.269783, 0.531579);
    colormap[53] = vec3(0.250425, 0.274290, 0.533103);
    colormap[54] = vec3(0.248629, 0.278775, 0.534556);
    colormap[55] = vec3(0.246811, 0.283237, 0.535941);
    colormap[56] = vec3(0.244972, 0.287675, 0.537260);
    colormap[57] = vec3(0.243113, 0.292092, 0.538516);
    colormap[58] = vec3(0.241237, 0.296485, 0.539709);
    colormap[59] = vec3(0.239346, 0.300855, 0.540844);
    colormap[60] = vec3(0.237441, 0.305202, 0.541921);
    colormap[61] = vec3(0.235526, 0.309527, 0.542944);
    colormap[62] = vec3(0.233603, 0.313828, 0.543914);
    colormap[63] = vec3(0.231674, 0.318106, 0.544834);
    colormap[64] = vec3(0.229739, 0.322361, 0.545706);
    colormap[65] = vec3(0.227802, 0.326594, 0.546532);
    colormap[66] = vec3(0.225863, 0.330805, 0.547314);
    colormap[67] = vec3(0.223925, 0.334994, 0.548053);
    colormap[68] = vec3(0.221989, 0.339161, 0.548752);
    colormap[69] = vec3(0.220057, 0.343307, 0.549413);
    colormap[70] = vec3(0.218130, 0.347432, 0.550038);
    colormap[71] = vec3(0.216210, 0.351535, 0.550627);
    colormap[72] = vec3(0.214298, 0.355619, 0.551184);
    colormap[73] = vec3(0.212395, 0.359683, 0.551710);
    colormap[74] = vec3(0.210503, 0.363727, 0.552206);
    colormap[75] = vec3(0.208623, 0.367752, 0.552675);
    colormap[76] = vec3(0.206756, 0.371758, 0.553117);
    colormap[77] = vec3(0.204903, 0.375746, 0.553533);
    colormap[78] = vec3(0.203063, 0.379716, 0.553925);
    colormap[79] = vec3(0.201239, 0.383670, 0.554294);
    colormap[80] = vec3(0.199430, 0.387607, 0.554642);
    colormap[81] = vec3(0.197636, 0.391528, 0.554969);
    colormap[82] = vec3(0.195860, 0.395433, 0.555276);
    colormap[83] = vec3(0.194100, 0.399323, 0.555565);
    colormap[84] = vec3(0.192357, 0.403199, 0.555836);
    colormap[85] = vec3(0.190631, 0.407061, 0.556089);
    colormap[86] = vec3(0.188923, 0.410910, 0.556326);
    colormap[87] = vec3(0.187231, 0.414746, 0.556547);
    colormap[88] = vec3(0.185556, 0.418570, 0.556753);
    colormap[89] = vec3(0.183898, 0.422383, 0.556944);
    colormap[90] = vec3(0.182256, 0.426184, 0.557120);
    colormap[91] = vec3(0.180629, 0.429975, 0.557282);
    colormap[92] = vec3(0.179019, 0.433756, 0.557430);
    colormap[93] = vec3(0.177423, 0.437527, 0.557565);
    colormap[94] = vec3(0.175841, 0.441290, 0.557685);
    colormap[95] = vec3(0.174274, 0.445044, 0.557792);
    colormap[96] = vec3(0.172719, 0.448791, 0.557885);
    colormap[97] = vec3(0.171176, 0.452530, 0.557965);
    colormap[98] = vec3(0.169646, 0.456262, 0.558030);
    colormap[99] = vec3(0.168126, 0.459988, 0.558082);
    colormap[100] = vec3(0.166617, 0.463708, 0.558119);
    colormap[101] = vec3(0.165117, 0.467423, 0.558141);
    colormap[102] = vec3(0.163625, 0.471133, 0.558148);
    colormap[103] = vec3(0.162142, 0.474838, 0.558140);
    colormap[104] = vec3(0.160665, 0.478540, 0.558115);
    colormap[105] = vec3(0.159194, 0.482237, 0.558073);
    colormap[106] = vec3(0.157729, 0.485932, 0.558013);
    colormap[107] = vec3(0.156270, 0.489624, 0.557936);
    colormap[108] = vec3(0.154815, 0.493313, 0.557840);
    colormap[109] = vec3(0.153364, 0.497000, 0.557724);
    colormap[110] = vec3(0.151918, 0.500685, 0.557587);
    colormap[111] = vec3(0.150476, 0.504369, 0.557430);
    colormap[112] = vec3(0.149039, 0.508051, 0.557250);
    colormap[113] = vec3(0.147607, 0.511733, 0.557049);
    colormap[114] = vec3(0.146180, 0.515413, 0.556823);
    colormap[115] = vec3(0.144759, 0.519093, 0.556572);
    colormap[116] = vec3(0.143343, 0.522773, 0.556295);
    colormap[117] = vec3(0.141935, 0.526453, 0.555991);
    colormap[118] = vec3(0.140536, 0.530132, 0.555659);
    colormap[119] = vec3(0.139147, 0.533812, 0.555298);
    colormap[120] = vec3(0.137770, 0.537492, 0.554906);
    colormap[121] = vec3(0.136408, 0.541173, 0.554483);
    colormap[122] = vec3(0.135066, 0.544853, 0.554029);
    colormap[123] = vec3(0.133743, 0.548535, 0.553541);
    colormap[124] = vec3(0.132444, 0.552216, 0.553018);
    colormap[125] = vec3(0.131172, 0.555899, 0.552459);
    colormap[126] = vec3(0.129933, 0.559582, 0.551864);
    colormap[127] = vec3(0.128729, 0.563265, 0.551229);
    colormap[128] = vec3(0.127568, 0.566949, 0.550556);
    colormap[129] = vec3(0.126453, 0.570633, 0.549841);
    colormap[130] = vec3(0.125394, 0.574318, 0.549086);
    colormap[131] = vec3(0.124395, 0.578002, 0.548287);
    colormap[132] = vec3(0.123463, 0.581687, 0.547445);
    colormap[133] = vec3(0.122606, 0.585371, 0.546557);
    colormap[134] = vec3(0.121831, 0.589055, 0.545623);
    colormap[135] = vec3(0.121148, 0.592739, 0.544641);
    colormap[136] = vec3(0.120565, 0.596422, 0.543611);
    colormap[137] = vec3(0.120092, 0.600104, 0.542530);
    colormap[138] = vec3(0.119738, 0.603785, 0.541400);
    colormap[139] = vec3(0.119512, 0.607464, 0.540218);
    colormap[140] = vec3(0.119423, 0.611141, 0.538982);
    colormap[141] = vec3(0.119483, 0.614817, 0.537692);
    colormap[142] = vec3(0.119699, 0.618490, 0.536347);
    colormap[143] = vec3(0.120081, 0.622161, 0.534946);
    colormap[144] = vec3(0.120638, 0.625828, 0.533488);
    colormap[145] = vec3(0.121380, 0.629492, 0.531973);
    colormap[146] = vec3(0.122312, 0.633153, 0.530398);
    colormap[147] = vec3(0.123444, 0.636809, 0.528763);
    colormap[148] = vec3(0.124780, 0.640461, 0.527068);
    colormap[149] = vec3(0.126326, 0.644107, 0.525311);
    colormap[150] = vec3(0.128087, 0.647749, 0.523491);
    colormap[151] = vec3(0.130067, 0.651384, 0.521608);
    colormap[152] = vec3(0.132268, 0.655014, 0.519661);
    colormap[153] = vec3(0.134692, 0.658636, 0.517649);
    colormap[154] = vec3(0.137339, 0.662252, 0.515571);
    colormap[155] = vec3(0.140210, 0.665859, 0.513427);
    colormap[156] = vec3(0.143303, 0.669459, 0.511215);
    colormap[157] = vec3(0.146616, 0.673050, 0.508936);
    colormap[158] = vec3(0.150148, 0.676631, 0.506589);
    colormap[159] = vec3(0.153894, 0.680203, 0.504172);
    colormap[160] = vec3(0.157851, 0.683765, 0.501686);
    colormap[161] = vec3(0.162016, 0.687316, 0.499129);
    colormap[162] = vec3(0.166383, 0.690856, 0.496502);
    colormap[163] = vec3(0.170948, 0.694384, 0.493803);
    colormap[164] = vec3(0.175707, 0.697900, 0.491033);
    colormap[165] = vec3(0.180653, 0.701402, 0.488189);
    colormap[166] = vec3(0.185783, 0.704891, 0.485273);
    colormap[167] = vec3(0.191090, 0.708366, 0.482284);
    colormap[168] = vec3(0.196571, 0.711827, 0.479221);
    colormap[169] = vec3(0.202219, 0.715272, 0.476084);
    colormap[170] = vec3(0.208030, 0.718701, 0.472873);
    colormap[171] = vec3(0.214000, 0.722114, 0.469588);
    colormap[172] = vec3(0.220124, 0.725509, 0.466226);
    colormap[173] = vec3(0.226397, 0.728888, 0.462789);
    colormap[174] = vec3(0.232815, 0.732247, 0.459277);
    colormap[175] = vec3(0.239374, 0.735588, 0.455688);
    colormap[176] = vec3(0.246070, 0.738910, 0.452024);
    colormap[177] = vec3(0.252899, 0.742211, 0.448284);
    colormap[178] = vec3(0.259857, 0.745492, 0.444467);
    colormap[179] = vec3(0.266941, 0.748751, 0.440573);
    colormap[180] = vec3(0.274149, 0.751988, 0.436601);
    colormap[181] = vec3(0.281477, 0.755203, 0.432552);
    colormap[182] = vec3(0.288921, 0.758394, 0.428426);
    colormap[183] = vec3(0.296479, 0.761561, 0.424223);
    colormap[184] = vec3(0.304148, 0.764704, 0.419943);
    colormap[185] = vec3(0.311925, 0.767822, 0.415586);
    colormap[186] = vec3(0.319809, 0.770914, 0.411152);
    colormap[187] = vec3(0.327796, 0.773980, 0.406640);
    colormap[188] = vec3(0.335885, 0.777018, 0.402049);
    colormap[189] = vec3(0.344074, 0.780029, 0.397381);
    colormap[190] = vec3(0.352360, 0.783011, 0.392636);
    colormap[191] = vec3(0.360741, 0.785964, 0.387814);
    colormap[192] = vec3(0.369214, 0.788888, 0.382914);
    colormap[193] = vec3(0.377779, 0.791781, 0.377939);
    colormap[194] = vec3(0.386433, 0.794644, 0.372886);
    colormap[195] = vec3(0.395174, 0.797475, 0.367757);
    colormap[196] = vec3(0.404001, 0.800275, 0.362552);
    colormap[197] = vec3(0.412913, 0.803041, 0.357269);
    colormap[198] = vec3(0.421908, 0.805774, 0.351910);
    colormap[199] = vec3(0.430983, 0.808473, 0.346476);
    colormap[200] = vec3(0.440137, 0.811138, 0.340967);
    colormap[201] = vec3(0.449368, 0.813768, 0.335384);
    colormap[202] = vec3(0.458674, 0.816363, 0.329727);
    colormap[203] = vec3(0.468053, 0.818921, 0.323998);
    colormap[204] = vec3(0.477504, 0.821444, 0.318195);
    colormap[205] = vec3(0.487026, 0.823929, 0.312321);
    colormap[206] = vec3(0.496615, 0.826376, 0.306377);
    colormap[207] = vec3(0.506271, 0.828786, 0.300362);
    colormap[208] = vec3(0.515992, 0.831158, 0.294279);
    colormap[209] = vec3(0.525776, 0.833491, 0.288127);
    colormap[210] = vec3(0.535621, 0.835785, 0.281908);
    colormap[211] = vec3(0.545524, 0.838039, 0.275626);
    colormap[212] = vec3(0.555484, 0.840254, 0.269281);
    colormap[213] = vec3(0.565498, 0.842430, 0.262877);
    colormap[214] = vec3(0.575563, 0.844566, 0.256415);
    colormap[215] = vec3(0.585678, 0.846661, 0.249897);
    colormap[216] = vec3(0.595839, 0.848717, 0.243329);
    colormap[217] = vec3(0.606045, 0.850733, 0.236712);
    colormap[218] = vec3(0.616293, 0.852709, 0.230052);
    colormap[219] = vec3(0.626579, 0.854645, 0.223353);
    colormap[220] = vec3(0.636902, 0.856542, 0.216620);
    colormap[221] = vec3(0.647257, 0.858400, 0.209861);
    colormap[222] = vec3(0.657642, 0.860219, 0.203082);
    colormap[223] = vec3(0.668054, 0.861999, 0.196293);
    colormap[224] = vec3(0.678489, 0.863742, 0.189503);
    colormap[225] = vec3(0.688944, 0.865448, 0.182725);
    colormap[226] = vec3(0.699415, 0.867117, 0.175971);
    colormap[227] = vec3(0.709898, 0.868751, 0.169257);
    colormap[228] = vec3(0.720391, 0.870350, 0.162603);
    colormap[229] = vec3(0.730889, 0.871916, 0.156029);
    colormap[230] = vec3(0.741388, 0.873449, 0.149561);
    colormap[231] = vec3(0.751884, 0.874951, 0.143228);
    colormap[232] = vec3(0.762373, 0.876424, 0.137064);
    colormap[233] = vec3(0.772852, 0.877868, 0.131109);
    colormap[234] = vec3(0.783315, 0.879285, 0.125405);
    colormap[235] = vec3(0.793760, 0.880678, 0.120005);
    colormap[236] = vec3(0.804182, 0.882046, 0.114965);
    colormap[237] = vec3(0.814576, 0.883393, 0.110347);
    colormap[238] = vec3(0.824940, 0.884720, 0.106217);
    colormap[239] = vec3(0.835270, 0.886029, 0.102646);
    colormap[240] = vec3(0.845561, 0.887322, 0.099702);
    colormap[241] = vec3(0.855810, 0.888601, 0.097452);
    colormap[242] = vec3(0.866013, 0.889868, 0.095953);
    colormap[243] = vec3(0.876168, 0.891125, 0.095250);
    colormap[244] = vec3(0.886271, 0.892374, 0.095374);
    colormap[245] = vec3(0.896320, 0.893616, 0.096335);
    colormap[246] = vec3(0.906311, 0.894855, 0.098125);
    colormap[247] = vec3(0.916242, 0.896091, 0.100717);
    colormap[248] = vec3(0.926106, 0.897330, 0.104071);
    colormap[249] = vec3(0.935904, 0.898570, 0.108131);
    colormap[250] = vec3(0.945636, 0.899815, 0.112838);
    colormap[251] = vec3(0.955300, 0.901065, 0.118128);
    colormap[252] = vec3(0.964894, 0.902323, 0.123941);
    colormap[253] = vec3(0.974417, 0.903590, 0.130215);
    colormap[254] = vec3(0.983868, 0.904867, 0.136897);

    vec3 ms_n = normalize(in_normal.xyz);
    vec3 ms_u;

    //**compute tangent vectors**//
    if(ms_n.z != 0.0)
    {
        ms_u = vec3(1, 1, (-ms_n.x - ms_n.y) / ms_n.z);
    }
    else if(ms_n.y != 0.0)
    {
        ms_u = vec3(1, (-ms_n.x - ms_n.z) / ms_n.y, 1);
    }
    else
    {
        ms_u = vec3((-ms_n.y - ms_n.z) / ms_n.x, 1, 1);
    }

    //**assign tangent vectors**//
    VertexOut.pass_ms_u = normalize(ms_u) * point_size_factor * model_radius_scale * in_radius;
    VertexOut.pass_ms_v = normalize(cross(ms_n, ms_u)) * point_size_factor * model_radius_scale * in_radius;

    VertexOut.pass_normal = normalize((inv_mv_matrix * vec4(in_normal, 0.0)).xyz);

#if 1
    if(state_lense && is_in_sphere())
    {
        float value;
        switch(mode_prov_data)
        {
        case 0:
        {
            value = (prov_float - heatmap_min) / (heatmap_max - heatmap_min);

            // VertexOut.pass_point_color = get_color(value);
            VertexOut.pass_point_color = quick_interp(heatmap_min_color, heatmap_max_color, value); // lerp(heatmap_min_color, heatmap_max_color, value);
            break;
        }
        case 1:
        {
            value = (prov_float1 - heatmap_min) / (heatmap_max - heatmap_min);

            VertexOut.pass_point_color = get_color(value);
            break;
        }
        case 2:
        {
            value = (prov_float2 * 0.5 - heatmap_min) / (heatmap_max - heatmap_min);

            // VertexOut.pass_point_color = get_color(value);
            VertexOut.pass_point_color = quick_interp(heatmap_min_color, heatmap_max_color, value); // lerp(heatmap_min_color, heatmap_max_color, value);
            break;
        }
        case 3:
        {
            VertexOut.pass_point_color = vec3(prov_float3 / 255.0, prov_float4 / 255.0, prov_float5 / 255.0);
            break;
        }
        }
        // float u = clamp( 4.0f, 0.0, 1.0 );
        // float u = clamp( float(prov_float), 0.0, 0.01 );
        // if( value < 1.0)
        // {
        //     VertexOut.pass_point_color = vec3(1.0, 0.0, 0.0);

        // } else {
        //     VertexOut.pass_point_color = vec3(0.0, 1.0, 0.0);

        // }
        // if( u < 0.005 )
        // {
        //     VertexOut.pass_point_color = mix( GREEN, WHITE, remap( 0.0, 0.005, u ) ).xyz;
        // }
        // else
        // {
        //     VertexOut.pass_point_color = mix( WHITE, RED, remap( 0.005, 0.01, u ) ).xyz;
        // }
        // VertexOut.pass_point_color = vec3(0.0, 1.0, 0.0);
    }
    else
    {
        if(render_normals)
        {
            vec3 normals_normalized = in_normal + vec3(1.0) / 2;
            VertexOut.pass_point_color = vec3(normals_normalized);
        }
        else
        {
            VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        }
    }
#else
    if(prov_float > 0.0)
    {
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    }
    else
    {
        VertexOut.pass_point_color = vec3(0.0, 1.0, 0.0);
    }
#endif

    gl_Position = vec4(in_position, 1.0);
}

