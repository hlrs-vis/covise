/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
//  ANSYS element data base
//
//  Initial version: 2006-5-12 Sven Kufer
//  Updated version: 2011-5-23 Eduardo Aguilar
//  (C) 2006 by Visenso
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ANSYS.h"
#include "EType.h"

ANSYS *ANSYS::ansys_ = NULL;

ANSYS &ANSYS::get_handle()
{
    return *ansys_;
}

ANSYS::ANSYS()
{
    if (ansys_ == NULL)
    {
        ansys_ = this;
    }

    memset(CovType_, 0, sizeof(int) * LIB_SIZE);
    memset(ANSYSNodes_, 0, sizeof(int) * LIB_SIZE);
    int elType;
    for (elType = 0; elType < LIB_SIZE; ++elType)
    {
        CovType_[elType] = TYPE_NONE;
        StressSupport_[elType] = NO_STRESS;
    }

    // LINK1 - 2D Spar (or Truss) (legacy element)
    CovType_[1] = TYPE_BAR;
    StressSupport_[1] = LINK;

    // PLANE2 - ?????????????
    CovType_[2] = TYPE_TRIANGLE;
    StressSupport_[2] = PLANE;

    // BEAM3 - 2D Elastic Beam (legacy element)
    CovType_[3] = TYPE_BAR;
    StressSupport_[3] = BEAM3;

    // BEAM4 - 3D Elastic Beam (legacy element)
    CovType_[4] = TYPE_BAR;
    StressSupport_[4] = BEAM4;

    // SOLID5 - 3D Coupled-Field Solid (legacy element)
    CovType_[5] = TYPE_HEXAEDER;
    StressSupport_[5] = SOLID;

    // COMBIN7 - Revolute Joint (legacy element)
    // Not supported by COVISE

    // LINK8 - 3D Spar (or Truss) (legacy element)
    CovType_[8] = TYPE_BAR;
    StressSupport_[8] = LINK;

    // INFIN9 - 2D Infinite Boundary (legacy element)
    // Not supported by COVISE

    // LINK10 - Tension-only or Compression-only Spar
    CovType_[10] = TYPE_BAR;
    StressSupport_[10] = LINK;

    // LINK11 - Linear Actuator
    CovType_[11] = TYPE_BAR;

    // CONTAC12 - 2D Point-to-Point Contact (legacy element)
    // Not supported by COVISE

    // PLANE13 - 2D Coupled-Field Solid (legacy element)
    // CovType_[13] = TYPE_QUAD;
    CovType_[13] = TYPE_4_NODE_PLANE;
    StressSupport_[13] = PLANE; // not tested

    // COMBIN14 - Spring Damper
    CovType_[14] = TYPE_BAR;

    // PIPE16 - Elastic Straight Pipe (legacy element)
    CovType_[16] = TYPE_BAR;

    // PIPE17 - Elastic Pipe Tee (legacy element)
    // Not supported by COVISE

    // PIPE18 - Elastic Curved Pipe (legacy element)
    // Not supported by COVISE

    // PIPE20 - Plastic Straight Thin-Walled Pipe (legacy element)
    // Supported by COVISE but not implemented

    // MASS21 - Structural Mass
    CovType_[21] = TYPE_POINT;

    // BEAM23 - 2D Plastic Beam (legacy element)
    CovType_[23] = TYPE_BAR;
    StressSupport_[23] = LINK; // not tested

    // BEAM24 - 3D Thin-Walled Beam (legacy element)
    // Implemented but actually not supported by COVISE!!!
    CovType_[24] = TYPE_BAR;
    StressSupport_[24] = LINK; // not tested

    // PLANE25 - Axisymmetric-Harmonic 4-Node Structural Solid (legacy element)
    // CovType_[25] = TYPE_QUAD;
    CovType_[25] = TYPE_4_NODE_PLANE;
    StressSupport_[25] = PLANE;

    // MATRIX27 - Stiffness, Damping, or Mass Matrix
    // Not supported by COVISE

    // SHELL28 - Shear/Twist Panel
    CovType_[28] = TYPE_QUAD;
    StressSupport_[28] = SHELL;

    // FLUID29 - 2D Axisymmetric-Harmonic Acoustic Fluid
    CovType_[29] = TYPE_4_NODE_PLANE;

    // FLUID30 - 3D Acoustic Fluid
    CovType_[30] = TYPE_HEXAEDER;

    // LINK31 - Radiation Link
    CovType_[31] = TYPE_BAR;

    // LINK32 - 2D Conduction Bar
    CovType_[32] = TYPE_BAR;

    // LINK33 - 3D Conduction Bar
    CovType_[33] = TYPE_BAR;

    // LINK34 - Convection Link
    CovType_[34] = TYPE_BAR;

    // PLANE35 - 2D 6-Node Triangular Thermal Solid
    CovType_[35] = TYPE_TRIANGLE; // Check for correctness
    StressSupport_[35] = THERMAL_PLANE;

    // SOURC36 - Current Source
    // Not supported by COVISE

    // COMBIN37 - Control (implemented with 4 nodes)
    CovType_[37] = TYPE_BAR; // Check for correctness

    // FLUID38 - Dynamic Fluid Coupling
    // Could be implemented with TYPE_BAR

    // COMBIN39 - Nonlinear Spring
    CovType_[39] = TYPE_BAR;

    // COMBIN40 - Combination
    CovType_[40] = TYPE_BAR;

    // SHELL41 - Membrane Shell (legacy element)
    // CovType_[41] = TYPE_QUAD;
    CovType_[41] = TYPE_4_NODE_PLANE;
    StressSupport_[41] = SHELL;

    // PLANE42 - 2D Structural Solid (legacy element)
    // CovType_[42] = TYPE_QUAD;
    CovType_[42] = TYPE_4_NODE_PLANE;
    StressSupport_[42] = PLANE;

    // SHELL43 ????????????????????????
    CovType_[43] = TYPE_QUAD; // This element may present degenerate configurations
    StressSupport_[43] = SHELL;

    // BEAM44 - 3D Elastic Tapered Unsymmetric Beam (legacy element) (implemented with 3 nodes)
    CovType_[44] = TYPE_BAR; // Check for correctness
    StressSupport_[44] = BEAM4;

    // SOLID45 - 3D Structural Solid (legacy element)
    CovType_[45] = TYPE_HEXAEDER;
    StressSupport_[45] = SOLID;

    // SOLID46 ??????????????????????
    CovType_[46] = TYPE_HEXAEDER; // This element may present degenerate configurations
    StressSupport_[46] = SOLID;

    // INFIN47 - 3D Infinite Boundary (legacy element)
    // Could be implemented with TYPE_QUAD

    // MATRIX50 - Superelement (or Substructure)
    // Not supported by COVISE

    // SHELL51 axisymm. str. shell ????????????????????
    CovType_[51] = TYPE_BAR;
    StressSupport_[51] = AXI_SHELL;

    // CONTAC52 - 3D Point-to-Point Contact (legacy element)
    // Not supported by COVISE

    // PLANE53 - 2D 8-Node Magnetic Solid
    CovType_[53] = TYPE_8_NODE_PLANE;
    StressSupport_[53] = THERMAL_PLANE; // not tested!

    // BEAM54 - 2D Elastic Tapered Unsymmetric Beam (legacy element)
    CovType_[54] = TYPE_BAR;
    StressSupport_[54] = BEAM3;

    // PLANE55 - 2D Thermal Solid
    // CovType_[55] = TYPE_QUAD;
    CovType_[55] = TYPE_4_NODE_PLANE;
    StressSupport_[55] = THERMAL_PLANE;

    // HYPER56 Hyperelas solid ???????????????
    CovType_[56] = TYPE_QUAD;
    StressSupport_[56] = SOLID;

    // SHELL57 - Thermal Shell (legacy element)
    // CovType_[57] = TYPE_QUAD;
    CovType_[57] = TYPE_4_NODE_PLANE;
    StressSupport_[57] = THERMAL_PLANE;

    // HYPER58 Hyperelas solid ???????????????
    CovType_[58] = TYPE_HEXAEDER;
    StressSupport_[58] = SOLID;

    // PIPE59 - Immersed Pipe or Cable (legacy element)
    // Could be implemented with TYPE_BAR

    // PIPE60 - Plastic Curved Thin-Walled Pipe (legacy element)
    // Not supported in COVISE

    // SHELL61 - Axisymmetric-Harmonic Structural Shell (output for several azymuthal angles!!!)
    CovType_[61] = TYPE_BAR;
    // StressSupport_[61] = AXI_SHELL;

    // SOLID62 - 3D Magneto-Structural Solid
    CovType_[62] = TYPE_HEXAEDER; // not tested
    StressSupport_[62] = SOLID;

    // SHELL63 - Elastic Shell (legacy element)
    // CovType_[63] = TYPE_QUAD;
    CovType_[63] = TYPE_4_NODE_PLANE;
    StressSupport_[63] = SHELL;

    // SOLID64 aniso solid ??????????????????
    CovType_[64] = TYPE_HEXAEDER;
    StressSupport_[64] = SOLID;

    // SOLID65 - 3D Reinforced Concrete Solid
    CovType_[65] = TYPE_HEXAEDER;
    StressSupport_[65] = SOLID;

    // PLANE67 - 2D Coupled Thermal-Electric Solid (legacy element)
    CovType_[67] = TYPE_4_NODE_PLANE;
    StressSupport_[67] = THERMAL_PLANE; // not tested

    // LINK68 - Coupled Thermal-Electric Line
    // Could be implemented with TYPE_BAR

    // SOLID69 - 3D Coupled Thermal-Electric Solid (legacy element)
    CovType_[69] = TYPE_HEXAEDER;
    StressSupport_[69] = THERMAL_SOLID; // not tested

    // SOLID70 - 3D Thermal Solid
    CovType_[70] = TYPE_HEXAEDER;
    StressSupport_[70] = THERMAL_SOLID;

    // MASS71 - Thermal Mass
    CovType_[71] = TYPE_POINT;

    // Hyper74 hyperelas solid 2*4 nodes ??????????????????
    CovType_[74] = TYPE_QUAD;
    StressSupport_[74] = PLANE;

    // PLANE75 - Axisymmetric-Harmonic 4-Node Thermal Solid
    // CovType_[75] = TYPE_QUAD;
    CovType_[75] = TYPE_4_NODE_PLANE;
    StressSupport_[75] = THERMAL_PLANE; // not tested

    // PLANE77 - 2D 8-Node Thermal Solid (2*4 nodes ????????)
    // CovType_[77] = TYPE_QUAD;
    CovType_[77] = TYPE_8_NODE_PLANE; // Check for correctness
    StressSupport_[77] = THERMAL_PLANE;

    // PLANE78 - Axisymmetric-Harmonic 8-Node Thermal Solid (2*3 nodes ???????)
    // CovType_[78] = TYPE_TRIANGLE; // Check for correctness (TYPE_QUAD)
    CovType_[78] = TYPE_8_NODE_PLANE;
    StressSupport_[78] = THERMAL_PLANE; // not tested

    // FLUID79 - 2D Contained Fluid
    CovType_[79] = TYPE_QUAD;

    // FLUID80 - 3D Contained Fluid
    CovType_[80] = TYPE_HEXAEDER;

    // FLUID81 - Axisymmetric-Harmonic Contained Fluid
    CovType_[81] = TYPE_QUAD;

    // PLANE82 - 2D 8-Node Structural Solid (legacy element) (2*4 ???????)
    // CovType_[82] = TYPE_QUAD; // Check for correctness
    CovType_[82] = TYPE_8_NODE_PLANE;
    StressSupport_[82] = PLANE;

    // PLANE83 - Axisymmetric-Harmonic (2*4-Node ????) 8-Node Structural Solid (legacy element)
    // CovType_[83] = TYPE_QUAD; // Check for correctness
    CovType_[83] = TYPE_8_NODE_PLANE;
    StressSupport_[83] = PLANE;

    // HYPER84 - 2-D Hyperelastic Solid 2*4 ??????????????????
    CovType_[84] = TYPE_QUAD;
    StressSupport_[84] = PLANE;

    // HYPER86 - 3-D Hyperelastic Solid ?????????????????
    CovType_[86] = TYPE_HEXAEDER;
    StressSupport_[86] = SOLID;

    // SOLID87 - 3D 10-Node Tetrahedral Thermal Solid
    // CovType_[87] = TYPE_TETRAHEDER; // Check for correctness
    CovType_[87] = TYPE_10_NODE_SOLID;
    StressSupport_[87] = THERMAL_SOLID;

    // VISCO88 - 2-D 8-Node Viscoelastic Solid ????????????????
    CovType_[88] = TYPE_QUAD;
    StressSupport_[88] = PLANE;

    // VISCO89 - 3-D 20-Node Viscoelastic Solid ????????????????
    CovType_[89] = TYPE_HEXAEDER;
    StressSupport_[89] = SOLID;

    // SOLID90 - 3D 20-Node Thermal Solid
    // CovType_[90] = TYPE_HEXAEDER; // Check for correctness
    CovType_[90] = TYPE_20_NODE_SOLID;
    StressSupport_[90] = THERMAL_SOLID;

    // SHELL91 - Nonlinear Layered Structural Shell ???????????
    CovType_[91] = TYPE_QUAD;
    StressSupport_[91] = SHELL;

    // SOLID92 - 3D 10-Node Tetrahedral Structural Solid (legacy element)
    // CovType_[92] = TYPE_TETRAHEDER; // Check for correctness
    CovType_[92] = TYPE_10_NODE_SOLID;
    StressSupport_[92] = SOLID;

    // SHELL93 - 8-Node Structural Shell ??????????????
    CovType_[93] = TYPE_QUAD;
    StressSupport_[93] = SHELL;

    // CIRCU94 - Piezoelectric Circuit
    // Not supported by COVISE

    // SOLID95 - 3D 20-Node Structural Solid
    // CovType_[95] = TYPE_HEXAEDER; // Check for correctness
    CovType_[95] = TYPE_20_NODE_SOLID;
    StressSupport_[95] = SOLID;

    // SOLID96 - 3-D 20-Node electro magnetic Solid ??????
    // Should be 3D Magnetic Scalar Solid
    CovType_[96] = TYPE_HEXAEDER; // Check for correctness
    StressSupport_[96] = THERMAL_SOLID; // Thermal ?????

    // SOLID97 - 3D Magnetic Solid
    CovType_[97] = TYPE_HEXAEDER; // Check for correctness
    StressSupport_[97] = THERMAL_SOLID; // Thermal??????

    // SOLID98 - 3-D 10-Node Tetrahedral Thermal Solid ????
    // Should be Tetrahedral Coupled-Field Solid (legacy element)
    // CovType_[98] = TYPE_TETRAHEDER; // Check for correctness
    CovType_[98] = TYPE_10_NODE_SOLID;
    StressSupport_[98] = SOLID;

    // SHELL99 - Linear Layered Structural Shell ???????????????
    CovType_[99] = TYPE_QUAD;
    StressSupport_[99] = SHELL;

    // Visco106 ?????????????????
    CovType_[106] = TYPE_QUAD;
    StressSupport_[106] = PLANE;

    // Visco107 ????????????????
    CovType_[107] = TYPE_HEXAEDER;
    StressSupport_[107] = SOLID;

    // Visco108 ???????????????
    CovType_[108] = TYPE_QUAD;
    StressSupport_[108] = PLANE;

    // TRANS109 - 2D Electromechanical Transducer
    // Could be implemented with TYPE_TRIANGLE

    // INFIN110 - 2D Infinite Solid
    // Not supported by COVISE

    // INFIN111 - 3D Infinite Solid
    // Not supported by COVISE

    // INTER115 - 3D Magnetic Interface
    // Could be implemented with TYPE_QUAD

    // FLUID116 - Coupled Thermal-Fluid Pipe
    // Could be implemented with TYPE_BAR

    // SOLID117 - 3D 20-Node Magnetic Solid
    CovType_[117] = TYPE_20_NODE_SOLID; //  TYPE_TETRAHEDER; // Check for correctness
    StressSupport_[117] = THERMAL_SOLID; // Thermal ??????

    // HF118 - 2D High-Frequency Quadrilateral Solid
    CovType_[118] = TYPE_8_NODE_PLANE;

    // HF119 - 3D High-Frequency Tetrahedral Solid
    CovType_[119] = TYPE_10_NODE_SOLID;

    // HF120 - 3D High-Frequency Brick Solid
    CovType_[120] = TYPE_20_NODE_SOLID;

    // PLANE121 - 2D 8-Node Electrostatic Solid
    CovType_[121] = TYPE_8_NODE_PLANE;

    // SOLID122 - 3D 20-Node Electrostatic Solid
    CovType_[122] = TYPE_20_NODE_SOLID;

    // SOLID123 - 3D 10-Node Tetrahedral Electrostatic Solid
    CovType_[123] = TYPE_10_NODE_SOLID;

    // CIRCU124 - Electric Circuit
    // Not supported by COVISE

    // CIRCU125 - Diode
    // Could be implemented with TYPE_BAR

    // TRANS126 - Electromechanical Transducer
    // Could be implemented with TYPE_BAR

    // SOLID127 - 3D Tetrahedral Electrostatic Solid p-Element
    CovType_[127] = TYPE_10_NODE_SOLID;

    // SOLID128 - 3D Brick Electrostatic Solid p-Element
    CovType_[128] = TYPE_20_NODE_SOLID;

    // FLUID129 - 2D Infinite Acoustic
    // Could be implemented with TYPE_BAR

    // FLUID130 - 3D Infinite Acoustic
    // Could be implemented with TYPE_QUAD

    // SHELL131 - 4-Node Thermal Shell
    // CovType_[131] =  TYPE_QUAD;
    CovType_[131] = TYPE_4_NODE_PLANE;

    // SHELL132 - 8-Node Thermal Shell
    // CovType_[132] =  TYPE_QUAD;
    CovType_[132] = TYPE_8_NODE_PLANE;

    // FLUID136 - 3D Squeeze Film Fluid Element
    // Could be implemented with TYPE_QUAD

    // FLUID138 - 3D Viscous Fluid Link Element
    // Could be implemented with TYPE_BAR

    // FLUID139 - 3D Slide Film Fluid Element
    // Could be implemented with TYPE_BAR

    // FLUID141 - 2D Fluid-Thermal (legacy element)
    // CovType_[141] = TYPE_QUAD; // Check for correctness
    CovType_[141] = TYPE_4_NODE_PLANE; // Check for correctness

    // FLUID142 - 3D Fluid-Thermal (legacy element)
    CovType_[142] = TYPE_HEXAEDER; // Check for correctness

    // Shell143 ???????????????????????????
    CovType_[143] = TYPE_QUAD; // Check for correctness
    StressSupport_[143] = SHELL;

    // ROM144 - Reduced Order Electrostatic-Structural
    // Not supported by COVISE

    // PLANE145 - 2D Quadrilateral Structural Solid p-Element
    // CovType_[145] = TYPE_QUAD; // Check for correctness
    CovType_[145] = TYPE_8_NODE_PLANE; // Check for correctness
    StressSupport_[145] = PLANE;

    // PLANE146 - 2D Triangular Structural Solid p-Element
    CovType_[146] = TYPE_TRIANGLE; // Check for correctness
    StressSupport_[146] = PLANE;

    // SOLID147 - 3D Brick Structural Solid p-Element
    // CovType_[147] = TYPE_HEXAEDER; // Check for correctness
    CovType_[147] = TYPE_20_NODE_SOLID;
    StressSupport_[147] = SOLID;

    // SOLID148 - 3D Tetrahedral Structural Solid p-Element
    // CovType_[148] = TYPE_TETRAHEDER; // Check for correctness
    CovType_[148] = TYPE_10_NODE_SOLID;
    StressSupport_[148] = SOLID;

    // SHELL150 - 8-Node Structural Shell p-Element
    // CovType_[150] = TYPE_QUAD; // Check for correctness
    CovType_[150] = TYPE_8_NODE_PLANE; // Check for correctness
    StressSupport_[150] = SHELL;

    // SURF151 - 2D Thermal Surface Effect
    // Could be implemented with TYPE_BAR

    // SURF152 - 3D Thermal Surface Effect
    // Could be implemented with TYPE_QUAD

    // SURF153 - 2D Structural Surface Effect
    // Could be implemented with TYPE_BAR

    // SURF154 - 3D Structural Surface Effect
    // Could be implemented with TYPE_QUAD

    // SURF156 - 3D Structural Surface Line Load Effect
    // Could be implemented with TYPE_BAR

    // SHELL157 - Thermal-Electric Shell
    // CovType_[157] =  TYPE_QUAD;
    CovType_[157] = TYPE_4_NODE_PLANE;

    // SOLID158 ???????????????????????
    CovType_[158] = TYPE_TETRAHEDER;
    StressSupport_[158] = SOLID;

    // LINK160 - Explicit 3D Spar (or Truss)
    // Could be implemented with TYPE_BAR

    // BEAM161 - Explicit 3D Beam
    // Could be implemented with TYPE_BAR

    // PLANE162 - Explicit 2D Structural Solid
    CovType_[162] = TYPE_4_NODE_PLANE;
    StressSupport_[162] = PLANE;

    // SHELL163 - Explicit Thin Structural Shell
    // CovType_[163] =  TYPE_QUAD;
    CovType_[163] = TYPE_4_NODE_PLANE;
    StressSupport_[163] = PLANE; // not tested

    // SOLID164 - Explicit 3D Structural Solid
    CovType_[164] = TYPE_HEXAEDER; // Check for correctness
    StressSupport_[164] = SOLID;

    // COMBI165 - Explicit Spring-Damper
    // Could be implemented with TYPE_BAR

    // MASS166 - Explicit 3D Structural Mass
    // Could be implemented with TYPE_POINT

    // LINK167 - Explicit Tension-Only Spar
    // Could be implemented with TYPE_BAR

    // SOLID168 - Explicit 3D 10-Node Tetrahedral Structural Solid
    CovType_[168] = TYPE_10_NODE_SOLID;
    StressSupport_[168] = SOLID; // not tested

    // TARGE169 - 2D Target Segment
    CovType_[169] = TYPE_TARGET_2D; // Check for correctness TYPE_TARGET ?????????????

    // TARGE170 - 3D Target Segment
    CovType_[170] = TYPE_TARGET; // Check for correctness

    // CONTA171 - 2D 2-Node Surface-to-Surface Contact
    // Could be implemented with TYPE_BAR

    // CONTA172 - 2D 3-Node Surface-to-Surface Contact
    // Could be implemented with TYPE_BAR

    // CONTA173 - 3D 4-Node Surface-to-Surface Contact
    // Could be implemented with TYPE_QUAD

    // CONTA174 - 3D 8-Node Surface-to-Surface Contact
    // Could be implemented with TYPE_QUAD

    // Target174 ?????????????????????????
    CovType_[174] = TYPE_TARGET;

    // CONTA175 - 2D/3D Node-to-Surface Contact
    // Could be implemented with TYPE_POINT

    // CONTA176 - 3D Line-to-Line Contact
    // Could be implemented with TYPE_BAR

    // CONTA177 - 3D Line-to-Surface Contact
    // Could be implemented with TYPE_BAR

    // CONTA178 - 3D Node-to-Node Contact
    // Could be implemented with TYPE_BAR

    // PRETS179 - Pretension
    // Not supported by COVISE

    // LINK180 - 3D Spar (or Truss)
    // Could be implemented with TYPE_BAR

    // SHELL181 - 4-Node Structural Shell
    // CovType_[181] = TYPE_QUAD; // Check for correctness
    CovType_[181] = TYPE_4_NODE_PLANE; // Check for correctness
    StressSupport_[181] = SHELL;

    // PLANE182 - 2D 4-Node Structural Solid
    // CovType_[182] = TYPE_QUAD; // Check for correctness
    CovType_[182] = TYPE_4_NODE_PLANE; // Check for correctness
    StressSupport_[182] = PLANE;

    // PLANE183 - 2D 8-Node or 6-Node Structural Solid
    // CovType_[183] = TYPE_QUAD; // Check for correctness Warning: double representation (keyopts)!!!!
    CovType_[183] = TYPE_8_NODE_PLANE;
    StressSupport_[183] = PLANE;

    // MPC184 - Multipoint Constraint Element
    // Not supported by COVISE (group of elements - keyopt)

    // SOLID185 - 3D 8-Node Structural Solid
    CovType_[185] = TYPE_HEXAEDER; // Check for correctness
    StressSupport_[185] = SOLID;

    // SOLID186 - 3D 20-Node Structural Solid
    // CovType_[186] = TYPE_HEXAEDER; // Check for correctness
    CovType_[186] = TYPE_20_NODE_SOLID;
    StressSupport_[186] = SOLID;

    // SOLID187 - 3D 10-Node Tetrahedral Structural Solid
    // CovType_[187] = TYPE_TETRAHEDER; // Check for correctness
    CovType_[187] = TYPE_10_NODE_SOLID;
    StressSupport_[187] = SOLID;

    // BEAM188 - 3D 2-Node Beam
    // Could be implemented with TYPE_BAR

    // BEAM189 - 3D 3-Node Beam
    // Could be implemented with TYPE_BAR

    // SOLSH190 - 3D 8-Node Structural Solid Shell
    CovType_[190] = TYPE_HEXAEDER;
    StressSupport_[190] = SOLID;

    // Solid191 ??????????????????
    CovType_[191] = TYPE_HEXAEDER;
    StressSupport_[191] = SOLID;

    // INTER192 - 2D 4-Node Gasket
    // Could be implemented with TYPE_QUAD

    // INTER193 - 2D 6-Node Gasket
    // Could be implemented with TYPE_QUAD

    // INTER194 - 3D 16-Node Gasket
    // Could be implemented with TYPE_HEXAEDER

    // INTER195 - 3D 8-Node Gasket
    // Could be implemented with TYPE_HEXAEDER

    // MESH200 - Meshing Facet
    // Not supported in COVISE (group of elements - keyopt)

    // FOLLW201 - Follower Load
    // Could be implemented with TYPE_POINT

    // INTER202 - 2D 4-Node Cohesive
    // Could be implemented with TYPE_QUAD

    // INTER203 - 2D 6-Node Cohesive
    // Could be implemented with TYPE_QUAD

    // INTER204 - 3D 16-Node Cohesive
    // Could be implemented with TYPE_HEXAEDER

    // INTER205 - 3D 8-Node Cohesive
    // Could be implemented with TYPE_HEXAEDER

    // SHELL208 - 2-Node Axisymmetric Shell
    // Could be implemented with TYPE_BAR

    // SHELL209 - 3-Node Axisymmetric Shell
    // Could be implemented with TYPE_BAR

    // CPT212 - 2D 4-Node Coupled Pore-Pressure Mechanical Solid
    // Could be implemented with TYPE_QUAD

    // CPT213 - 2D 8-Node Coupled Pore-Pressure Mechanical Solid
    // Could be implemented with TYPE_QUAD

    // COMBI214 - 2D Spring-Damper Bearing
    // Could be implemented with TYPE_BAR

    // CPT215 - 3D 8-Node Coupled Pore-Pressure Mechanical Solid
    // Could be represented with TYPE_HEXAEDER

    // CPT216 - 3D 20-Node Coupled Pore-Pressure Mechanical Solid
    // Could be implemented with TYPE_HEXAEDER

    // CPT217 - 3D 10-Node Coupled Pore-Pressure Mechanical Solid
    // Could be represented with TYPE_TETRAHEDER

    // PLANE223 - 2D 8-Node Coupled-Field Solid
    CovType_[223] = TYPE_8_NODE_PLANE;
    StressSupport_[223] = PLANE; // not tested

    // SOLID226 - 3D 20-Node Coupled-Field Solid
    // CovType_[226] = TYPE_HEXAEDER;  // Check for correctness
    CovType_[226] = TYPE_20_NODE_SOLID;
    StressSupport_[226] = SOLID;

    // SOLID227 - 3D 10-Node Coupled-Field Solid
    CovType_[227] = TYPE_10_NODE_SOLID;
    StressSupport_[227] = SOLID; // not tested

    // PLANE230 - 2D 8-Node Electric Solid
    CovType_[230] = TYPE_8_NODE_PLANE;

    // SOLID231 - 3D 20-Node Electric Solid
    CovType_[231] = TYPE_20_NODE_SOLID;

    // SOLID232 - 3D 10-Node Tetrahedral Electric Solid
    CovType_[232] = TYPE_10_NODE_SOLID;

    // SOLID236 - 3D 20-Node Electromagnetic Solid
    CovType_[236] = TYPE_20_NODE_SOLID;

    // SOLID237 - 3D 10-Node Electromagnetic Solid
    CovType_[237] = TYPE_10_NODE_SOLID;

    // SURF251 - 2D Radiosity Surface
    // Could be implemented with TYPE_BAR

    // SURF252 - 3D Thermal Radiosity Surface
    // Could be implemented with TYPE_QUAD

    // REINF264 - 3D Discrete Reinforcing
    // Not supported by COVISE (group of elements)

    // REINF265 - 3D Smeared Reinforcing
    // Not supported by COVISE (group of elements)

    // SOLID272 - General Axisymmetric Solid with 4 Base Nodes
    // Not supported by COVISE

    // SOLID273 - General Axisymmetric Solid with 8 Base Nodes
    // Not supported by COVISE

    // SHELL281 - 8-Node Structural Shell
    // CovType_[281] =  TYPE_QUAD;
    CovType_[281] = TYPE_8_NODE_PLANE;

    // SOLID285 - 3D 4-Node Tetrahedral Structural Solid with Nodal Pressures
    CovType_[285] = TYPE_TETRAHEDER;

    // PIPE288 - 3D 2-Node Pipe
    // Could be implemented with TYPE_BAR

    // PIPE289 - 3D 3-Node Pipe
    // Could be implemented with TYPE_BAR

    // ELBOW290 - 3D 3-Node Elbow
    // Not supported by COVISE

    // USER300 - User-Defined Element
    // Not supported by COVISE
}

int
ANSYS::ElementType(int routine, int noCovNodes)
{
    int ret = TYPE_NONE;

    // For the moment we will not consider target elements
    if (CovType_[routine] != TYPE_TARGET && CovType_[routine] != TYPE_TARGET_2D)
    {
        //ret = CovType_[routine];
        switch (noCovNodes)
        {
        case 1:
            ret = TYPE_POINT;
            break;
        case 2:
            ret = TYPE_BAR;
            break;
        case 3:
            ret = TYPE_TRIANGLE;
            break;

        // We must distinguish between tetrahedra and rectangles!!!!! TODO
        case 4:
            if (CovType_[routine] == TYPE_4_NODE_PLANE || CovType_[routine] == TYPE_8_NODE_PLANE)
                ret = TYPE_QUAD;
            else
                ret = TYPE_TETRAHEDER;
            break;
        case 5:
            ret = TYPE_PYRAMID;
            break;
        case 6:
            ret = TYPE_PRISM;
            break;
        case 8:
            ret = TYPE_HEXAEDER;
            break;
        }
    }
    // Note:  for target elements noCovNodes = 0  so far, therefore ret = TYPE_NONE
    else
    {
        switch (noCovNodes)
        {
        case 1:
            ret = TYPE_POINT;
            break;
        case 2:
            ret = TYPE_BAR;
            break;
        case 3:
            ret = TYPE_TRIANGLE;
            break;
        case 4:
            ret = TYPE_QUAD;
            break;
        default:
            cerr << "Could not identify target shape" << endl;
            break;
        }
    }

    // take care of degenerated elements
    // so far supported: Solid 95 and some more

    //    if (routine==95 || routine==117 || routine == 185|| routine == 186|| routine == 187|| routine == 191|| routine == 226)
    //    {
    //       if (noCovNodes==4)
    //       {
    //          ret = TYPE_TETRAHEDER;
    //       }
    //       else if (noCovNodes==5)
    //       {
    //          ret = TYPE_PYRAMID;
    //       }
    //       else if (noCovNodes==6)
    //       {
    //          ret = TYPE_PRISM;
    //       }
    //       else if (noCovNodes==8)
    //       {
    //          ret = TYPE_HEXAEDER;
    //       }
    //    }

    return ret;
}

ANSYS::StressSupport
ANSYS::getStressSupport(int elem)
{
    return StressSupport_[elem];
}

int ANSYS::getCovType(int elem)
{
    return CovType_[elem];
}
