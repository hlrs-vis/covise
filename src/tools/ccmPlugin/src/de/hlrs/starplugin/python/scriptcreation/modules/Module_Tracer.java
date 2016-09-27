package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.util.Vec;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_Tracer extends Module {

    private Vec no_startp;              //Tracer_1.set_no_startp( 1, 100, 10 )
    private Vec startpoint1;            //Tracer_1.set_startpoint1( 0, 0, 0 )
    private Vec startpoint2;            //Tracer_1.set_startpoint2( 1, 0, 0 )
    private Vec direction;              //Tracer_1.set_direction( 0, 1, 0 )
//    private Vec cyl_axis;               //Tracer_1.set_cyl_axis( 0, 0, 1 )
//    private float cyl_radius;           //Tracer_1.set_cyl_radius( 1.000000 )
//    private float cyl_height;           //Tracer_1.set_cyl_height( 1.000000 )
//    private Vec cyl_bottompoint_on_axis;//Tracer_1.set_cyl_bottompoint_on_axis( 1, 0, 0 )
//    private Vec Displacement;           //Tracer_1.set_Displacement( 0, 0, 0 )
    private String tdirection;             //Tracer_1.set_tdirection( 1 )
    private int whatout;                //Tracer_1.set_whatout( 1 )
    private int taskType;               //Tracer_1.set_taskType( 1 )
    private int startStyle;             //Tracer_1.set_startStyle( 2 )
//    private float trace_eps;            //Tracer_1.set_trace_eps( 0.000010 )
//    private float trace_abs;            //Tracer_1.set_trace_abs( 0.000100 )
//    private float grid_tol;             //Tracer_1.set_grid_tol( 0.000100 )
    private float trace_len;            //Tracer_1.set_trace_len( 10 )
//    private float min_vel;              //Tracer_1.set_min_vel( 0.001000 )
//    private int MaxPoints;              //Tracer_1.set_MaxPoints( 1000 )
//    private float stepDuration;         //Tracer_1.set_stepDuration( 0.010000 )
//    private float NoCycles;             //Tracer_1.set_NoCycles( 1 )
//    private boolean NoInterpolation;    //Tracer_1.set_NoInterpolation( "FALSE" )
//    private boolean ThrowNewParticles;  //Tracer_1.set_ThrowNewParticles( "FALSE" )
//    private float ParticlesReleaseRate; //Tracer_1.set_ParticlesReleaseRate( 0.000000 )
//    private boolean RandomOffset;       //Tracer_1.set_RandomOffset( "FALSE" )
//    private boolean RandomStartpoints;  //Tracer_1.set_RandomStartpoints( "FALSE" )
//    private float divideCell;           //Tracer_1.set_divideCell( 0.125000 )
    private float maxOutOfDomain;       //Tracer_1.set_maxOutOfDomain( 0.8 )
//    private int NoWThreads;             //Tracer_1.set_NoWThreads( 1 )
//    private float SearchLevel;          //Tracer_1.set_SearchLevel( 0 )
//    private float SkipInitialSteps;     //Tracer_1.set_SkipInitialSteps( 0 )
    private String color;               //Tracer_1.set_color( "red" )

    public Module_Tracer(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_Tracer, Name, param_pos_x, param_pos_y);
        this.no_startp = new Vec(1, 100, 10);
        this.startpoint1 = new Vec(0, 0, 0);
        this.startpoint2 = new Vec(0, 0, 1);
        this.direction = new Vec(0, 1, 0);
//        this.cyl_axis = new Vec(0, 1, 0);
//        this.cyl_radius = 1;
//        this.cyl_height = 1;
//        this.cyl_bottompoint_on_axis = new Vec(1, 0, 0);
//        this.Displacement = new Vec(0, 0, 0);
        this.tdirection = "1";
        this.whatout = 1;
        this.taskType = 1;
        this.startStyle = 2;
//        this.trace_eps = 0.00001f;
//        this.trace_abs = 0.0001f;
//        this.grid_tol = 0.0001f;
        this.trace_len = 5;
//        this.min_vel = 0.001f;
//        this.MaxPoints = 1000;
//        this.stepDuration = 0.01f;
//        this.NoCycles = 1;
//        this.NoInterpolation = false;
//        this.ThrowNewParticles = false;
//        this.ParticlesReleaseRate = 0;
//        this.RandomOffset = false;
//        this.RandomStartpoints = false;
//        this.divideCell = 0.125f;
        this.maxOutOfDomain = 0.5f;
//        this.NoWThreads = 1;
//        this.SearchLevel = 0;
//        this.SkipInitialSteps = 0;
        this.color = "red";
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[20];

        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: Tracer";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=Tracer()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.toString(this.param_pos_y) + ")";
        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_no_startp(" + this.no_startp.x + "," + this.no_startp.y + "," + this.no_startp.z + ")";
        ExportStringLines[10] = this.Name + ".set_startpoint1(" + this.startpoint1.x + "," + this.startpoint1.y + "," + this.startpoint1.z + ")";
        ExportStringLines[11] = this.Name + ".set_startpoint2(" + this.startpoint2.x + "," + this.startpoint2.y + "," + this.startpoint2.z + ")";
        ExportStringLines[12] = this.Name + ".set_direction(" + this.direction.x + "," + this.direction.y + "," + this.direction.z + ")";
        ExportStringLines[13] = this.Name + ".set_whatout(" + this.whatout + ")";
        ExportStringLines[14] = this.Name + ".set_startStyle(" + this.startStyle + ")";
        ExportStringLines[15] = this.Name + ".set_trace_len(" + this.trace_len + ")";
//        ExportStringLines[16] = this.Name + ".set_MaxPoints(" + this.MaxPoints + ")";
        ExportStringLines[16] = this.Name + ".set_maxOutOfDomain(" + this.maxOutOfDomain + ")";
        ExportStringLines[17] = this.Name + ".set_color(\"" + this.color + "\")";
        ExportStringLines[18] = this.Name + ".set_taskType(" + this.taskType + ")";
        ExportStringLines[19] = this.Name + ".set_tdirection(" + this.tdirection + ")";
//        ExportStringLines[20] = this.Name + ".set_cyl_axis(" + this.cyl_axis.x + "," + this.cyl_axis.y + "," + this.cyl_axis.z + ")";
//        ExportStringLines[21] = this.Name + ".set_cyl_radius(" + this.cyl_radius + ")";
//        ExportStringLines[22] = this.Name + ".set_cyl_height(" + this.cyl_height + ")";
//        ExportStringLines[23] = this.Name + ".set_cyl_bottompoint_on_axis(" + this.cyl_bottompoint_on_axis.x + "," + this.cyl_bottompoint_on_axis.y + "," + this.cyl_bottompoint_on_axis.z + ")";
//        ExportStringLines[24] = this.Name + ".set_Displacement(" + this.Displacement.x + "," + this.Displacement.y + "," + this.Displacement.z + ")";




//        ExportStringLines[26] = this.Name + ".set_trace_eps(" + this.trace_eps + ")";
//        ExportStringLines[27] = this.Name + ".set_trace_abs(" + this.trace_abs + ")";
//        ExportStringLines[28] = this.Name + ".set_grid_tol(" + this.grid_tol + ")";

//        ExportStringLines[29] = this.Name + ".set_min_vel(" + this.min_vel + ")";

//        ExportStringLines[30] = this.Name + ".set_stepDuration(" + this.stepDuration + ")";
//        ExportStringLines[31] = this.Name + ".set_NoCycles(" + this.NoCycles + ")";
//        ExportStringLines[32] = this.Name + ".set_NoInterpolation(\"" + this.NoInterpolation + "\")";
//        ExportStringLines[33] = this.Name + ".set_ThrowNewParticles(\"" + this.ThrowNewParticles + "\")";
//        ExportStringLines[34] = this.Name + ".set_ParticlesReleaseRate(" + this.ParticlesReleaseRate + ")";
//        ExportStringLines[35] = this.Name + ".set_RandomOffset(\"" + this.RandomOffset + "\")";
//        ExportStringLines[36] = this.Name + ".set_RandomStartpoints(\"" + this.RandomStartpoints + "\")";
//        ExportStringLines[37] = this.Name + ".set_divideCell(" + this.divideCell + ")";

//        ExportStringLines[38] = this.Name + ".set_NoWThreads(" + this.NoWThreads + ")";
//        ExportStringLines[39] = this.Name + ".set_SearchLevel(" + this.SearchLevel + ")";
//        ExportStringLines[40] = this.Name + ".set_SkipInitialSteps(" + this.SkipInitialSteps + ")";


        return ExportStringLines;
    }

    public float getMaxOutOfDomain() {
        return maxOutOfDomain;
    }

    public void setMaxOutOfDomain(float maxOutOfDomain) {
        this.maxOutOfDomain = maxOutOfDomain;
    }

    public float getTrace_len() {
        return trace_len;
    }

    public void setTrace_len(float trace_len) {
        this.trace_len = trace_len;
    }

    public String getTdirection() {
        return tdirection;
    }

    public void setTdirection(String tdirection) {
        this.tdirection = tdirection;
    }
}


