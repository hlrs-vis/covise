/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define SLIDER "\n DEF TS TimeSensor { loop FALSE\n\
         cycleInterval 4.0 }\n\n\
PROTO Slider [\n\
	eventIn SFFloat set_fraction\n\
	field SFFloat fraction .5\n\
	field SFBool noloop FALSE\n\
	eventOut SFFloat fraction_changed\n\
	eventOut SFTime touchTime\n\
	exposedField SFBool enabled TRUE\n\
	exposedField MFNode thumb Transform {\n\
				scale .08 .04 .08\n\
			children Shape {\n\
				appearance Appearance {\n\
					material Material {\n\
						diffuseColor 0 0 1\n\
					}\n\
				}\n\
				geometry Sphere {}\n\
				}\n\
			}\n\
	exposedField MFNode slide Shape {\n\
			appearance Appearance {\n\
				material Material {\n\
					diffuseColor .75 .75 .75\n\
					shininess .9\n\
					}\n\
				}\n\
			geometry Cylinder {\n\
				radius .01\n\
				height 1.1\n\
				}\n\
			}\n\
	] {\n"

#define SLIDER2 "	Group {\n\
	children [\n\
		DEF OUTPUT Script {\n\
			eventIn SFFloat	set_fraction IS set_fraction\n\
			field SFFloat fraction IS fraction\n\
			field SFBool noloop IS noloop\n\
			eventOut SFVec3f offset_changed\n\
			eventIn SFVec3f set_translation\n\
			eventOut SFFloat fraction_changed IS fraction_changed\n\
			eventIn SFBool is_Active\n\
			eventOut SFTime touchTime IS touchTime\n\
			url [  \"vrmlscript:\n\
			function initialize()\n\
			{\n\
			fraction_changed = fraction;\n\
			offset_changed[0] = fraction;\n\
			offset_changed[1] = 0;\n\
			offset_changed[2] = 0;\n\
			}\n\
\n\
			function set_fraction(value)\n\
			{\n\
			if (!(noloop)) fraction_changed = value;\n\
			fraction = value;\n\
			offset_changed[0] = value;\n\
			offset_changed[1] = 0;\n\
			offset_changed[2] = 0;\n\
			}\n\
\n\
			function set_translation(value)\n\
			{\n\
			fraction_changed = value[0];\n\
			fraction = value[0];\n\
			}\n\
\n\
			function is_Active(value, thetime)\n\
			{\n\
			if (value) touchTime = thetime;\n\
			}\n\
			\", \"widget/Slider.class\"]\n\
		}\n"

#define SLIDER3 "		Transform {\n\
			translation -.5 0 0\n\
		children [\n\
			DEF RAIL PlaneSensor {\n\
				enabled IS enabled\n\
				maxPosition 1 0\n\
				minPosition 0 0\n\
				offset .5 0 0\n\
				autoOffset TRUE\n\
			}\n\
			Transform {\n\
				rotation 0 0 1 1.5707963\n\
				translation .5 0 0\n\
				children IS slide\n\
			}\n\
			DEF BEAD Transform {\n\
				rotation 0 0 1 1.5707963\n\
				translation .5 0 0\n\
				children IS thumb\n\
			}\n\
		]\n\
		}\n\
	]\n\
	ROUTE RAIL.translation_changed TO BEAD.set_translation\n\
	ROUTE OUTPUT.offset_changed TO BEAD.set_translation\n\
	ROUTE OUTPUT.offset_changed TO RAIL.set_offset\n\
	ROUTE RAIL.translation_changed TO OUTPUT.set_translation\n\
	ROUTE RAIL.isActive TO OUTPUT.is_Active\n\
	}\n\
}\n"

#define SLIDER4 "PROTO Button [\n\
	field SFBool noloop FALSE\n\
	eventIn SFBool set_state\n\
	field SFBool state FALSE\n\
	eventOut SFBool state_changed\n\
	exposedField SFColor onColor 0.0 1.0 0.0\n\
	exposedField SFColor offColor 1.0 0.0 0.0\n\
	exposedField SFNode button Cylinder { height .0145 radius .015 }\n\
	exposedField MFNode base Transform {\n\
				rotation 1 0 0 1.5707963\n\
			children Shape {\n\
				appearance Appearance {\n\
					material Material { diffuseColor 0.0 0.0 1.0 }\n\
					}\n\
				geometry Cylinder { height .015 radius .02 }\n\
				}\n\
			}\n\
	] {\n\
	Group {\n\
	children [\n\
		Transform {\n\
			translation 0 0 -.005\n\
			children IS base\n\
		}\n\
		DEF INOUT Transform {\n\
			rotation 1 0 0 1.5707963\n\
			translation 0 0 .005\n\
		children [\n\
			DEF TOUCH TouchSensor {}\n\
			DEF COLOR Switch {\n\
				whichChoice 0\n\
			choice [\n\
				Shape {\n\
				appearance Appearance {\n\
				material Material { diffuseColor IS offColor }\n\
					}\n\
					geometry IS button\n\
				}\n\
				Shape {\n\
				appearance Appearance {\n\
					material Material {\n\
						diffuseColor IS onColor }\n\
					}\n\
					geometry IS button\n\
				}\n\
			]\n\
			}\n\
		]\n\
		}\n\
		DEF CONTROL Script {\n\
			field SFBool noloop IS noloop\n\
			eventIn SFBool isOver\n\
			eventIn SFBool isActive\n\
			eventIn SFBool set_state IS set_state\n\
			field SFBool onGeom FALSE\n\
			field SFBool bDown FALSE\n\
			field SFBool state IS state\n\
			eventOut SFBool state_changed IS state_changed\n\
			field SFVec3f push_pos 0 0 -.004\n\
			field SFVec3f in_pos 0 0 0\n\
			field SFVec3f out_pos 0 0 .005\n\
			eventOut SFVec3f pos_changed\n\
			eventOut SFInt32 choice_changed\n\
			url [ \"vrmlscript:\n\
			function initialize()\n\
			{\n\
			if (state) {\n\
				pos_changed = in_pos;\n\
				choice_changed = 1;\n\
			} else {\n\
				pos_changed = out_pos;\n\
				choice_changed = 0;\n\
				}\n\
			if (!(noloop)) state_changed = state;\n\
			}\n\
\n\
			function set_state(value)\n\
			{\n\
			state = value;\n\
			if (!(noloop)) state_changed = value;\n\
			if (value) {\n\
				pos_changed = in_pos;\n\
				choice_changed = 1;\n\
			} else {\n\
				pos_changed = out_pos;\n\
				choice_changed = 0;\n\
				}\n\
			}\n\
\n\
			function isOver(value)\n\
			{\n\
			onGeom = value;\n\
			if (value && bDown)\n\
				pos_changed = push_pos;\n\
			else if (state)\n\
				pos_changed = in_pos;\n\
			else\n\
				pos_changed = out_pos;\n\
			}\n\
\n\
			function isActive(value)\n\
			{\n\
			bDown = value;\n\
			if (value) pos_changed = push_pos;\n\
			else {\n\
				if (onGeom) {\n\
					state_changed = (!(state));\n\
					state = (!(state));\n\
					}\n\
				if (state) {\n\
					pos_changed = in_pos;\n\
					choice_changed = 1;\n\
				} else {\n\
					pos_changed = out_pos;\n\
					choice_changed = 0;\n\
					}\n\
				}\n\
			}\n\
			\", \"widget/Button.class\" ]\n\
		}\n\
	]\n\
	}\n" \
                "ROUTE TOUCH.isOver TO CONTROL.isOver\n\
ROUTE TOUCH.isActive TO CONTROL.isActive\n\
ROUTE CONTROL.choice_changed TO COLOR.set_whichChoice\n\
ROUTE CONTROL.pos_changed TO INOUT.set_translation\n\
}\n\
\n"

#define SLIDER5 "Transform {\n\
	rotation 0 0 1 3.1415\n\
#	translation 300 -150.0 0 \n\
	scale	3.0 3.0 3.0\n\
children DEF SLIDE1 Slider {\n\
		noloop TRUE\n\
	thumb DEF THUMB Transform {\n\
		children Shape {\n\
			appearance Appearance {\n\
				material Material { diffuseColor 0 0 1 }\n\
				}\n\
			geometry Cylinder {\n\
					height 1\n\
					radius 1\n\
				}\n\
			}\n\
		}\n\
	slide Shape {\n\
		appearance Appearance {\n\
			material Material {\n\
				diffuseColor .75 .75 .75\n\
				shininess .9\n\
				}\n\
			}\n\
		geometry Extrusion {\n\
			beginCap TRUE\n\
			endCap TRUE\n\
			ccw FALSE\n\
			convex TRUE\n\
			solid TRUE\n\
			creaseAngle 0.19635\n\
			crossSection [\n\
				1.00000 0.00000,\n\
				0.98079 0.19509,\n\
				0.92388 0.38268,\n\
				0.83147 0.55557,\n\
				0.70711 0.70711,\n\
				0.55557 0.83147,\n\
				0.38268 0.92388,\n\
				0.19509 0.98079,\n\
				0.00000 1.00000,\n\
				-0.19509 0.98079,\n\
				-0.38268 0.92388,\n\
				-0.55557 0.83147,\n\
				-0.70711 0.70711,\n\
				-0.83147 0.55557,\n\
				-0.92388 0.38268,\n\
				-0.98079 0.19509,\n\
				-1.00000 0.00000,\n\
				-0.98079 -0.19509,\n\
				-0.92388 -0.38268,\n\
				-0.83147 -0.55557,\n\
				-0.70711 -0.70711,\n\
				-0.55557 -0.83147,\n\
				-0.38268 -0.92388,\n\
				-0.19509 -0.98079,\n\
				0.00000 -1.00000,\n\
				0.19509 -0.98079,\n\
				0.38268 -0.92388,\n\
				0.55557 -0.83147,\n\
				0.70711 -0.70711,\n\
				0.83147 -0.55557,\n\
				0.92388 -0.38268,\n\
				0.98079 -0.19509,\n\
				1.00000 0.00000\n\
				]\n"
#define SLIDER6 "			spine [\n\
				0 .55 0,\n\
				0 .45 0,\n\
				0 .35 0,\n\
				0 .25 0,\n\
				0 .15 0,\n\
				0 .05 0,\n\
				0 -.05 0,\n\
				0 -.15 0,\n\
				0 -.25 0,\n\
				0 -.35 0,\n\
				0 -.45 0,\n\
				0 -.55 0\n\
				]\n\
			orientation []\n\
			scale [\n\
				.05 .05,\n\
				.032 .032,\n\
				.02 .02,\n\
				.013 .013,\n\
				.01 .01,\n\
				.005 .005,\n\
				.005 .005,\n\
				.01 .01,\n\
				.013 .013,\n\
				.02 .02,\n\
				.032 .032,\n\
				.05 .05\n\
				]\n\
			}\n\
		}\n\
	}\n\
}\n\
Transform {\n\
        #translation 300 -220 0\n\
        scale 10 10 10\n\
        children DEF BUTTON Button {\n\
                        button Cylinder { height .0145 radius .015 }\n\
                }\n\
}\n"

#define SLIDER7 "DEF SCALE ScalarInterpolator {\n\
	key [ 0, 1 ]\n\
	keyValue [ .1, 1.1 ]\n\
}\n\
\n\
DEF CONTROL1 Script {\n\
	eventIn SFFloat set_fraction\n\
	eventIn SFFloat set_scale\n\
	field SFFloat twist 6.2831853\n\
	field MFRotation baseO [\n\
		0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20,\n\
		0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20,\n\
		0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20,\n\
		0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20, 0 1 0 1e-20,\n\
		0 1 0 1e-20 ]\n\
	eventOut MFRotation value_changed\n\
	eventOut SFVec3f scale_changed\n\
	url [ \"vrmlscript:\n\
	function set_scale(value, ts)\n\
	{\n\
	val = value + .01;\n\
	scale_changed = new SFVec3f(val, .04, val);\n\
	}\n\
	function set_fraction(value, ts)\n\
	{\n\
	val = value * (twist / baseO.length);\n\
	addon = value * twist * (-.5);\n\
	addon = 0.0;\n\
	orient = baseO;\n\
	for (i=0;i<orient.length;i++) {\n\
		orient[i] = new\n\
			SFRotation(0, 1, 0, addon);\n\
		addon += val;\n\
		}\n\
	value_changed = orient;\n\
	}\n\
	\", \"widget/TwistSlider.class\" ]\n\
	\n\
\n"
#define SLIDER8 "#this script twists the extrusion at each spine point by the fraction\n\
#times the twist field\n\
}\n\
\n\
DEF INIT TimeSensor { loop TRUE cycleInterval .1 }\n\
DEF INITSCALE ScalarInterpolator {\n\
	key [0, 1]\n\
	keyValue [.5, .5]\n\
}\n\
\n\
DEF TISC ScalarInterpolator {\n\
	key [0, 1]\n\
	keyValue [0.0, 30]\n\
}\n\
ROUTE INIT.fraction_changed TO INITSCALE.set_fraction\n\
ROUTE INIT.cycleTime TO INIT.set_stopTime\n\
\n\
DEF TSCALE ScalarInterpolator {\n\
	key [	-.05, .05, .15, .25,\n\
		.35, .45, .55, .65,\n\
		.75, .85, .95, 1.05\n\
		]\n\
	keyValue [ .05, .032, .02, .013,\n\
		.01, .005, .005, .01,\n\
		.013, .02, .032, .05\n\
		]\n\
}\n\
\n\
ROUTE INITSCALE.value_changed TO SLIDE1.set_fraction\n\
ROUTE INITSCALE.value_changed TO SCALE.set_fraction\n\
ROUTE INITSCALE.value_changed TO TSCALE.set_fraction\n\
\n\
ROUTE SLIDE1.fraction_changed TO SCALE.set_fraction\n\
ROUTE SCALE.value_changed TO CONTROL1.set_fraction\n\
ROUTE SLIDE1.fraction_changed TO TSCALE.set_fraction\n\
ROUTE TSCALE.value_changed TO CONTROL1.set_scale\n\
ROUTE CONTROL1.scale_changed TO THUMB.set_scale\n\
\n\
\n\
DEF SCR Script {\n\
url\"vrmlscript: \n\
      function initialize() { \n\
        switchValue = 0; \n\
      } \n\
      function switchToNext(value, ts) { \n\
          switchValue = ((value-0.001) * (sizeOfSwitch)); \n\
      } \" \n\
eventIn SFFloat switchToNext \n\
eventOut SFInt32 switchValue\n"

#define ROUTES "ROUTE SLIDE1.fraction_changed TO SCR.switchToNext\n\
ROUTE TS.fraction_changed TO SCR.switchToNext\n\
ROUTE TS.fraction_changed TO TSCALE.set_fraction\n\
ROUTE TS.fraction_changed TO SLIDE1.set_fraction\n\
#ROUTE SCR.switchValue TO SW.set_whichChoice\n\
ROUTE BUTTON.state_changed TO TS.set_loop\n"
