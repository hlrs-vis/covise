// Create 2D-Plots

#include "RadialRunner.h"
#include <do/coDoData.h>
#include <General/include/log.h>

void RadialRunner::CreatePlot(void)
{
   char buf[500];
	int i, j, num_points, v_count;
	char plbuf[1000];
	float *xpl,*ypl;
	float xy_border[4];
	coDoVec2 *plot;

	for(i = 0; i < NUM_PLOT_PORTS; i++) {
		dprintf(2,"i = %d:\n",i);
		sprintf(buf,"XMGR%s_%d",M_2DPLOT,i+1);
#ifndef YAC
		char *PLOT_Name = Covise::get_object_name(buf);
		dprintf(2,"PLOT_Name: %s\n",PLOT_Name);
#else
                coObjInfo PLOT_Name = plot2d[i]->getNewObjectInfo(); // is this correct? ...
#endif
		dprintf(1,"plot port %2d: ",i+1);
		switch(m_2DplotChoice[i]->getValue()) {
			case 0:
				dprintf(1,"Nothing to plot\n");
				break;
			case 1:									 // meridian contour
				dprintf(1,"Meridian contour\n");
				GetMeridianContourNumbers(&num_points, xy_border, geo->rr,0);
				plot = new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetXMGRCommands(plbuf, xy_border, "Meridian contour",
								"Radius","Height", 0);
				plot->addAttribute("COMMANDS", plbuf);
				GetMeridianContourPlotData2(geo->rr, xpl, ypl, num_points, 0);
				delete plot;
				break;
			case 2:						 // meridian contour, extended
				dprintf(1,"Meridian contour (extended)\n");
				GetMeridianContourNumbers(&num_points, xy_border, geo->rr, 1);
				plot = new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetXMGRCommands(plbuf, xy_border, "Meridian contour",
								"Radius","Height", 0);
				plot->addAttribute("COMMANDS", plbuf);
				GetMeridianContourPlotData2(geo->rr, xpl, ypl, num_points, 1);
				delete plot;
				break;
			case 3:									 // conformal view
				// count number of cuts to show
				v_count = 0;
				for(j = 0; j < geo->rr->be_num; j++) {
					dprintf(1,"j: %d: p_ShowConformal = %d\n",j,p_ShowConformal[j][i]->getValue());
					if(p_ShowConformal[j][i]->getValue()) {
						v_count++;
					}
				}
				if(!v_count) {
					dprintf(1,"No blade elements for conformal view plot selected!\n");
				}
				else {
					dprintf(1,"Selected blade elements for conformal view plot: ");
					num_points = v_count * (3*2*(geo->rr->be[0]->cl->nump-1));
					plot = new coDoVec2(PLOT_Name, num_points);
					plot->getAddresses(&xpl, &ypl);
					v_count = 0;
					for(j = 0; j < geo->rr->be_num; j++) {
						if(p_ShowConformal[j][i]->getValue()) {
							dprintf(1," %d,",j+1);
							GetConformalViewPlotData(geo->rr, xpl, ypl, xy_border, j, v_count);
							v_count++;
						}
					}
					dprintf(1,"\b\n");
					GetXMGRCommands(plbuf, xy_border, "Conformal view",
									"SUM(s*dphi)","SUM(dl)", 1);
					plot->addAttribute("COMMANDS", plbuf);
					dprintf(2,"plbuf = %s\n",plbuf);
					delete plot;
				}
				break;
			case 4:									 // camber
				// count number of cuts to show
				v_count = 0;
				for(j = 0; j < geo->rr->be_num; j++) {
					dprintf(1,"j: %d: p_ShowCamber = %d\n",j,p_ShowCamber[j][i]->getValue());
					if(p_ShowCamber[j][i]->getValue()) {
						v_count++;
					}
				}
				if(!v_count) {
					dprintf(1,"No blade elements for camber plot selected!\n");
				}
				else {
					dprintf(1,"Selected blade elements for camber plot: ");
					num_points = v_count * (2*(geo->rr->be[0]->cl->nump-2));
					plot = new coDoVec2(PLOT_Name, num_points);
					plot->getAddresses(&xpl, &ypl);
					v_count = 0;
					for(j = 0; j < geo->rr->be_num; j++) {
						if(p_ShowCamber[j][i]->getValue()) {
							dprintf(1," %d,",j+1);
							GetCamberPlotData(geo->rr, xpl, ypl, 
											  xy_border, j, v_count);
							v_count++;
						}
					}
					dprintf(1,"\b\n");
					GetXMGRCommands(plbuf, xy_border, "center line tangent angles",
									"normalized cl-length","angle[deg]", 0);
					plot->addAttribute("COMMANDS", plbuf);
					dprintf(2,"plbuf = %s\n",plbuf);
					for(j = 0; j < num_points; j++) {
						dprintf(2," %16.8f	  %16.8f\n",xpl[j], ypl[j]);
					}
					delete plot;
				}
				break;
			case 5:									 // normalized camber
				// count number of cuts to show
				v_count = 0;
				for(j = 0; j < geo->rr->be_num; j++) {
					if(p_ShowNormCamber[j][i]->getValue()) {
						v_count++;
					}
				}
				if(!v_count) {
					dprintf(1,"No blade elements for camber plot selected!\n");
				}
				else {
					dprintf(1,"Selected blade elements for camber plot: ");
					num_points = v_count * (2*(geo->rr->be[0]->cl->nump-2));
					plot = new coDoVec2(PLOT_Name, num_points);
					plot->getAddresses(&xpl, &ypl);
					v_count = 0;
					for(j = 0; j < geo->rr->be_num; j++) {
						if(p_ShowNormCamber[j][i]->getValue()) {
							dprintf(1," %d,",j+1);
							GetNormalizedCamber(geo->rr, xpl, ypl, xy_border, j, v_count);
							v_count++;
						}
					}
					dprintf(1,"\b\n");
					GetXMGRCommands(plbuf, xy_border, "center line tangent angles",
									"normalized cl-length","angle[deg]", 0);
					plot->addAttribute("COMMANDS", plbuf);
					delete plot;
				}
				break;
			case 6:									 // thickness distribution
				dprintf(2,"Thickness distribution\n");
				num_points = 4*(geo->rr->be_num-1)+2;
				dprintf(2,"num_points = %d\n",num_points);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetMaxThicknessData(geo->rr, xpl, ypl, xy_border);
				GetXMGRCommands(plbuf, xy_border, "Max. thickness distribution",
								"parameter (hub (0) -> shroud (1))", "max. thickness [m]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			case 7:									 // overlap
				dprintf(1,"Overlap ratio\n");
				num_points = 2*(geo->rr->be_num-1);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetOverlapPlotData(geo->rr, xpl, ypl, xy_border);
				GetXMGRCommands(plbuf, xy_border, "Overlap ratio",
								"parameter (hub (0) -> shroud (1))", "overlap ratio [%]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			case 8:									 // blade angles, from conformal view
				dprintf(1,"Blade angles\n");
				num_points = 4*(geo->rr->be_num-1);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetBladeAnglesPlotData(geo->rr, xpl, ypl, xy_border);
				GetXMGRCommands(plbuf, xy_border, "Blade angles, real",
								"parameter (hub (0) -> shroud (1))", "angle [deg]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			case 9:								 // blade angles, from euler
				dprintf(1,"Blade angles, euler\n");
				num_points = 4*(geo->rr->be_num-1);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetEulerAnglesPlotData(geo->rr, xpl, ypl, xy_border);
				GetXMGRCommands(plbuf, xy_border, "Blade angles, Euler",
								"parameter (hub (0) -> shroud (1))", "angle [deg]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			case 10:								 // meridian velocities
				dprintf(1,"Meridian velocities\n");
				num_points = 4*(geo->rr->be_num-1);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetMeridianVelocityPlotData(geo->rr, xpl, ypl, xy_border);
				GetXMGRCommands(plbuf, xy_border, "Meridian velocities",
								"parameter (hub (0) -> shroud (1))", "velocity [m/s]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			case 11:								 // circumf velocities
				dprintf(1,"Circumferential velocities\n");
				num_points = 4*(geo->rr->be_num-1);
				plot =	new coDoVec2(PLOT_Name, num_points);
				plot->getAddresses(&xpl, &ypl);
				GetCircumferentialVelocityPlotData(geo->rr,xpl,ypl,xy_border);
				GetXMGRCommands(plbuf, xy_border, "Circumferential velocities",
								"parameter (hub (0) -> shroud (1))", "velocity [m/s]",0);
				plot->addAttribute("COMMANDS", plbuf);
				delete plot;
				break;
			default: dprintf(0,"Sorry, unrecognized plot menu option %d!\n",
							 m_2DplotChoice[i]->getValue());
				break;
		}											// end switch(m_2DplotChoice[i])

		// dump it!
		dprintf(2,"i = %d done!\n",i);
	}											   // end i
}
