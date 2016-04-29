// Create 2D-Plots

#include "AxialRunner.h"
#include <General/include/log.h>
#include <do/coDoData.h>

void AxialRunner::CreatePlot(void)
{
   char buf[500];
   int i, j, num_points, v_count;
   char *PLOT_Name;
   char plbuf[1000];
   float *xpl,*ypl;
   float xy_border[4];
   coDoVec2 *plot;

   dprintf(5,"CreatePlot() ...\n");
   for(i = 0; i < NUM_PLOT_PORTS; i++)
   {
      dprintf(5,"CreatePlot(): i = %d:\n",i);
      sprintf(buf,"XMGR%s_%d",M_2DPLOT,i+1);
      if(!(PLOT_Name = Covise::get_object_name(buf)))
      {
         sendError("No plot port named '%s' available!",buf);
         continue;
      }
      dprintf(5,"PLOT_Name: %s\n",PLOT_Name);
      dprintf(1,"plot port %2d: ",i+1);
      switch(m_2DplotChoice[i]->getValue())
      {
         case 0:
            dprintf(1,"Nothing to plot\n");
            break;
         case 1:                                  // meridian contour
            dprintf(1,"Meridian contour\n");
            GetMeridianContourNumbers(&num_points, xy_border,
               geo->ar,0);
            plot = new coDoVec2(PLOT_Name,
               num_points);
            plot->getAddresses(&xpl, &ypl);
            GetXMGRCommands(plbuf, xy_border, "Meridian contour",
               "Radius","Height", 0);
            plot->addAttribute("COMMANDS", plbuf);
            GetMeridianContourPlotData(geo->ar, xpl, ypl, 0);
            delete plot;
            break;
         case 2:                                  // conformal view
            // count number of cuts to show
            v_count = 0;
            for(j = 0; j < geo->ar->be_num; j++)
            {
               dprintf(1,"j: %d: p_ShowConformal = %d\n",
                  j,p_ShowConformal[j][i]->getValue());
               if(p_ShowConformal[j][i]->getValue())
               {
                  v_count++;
               }
            }
            if(!v_count)
            {
               dprintf(1,"No blade elems for conformal view plot selected!\n");
            }
            else
            {
               dprintf(1,"Selected blade elems for conformal view plot: ");
               num_points = v_count * (3 * 2*(geo->ar->me[0]->cl->nump-1));
               plot = new coDoVec2(PLOT_Name, num_points);
               plot->getAddresses(&xpl, &ypl);
               v_count = 0;
               for(j = 0; j < geo->ar->be_num; j++)
               {
                  if(p_ShowConformal[j][i]->getValue())
                  {
                     dprintf(1," %d,",j+1);
                     GetConformalViewPlotData(geo->ar,xpl,ypl,
                        xy_border,j,v_count);
                     v_count++;
                  }
               }
               dprintf(1,"\b\n");
               GetXMGRCommands(plbuf, xy_border, "Conformal view",
                  "SUM(s*dphi)","SUM(dl)", 1);
               plot->addAttribute("COMMANDS", plbuf);
               delete plot;
            }
            break;
         case 3:                                  // camber
            // count number of cuts to show
            v_count = 0;
            for(j = 0; j < geo->ar->be_num; j++)
            {
               dprintf(1,"j: %d: p_ShowCamber = %d\n",
                  j,p_ShowCamber[j][i]->getValue());
               if(p_ShowCamber[j][i]->getValue())
               {
                  v_count++;
               }
            }
            if(!v_count)
            {
               dprintf(1,"No blade elements for camber plot selected!\n");
            }
            else
            {
               fprintf(stderr,"Selected blade elements for camber plot: ");
               num_points = v_count * (2*(geo->ar->me[0]->cl->nump-2));
               plot = new coDoVec2(PLOT_Name, num_points);
               plot->getAddresses(&xpl, &ypl);
               v_count = 0;
               for(j = 0; j < geo->ar->be_num; j++)
               {
                  if(p_ShowCamber[j][i]->getValue())
                  {
                     dprintf(3," %d,",j+1);
                     GetCamberPlotData(geo->ar, xpl, ypl, xy_border, j, v_count);
                     v_count++;
                  }
               }
               dprintf(3,"\b\n");
               GetXMGRCommands(plbuf, xy_border, "center line tangent angles",
                  "normalized cl-length","angle[deg]", 0);
               plot->addAttribute("COMMANDS", plbuf);
               delete plot;
            }
            break;
         case 4:                                  // normalized camber
            // count number of cuts to show
            v_count = 0;
            for(j = 0; j < geo->ar->be_num; j++)
            {
               dprintf(1,"j: %d: p_ShowNormCamber = %d\n",
                  j,p_ShowNormCamber[j][i]->getValue());
               if(p_ShowNormCamber[j][i]->getValue())
               {
                  v_count++;
               }
            }
            if(!v_count)
            {
               dprintf(1,"No blade elements for camber plot selected!\n");
            }
            else
            {
               fprintf(stderr,"Selected blade elements for camber plot: ");
               num_points = v_count * (2*(geo->ar->me[0]->cl->nump-2));
               plot = new coDoVec2(PLOT_Name, num_points);
               plot->getAddresses(&xpl, &ypl);
               v_count = 0;
               for(j = 0; j < geo->ar->be_num; j++)
               {
                  if(p_ShowNormCamber[j][i]->getValue())
                  {
                     dprintf(3," %d,",j+1);
                     GetNormalizedCamber(geo->ar, xpl, ypl, xy_border, j, v_count);
                     v_count++;
                  }
               }
               dprintf(3,"\b\n");
               GetXMGRCommands(plbuf, xy_border, "normalized tangent angles",
                  "normalized cl-length","angle[deg]", 0);
               plot->addAttribute("COMMANDS", plbuf);
               delete plot;
            }
            break;
         case 5:                                  // thickness distribution
            dprintf(1,"Thickness distribution\n");
            num_points = 4*(geo->ar->be_num-1)+2;
            dprintf(5,"num_points = %d\n",num_points);
            plot =   new coDoVec2(PLOT_Name, num_points);
            plot->getAddresses(&xpl, &ypl);
            GetMaxThicknessData(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border, "Max. thickness distribution",
               "parameter (hub (0) -> shroud (1))", "max. thickness [m]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 6:                                  // max. thickness distrib.
            dprintf(1,"Max. thickness distribution\n");
            num_points = 2*(geo->ar->be_num-1);
            dprintf(5,"num_points = %d\n",num_points);
            plot =   new coDoVec2(PLOT_Name, num_points);
            plot->getAddresses(&xpl, &ypl);
            GetMaxThicknessDistrib(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border, "Max. thickness distribution",
               "parameter (hub (0) -> shroud (1))", "cl-param.",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 7:                                  // overlap
            dprintf(1,"Overlap ratio\n");
            num_points = 2*(geo->ar->be_num-1);
            plot =   new coDoVec2(PLOT_Name, num_points);
            plot->getAddresses(&xpl, &ypl);
            GetOverlapPlotData(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border, "Overlap ratio",
               "parameter (hub (0) -> shroud (1))", "overlap ratio [%]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 8:                                  // blade angles, from conformal view
            dprintf(1,"Blade angles\n");
            num_points = 4*(geo->ar->be_num-1);
            plot =   new coDoVec2(PLOT_Name, num_points);
            plot->getAddresses(&xpl, &ypl);
            GetBladeAnglesPlotData(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border, "Blade angles, real",
               "parameter (hub (0) -> shroud (1))", "angle [deg]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 9:                                 // blade angles, from euler
            fprintf(stderr,"Blade angles, euler\n");
            num_points = 4*(geo->ar->be_num-1);
            plot =   new coDoVec2(PLOT_Name,
               num_points);
            plot->getAddresses(&xpl, &ypl);
            GetEulerAnglesPlotData(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border,
               "Blade angles, Euler",
               "parameter (hub (0) -> shroud (1))",
               "angle [deg]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 10:                                 // chord angle
            num_points = 2*(geo->ar->be_num-1);
            plot =   new coDoVec2(PLOT_Name,
               num_points);
            plot->getAddresses(&xpl, &ypl);
            GetChordAnglesPlotData(geo->ar, xpl, ypl, xy_border);
            GetXMGRCommands(plbuf, xy_border,
               "Chord angle",
               "parameter (hub (0) -> shroud (1))",
               "angle [deg]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
         case 11:                                 // parameter values from sliders
            num_points = 2*(geo->ar->be_num-1);
            plot =   new coDoVec2(PLOT_Name,
               num_points);
            plot->getAddresses(&xpl, &ypl);
            GetParamPlotData(geo->ar, xpl, ypl, xy_border,
               m_sliderChoice->getValue() + 1);
            GetXMGRCommands(plbuf, xy_border,
               "Parameter value",
               "parameter (hub (0) -> shroud (1))",
               "angle [deg]",0);
            plot->addAttribute("COMMANDS", plbuf);
            delete plot;
            break;
          default:
            dprintf(1,"Sorry, unrecognized plot menu option %d!\n",
               m_2DplotChoice[i]->getValue());
            break;
      }                                           // end switch(m_2DplotChoice[i])

   }                                              // end i
   dprintf(5,"CreatePlot() ... done!\n");
}
