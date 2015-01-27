/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: params.c,v 1.3 1994/08/20 04:00:31 pturner Exp pturner $
 *
 * write a parameter file
 *
 */

#include <stdio.h>
#include <string.h>
#include "globals.h"
#include "noxprotos.h"

void putparms(int gno, FILE *pp, int imbed)
{
    int i, j, k, ming, maxg;
    int ps, pt, gh, gl, gt, fx, fy, px, py;
    double dsx, dsy;
    char imbedstr[2], tmpstr1[128], tmpstr2[128];
    // defaults d;
    framep f;
    legend leg;
    labels lab;
    plotarr p;
    tickmarks t;
    world w;
    view v;

    if (imbed)
    {
        strcpy(imbedstr, "@");
    }
    else
    {
        imbedstr[0] = 0;
    }
    fprintf(pp, "# ACE/gr parameter file\n");
    fprintf(pp, "#\n");
    fprintf(pp, "#\n");
    fprintf(pp, "%spage %d\n", imbedstr, (int)(scrollper * 100));
    fprintf(pp, "%spage inout %d\n", imbedstr, (int)(shexper * 100));
    fprintf(pp, "%slink page %s\n", imbedstr, scrolling_islinked ? "on" : "off");

    fprintf(pp, "%sdefault linestyle %d\n", imbedstr, grdefaults.lines);
    fprintf(pp, "%sdefault linewidth %d\n", imbedstr, grdefaults.linew);
    fprintf(pp, "%sdefault color %d\n", imbedstr, grdefaults.color);
    fprintf(pp, "%sdefault char size %lf\n", imbedstr, grdefaults.charsize);
    fprintf(pp, "%sdefault font %d\n", imbedstr, grdefaults.font);
    fprintf(pp, "%sdefault font source %d\n", imbedstr, grdefaults.fontsrc);
    fprintf(pp, "%sdefault symbol size %lf\n", imbedstr, grdefaults.symsize);
    put_annotation(gno, pp, imbed);
    put_region(gno, pp, imbed);
    if (gno == -1)
    {
        maxg = maxgraph - 1;
        ming = 0;
    }
    else
    {
        maxg = gno;
        ming = gno;
    }
    for (k = ming; k <= maxg; k++)
    {
        if (isactive_graph(k))
        {
            gno = k;
            gh = g[gno].hidden;
            gl = g[gno].label;
            gt = g[gno].type;
            ps = g[gno].pointset;
            pt = g[gno].pt_type;
            dsx = g[gno].dsx;
            dsy = g[gno].dsy;
            fx = g[gno].fx;
            fy = g[gno].fy;
            px = g[gno].px;
            py = g[gno].py;

            fprintf(pp, "%swith g%1d\n", imbedstr, gno);

            fprintf(pp, "%sg%1d %s\n", imbedstr, gno, on_or_off(g[gno].active));
            fprintf(pp, "%sg%1d label %s\n", imbedstr, gno, on_or_off(gl));
            fprintf(pp, "%sg%1d hidden %s\n", imbedstr, gno, gh ? "true" : "false");
            fprintf(pp, "%sg%1d type %s\n", imbedstr, gno, graph_types(g[gno].type, 1));
            fprintf(pp, "%sg%1d autoscale type %s\n", imbedstr, gno, g[gno].auto_type == AUTO ? "AUTO" : "SPEC");
            fprintf(pp, "%sg%1d fixedpoint %s\n", imbedstr, gno, on_or_off(ps));
            fprintf(pp, "%sg%1d fixedpoint type %d\n", imbedstr, gno, pt);
            fprintf(pp, "%sg%1d fixedpoint xy %lf, %lf\n", imbedstr, gno, dsx, dsy);
            strcpy(tmpstr1, getFormat_types(fx));
            strcpy(tmpstr2, getFormat_types(fy));
            fprintf(pp, "%sg%1d fixedpoint format %s %s\n", imbedstr, gno, tmpstr1, tmpstr2);
            fprintf(pp, "%sg%1d fixedpoint prec %d, %d\n", imbedstr, gno, px, py);

            get_graph_world(gno, &w);
            fprintf(pp, "%s    world xmin %.12lg\n", imbedstr, w.xg1);
            fprintf(pp, "%s    world xmax %.12lg\n", imbedstr, w.xg2);
            fprintf(pp, "%s    world ymin %.12lg\n", imbedstr, w.yg1);
            fprintf(pp, "%s    world ymax %.12lg\n", imbedstr, w.yg2);

            for (i = 0; i < g[gno].ws_top; i++)
            {
                fprintf(pp, "%s    stack world %.9lg, %.9lg, %.9lg, %.9lg tick %lg, %lg, %lg, %lg\n", imbedstr,
                        g[gno].ws[i].w.xg1, g[gno].ws[i].w.xg2, g[gno].ws[i].w.yg1, g[gno].ws[i].w.yg2,
                        g[gno].ws[i].t[0].xg1, g[gno].ws[i].t[0].xg2, g[gno].ws[i].t[0].yg1, g[gno].ws[i].t[0].yg2);
            }

            get_graph_view(gno, &v);
            fprintf(pp, "%s    view xmin %lf\n", imbedstr, v.xv1);
            fprintf(pp, "%s    view xmax %lf\n", imbedstr, v.xv2);
            fprintf(pp, "%s    view ymin %lf\n", imbedstr, v.yv1);
            fprintf(pp, "%s    view ymax %lf\n", imbedstr, v.yv2);

            get_graph_labels(gno, &lab);
            fprintf(pp, "%s    title \"%s\"\n", imbedstr, lab.title.s);
            fprintf(pp, "%s    title font %d\n", imbedstr, lab.title.font);
            fprintf(pp, "%s    title size %lf\n", imbedstr, lab.title.charsize);
            fprintf(pp, "%s    title color %d\n", imbedstr, lab.title.color);
            fprintf(pp, "%s    title linewidth %d\n", imbedstr, lab.title.linew);
            fprintf(pp, "%s    subtitle \"%s\"\n", imbedstr, lab.stitle.s);
            fprintf(pp, "%s    subtitle font %d\n", imbedstr, lab.stitle.font);
            fprintf(pp, "%s    subtitle size %lf\n", imbedstr, lab.stitle.charsize);
            fprintf(pp, "%s    subtitle color %d\n", imbedstr, lab.stitle.color);
            fprintf(pp, "%s    subtitle linewidth %d\n", imbedstr, lab.title.linew);

            for (i = 0; i < g[gno].maxplot; i++)
            {
                get_graph_plotarr(gno, i, &p);
                if (isactive_set(gno, i))
                {
                    fprintf(pp, "%s    s%1d type %s\n", imbedstr, i, (char *)set_types(p.type));
                    fprintf(pp, "%s    s%1d symbol %d\n", imbedstr, i, p.sym);
                    fprintf(pp, "%s    s%1d symbol size %lf\n", imbedstr, i, p.symsize);
                    fprintf(pp, "%s    s%1d symbol fill %d\n", imbedstr, i, p.symfill);
                    fprintf(pp, "%s    s%1d symbol color %d\n", imbedstr, i, p.symcolor);
                    fprintf(pp, "%s    s%1d symbol linewidth %d\n", imbedstr, i, p.symlinew);
                    fprintf(pp, "%s    s%1d symbol linestyle %d\n", imbedstr, i, p.symlines);
                    fprintf(pp, "%s    s%1d symbol center %s\n", imbedstr, i, p.symdot ? "true" : "false");
                    fprintf(pp, "%s    s%1d symbol char %d\n", imbedstr, i, p.symchar);
                    fprintf(pp, "%s    s%1d skip %d\n", imbedstr, i, p.symskip);
                    fprintf(pp, "%s    s%1d linestyle %d\n", imbedstr, i, p.lines);
                    fprintf(pp, "%s    s%1d linewidth %d\n", imbedstr, i, p.linew);
                    fprintf(pp, "%s    s%1d color %d\n", imbedstr, i, p.color);
                    fprintf(pp, "%s    s%1d fill %d\n", imbedstr, i, p.fill);
                    fprintf(pp, "%s    s%1d fill with %s\n", imbedstr, i,
                            p.fillusing == COLOR ? "color" : "pattern");
                    fprintf(pp, "%s    s%1d fill color %d\n", imbedstr, i, p.fillcolor);
                    fprintf(pp, "%s    s%1d fill pattern %d\n", imbedstr, i, p.fillpattern);
                    switch (p.errbarxy)
                    {
                    case TOP:
                        fprintf(pp, "%s    s%1d errorbar type TOP\n", imbedstr, i);
                        break;
                    case BOTTOM:
                        fprintf(pp, "%s    s%1d errorbar type BOTTOM\n", imbedstr, i);
                        break;
                    case LEFT:
                        fprintf(pp, "%s    s%1d errorbar type LEFT\n", imbedstr, i);
                        break;
                    case RIGHT:
                        fprintf(pp, "%s    s%1d errorbar type RIGHT\n", imbedstr, i);
                        break;
                    case BOTH:
                        fprintf(pp, "%s    s%1d errorbar type BOTH\n", imbedstr, i);
                        break;
                    }
                    fprintf(pp, "%s    s%1d errorbar length %lf\n", imbedstr, i, p.errbarper);
                    fprintf(pp, "%s    s%1d errorbar linewidth %d\n", imbedstr, i, p.errbar_linew);
                    fprintf(pp, "%s    s%1d errorbar linestyle %d\n", imbedstr, i, p.errbar_lines);
                    fprintf(pp, "%s    s%1d errorbar riser %s\n", imbedstr, i, p.errbar_riser == ON ? "on" : "off");
                    fprintf(pp, "%s    s%1d errorbar riser linewidth %d\n", imbedstr, i, p.errbar_riser_linew);
                    fprintf(pp, "%s    s%1d errorbar riser linestyle %d\n", imbedstr, i, p.errbar_riser_lines);
                    fprintf(pp, "%s    s%1d xyz %lf, %lf\n", imbedstr, i, p.zmin, p.zmax);
                    if (is_hotlinked(gno, i))
                    {
                        fprintf(pp, "%s    s%1d link %s \"%s\"\n", imbedstr, i,
                                p.hotsrc == DISK ? "disk" : "pipe", p.hotfile);
                    }
                    fprintf(pp, "%s    s%1d comment \"%s\"\n", imbedstr, i, p.comments);
                }
            }

            for (i = 0; i < MAXAXES; i++)
            {
                switch (i)
                {
                case 0:
                    get_graph_tickmarks(gno, &t, X_AXIS);
                    if (t.active == OFF)
                    {
                        fprintf(pp, "%s    xaxis off\n", imbedstr);
                        continue;
                    }
                    sprintf(buf, "%s    xaxis ", imbedstr);
                    break;
                case 1:
                    get_graph_tickmarks(gno, &t, Y_AXIS);
                    if (t.active == OFF)
                    {
                        fprintf(pp, "%s    yaxis off\n", imbedstr);
                        continue;
                    }
                    sprintf(buf, "%s    yaxis ", imbedstr);
                    break;
                case 2:
                    get_graph_tickmarks(gno, &t, ZX_AXIS);
                    if (t.active == OFF)
                    {
                        fprintf(pp, "%s    zeroxaxis off\n", imbedstr);
                        continue;
                    }
                    sprintf(buf, "%s    zeroxaxis ", imbedstr);
                    break;
                case 3:
                    get_graph_tickmarks(gno, &t, ZY_AXIS);
                    if (t.active == OFF)
                    {
                        fprintf(pp, "%s    zeroyaxis off\n", imbedstr);
                        continue;
                    }
                    sprintf(buf, "%s    zeroyaxis ", imbedstr);
                    break;
                }

                fprintf(pp, "%s tick %s\n", buf, on_or_off(t.active));
                fprintf(pp, "%s tick major %.12lg\n", buf, t.tmajor);
                fprintf(pp, "%s tick minor %.12lg\n", buf, t.tminor);
                fprintf(pp, "%s tick offsetx %lf\n", buf, t.offsx);
                fprintf(pp, "%s tick offsety %lf\n", buf, t.offsy);
                /* DEFUNCT
                  fprintf(pp, "%s tick alt %s\n", buf, on_or_off(t.alt));
                  fprintf(pp, "%s tick min %.12lg\n", buf, t.tmin);
                  fprintf(pp, "%s tick max %.12lg\n", buf, t.tmax);
            */

                fprintf(pp, "%s label \"%s\"\n", buf, t.label.s);
                if (t.label_layout == PERP)
                {
                    fprintf(pp, "%s label layout perp\n", buf);
                }
                else
                {
                    fprintf(pp, "%s label layout para\n", buf);
                }
                if (t.label_place == AUTO)
                {
                    fprintf(pp, "%s label place auto\n", buf);
                }
                else
                {
                    fprintf(pp, "%s label place spec\n", buf);
                }
                fprintf(pp, "%s label char size %lf\n", buf, t.label.charsize);
                fprintf(pp, "%s label font %d\n", buf, t.label.font);
                fprintf(pp, "%s label color %d\n", buf, t.label.color);
                fprintf(pp, "%s label linewidth %d\n", buf, t.label.linew);

                fprintf(pp, "%s ticklabel %s\n", buf, on_or_off(t.tl_flag));
                if (t.tl_type == AUTO)
                {
                    fprintf(pp, "%s ticklabel type auto\n", buf);
                }
                else
                {
                    fprintf(pp, "%s ticklabel type spec\n", buf);
                }
                fprintf(pp, "%s ticklabel prec %d\n", buf, t.tl_prec);
                fprintf(pp, "%s ticklabel format %s\n", buf, getFormat_types(t.tl_format));
                fprintf(pp, "%s ticklabel append \"%s\"\n", buf, t.tl_appstr);
                fprintf(pp, "%s ticklabel prepend \"%s\"\n", buf, t.tl_prestr);
                switch (t.tl_layout)
                {
                case HORIZONTAL:
                    fprintf(pp, "%s ticklabel layout horizontal\n", buf);
                    break;
                case VERTICAL:
                    fprintf(pp, "%s ticklabel layout vertical\n", buf);
                    break;
                case SPEC:
                    fprintf(pp, "%s ticklabel layout spec\n", buf);
                    fprintf(pp, "%s ticklabel angle %d\n", buf, t.tl_angle);
                    break;
                }
                fprintf(pp, "%s ticklabel skip %d\n", buf, t.tl_skip);
                fprintf(pp, "%s ticklabel stagger %d\n", buf, t.tl_staggered);
                switch (t.tl_op)
                {
                case TOP:
                    fprintf(pp, "%s ticklabel op top\n", buf);
                    break;
                case BOTTOM:
                    fprintf(pp, "%s ticklabel op bottom\n", buf);
                    break;
                case LEFT:
                    fprintf(pp, "%s ticklabel op left\n", buf);
                    break;
                case RIGHT:
                    fprintf(pp, "%s ticklabel op right\n", buf);
                    break;
                case BOTH:
                    fprintf(pp, "%s ticklabel op both\n", buf);
                    break;
                }
                switch (t.tl_sign)
                {
                case NORMAL:
                    fprintf(pp, "%s ticklabel sign normal\n", buf);
                    break;
                case ABSOLUTE:
                    fprintf(pp, "%s ticklabel sign absolute\n", buf);
                    break;
                case NEGATE:
                    fprintf(pp, "%s ticklabel sign negate\n", buf);
                    break;
                }
                fprintf(pp, "%s ticklabel start type %s\n", buf, t.tl_starttype == AUTO ? "auto" : "spec");
                fprintf(pp, "%s ticklabel start %lf\n", buf, t.tl_start);
                fprintf(pp, "%s ticklabel stop type %s\n", buf, t.tl_stoptype == AUTO ? "auto" : "spec");
                fprintf(pp, "%s ticklabel stop %lf\n", buf, t.tl_stop);
                fprintf(pp, "%s ticklabel char size %lf\n", buf, t.tl_charsize);
                fprintf(pp, "%s ticklabel font %d\n", buf, t.tl_font);
                fprintf(pp, "%s ticklabel color %d\n", buf, t.tl_color);
                fprintf(pp, "%s ticklabel linewidth %d\n", buf, t.tl_linew);

                fprintf(pp, "%s tick major %s\n", buf, on_or_off(t.t_flag));
                fprintf(pp, "%s tick minor %s\n", buf, on_or_off(t.t_mflag));
                fprintf(pp, "%s tick default %d\n", buf, t.t_num);
                switch (t.t_inout)
                {
                case IN:
                    fprintf(pp, "%s tick in\n", buf);
                    break;
                case OUT:
                    fprintf(pp, "%s tick out\n", buf);
                    break;
                case BOTH:
                    fprintf(pp, "%s tick both\n", buf);
                    break;
                }
                fprintf(pp, "%s tick major color %d\n", buf, t.t_color);
                fprintf(pp, "%s tick major linewidth %d\n", buf, t.t_linew);
                fprintf(pp, "%s tick major linestyle %d\n", buf, t.t_lines);
                fprintf(pp, "%s tick minor color %d\n", buf, t.t_mcolor);
                fprintf(pp, "%s tick minor linewidth %d\n", buf, t.t_mlinew);
                fprintf(pp, "%s tick minor linestyle %d\n", buf, t.t_mlines);
                fprintf(pp, "%s tick log %s\n", buf, on_or_off(t.t_log));
                fprintf(pp, "%s tick size %lf\n", buf, t.t_size);
                fprintf(pp, "%s tick minor size %lf\n", buf, t.t_msize);
                fprintf(pp, "%s bar %s\n", buf, on_or_off(t.t_drawbar));
                fprintf(pp, "%s bar color %d\n", buf, t.t_drawbarcolor);
                fprintf(pp, "%s bar linestyle %d\n", buf, t.t_drawbarlines);
                fprintf(pp, "%s bar linewidth %d\n", buf, t.t_drawbarlinew);
                fprintf(pp, "%s tick major grid %s\n", buf, on_or_off(t.t_gridflag));
                fprintf(pp, "%s tick minor grid %s\n", buf, on_or_off(t.t_mgridflag));
                switch (t.t_op)
                {
                case TOP:
                    fprintf(pp, "%s tick op top\n", buf);
                    break;
                case BOTTOM:
                    fprintf(pp, "%s tick op bottom\n", buf);
                    break;
                case LEFT:
                    fprintf(pp, "%s tick op left\n", buf);
                    break;
                case RIGHT:
                    fprintf(pp, "%s tick op right\n", buf);
                    break;
                case BOTH:
                    fprintf(pp, "%s tick op both\n", buf);
                    break;
                }
                if (t.t_type == AUTO)
                {
                    fprintf(pp, "%s tick type auto\n", buf);
                }
                else
                {
                    fprintf(pp, "%s tick type spec\n", buf);
                }
                fprintf(pp, "%s tick spec %d\n", buf, t.t_spec);
                for (j = 0; j < t.t_spec; j++)
                {
                    fprintf(pp, "%s tick %d, %lg\n", buf, j, t.t_specloc[j]);
                    fprintf(pp, "%s ticklabel %d, \"%s\"\n", buf, j, t.t_speclab[j].s);
                }
            }

            get_graph_legend(gno, &leg);
            fprintf(pp, "%s    legend %s\n", imbedstr, on_or_off(leg.active));
            fprintf(pp, "%s    legend loctype %s\n", imbedstr, w_or_v(leg.loctype));
            fprintf(pp, "%s    legend layout %d\n", imbedstr, leg.layout);
            fprintf(pp, "%s    legend vgap %d\n", imbedstr, leg.vgap);
            fprintf(pp, "%s    legend hgap %d\n", imbedstr, leg.hgap);
            fprintf(pp, "%s    legend length %d\n", imbedstr, leg.len);
            fprintf(pp, "%s    legend box %s\n", imbedstr, on_or_off(leg.box));
            fprintf(pp, "%s    legend box fill %s\n", imbedstr, on_or_off(leg.box));
            fprintf(pp, "%s    legend box fill with %s\n", imbedstr, leg.boxfillusing == COLOR ? "color" : "pattern");
            fprintf(pp, "%s    legend box fill color %d\n", imbedstr, leg.boxfillcolor);
            fprintf(pp, "%s    legend box fill pattern %d\n", imbedstr, leg.boxfillpat);
            fprintf(pp, "%s    legend box color %d\n", imbedstr, leg.boxlcolor);
            fprintf(pp, "%s    legend box linewidth %d\n", imbedstr, leg.boxlinew);
            fprintf(pp, "%s    legend box linestyle %d\n", imbedstr, leg.boxlines);
            fprintf(pp, "%s    legend x1 %.12lg\n", imbedstr, leg.legx);
            fprintf(pp, "%s    legend y1 %.12lg\n", imbedstr, leg.legy);
            fprintf(pp, "%s    legend font %d\n", imbedstr, leg.font);
            fprintf(pp, "%s    legend char size %lf\n", imbedstr, leg.charsize);
            fprintf(pp, "%s    legend linestyle %d\n", imbedstr, leg.lines);
            fprintf(pp, "%s    legend linewidth %d\n", imbedstr, leg.linew);
            fprintf(pp, "%s    legend color %d\n", imbedstr, leg.color);
            for (i = 0; i < MAXPLOT; i++)
            {
                if (isactive_set(gno, i))
                {
                    if (strlen(leg.str[i].s))
                    {
                        fprintf(pp, "%s    legend string %d \"%s\"\n", imbedstr, i, leg.str[i].s);
                    }
                }
            }

            get_graph_framep(gno, &f);
            fprintf(pp, "%s    frame %s\n", imbedstr, on_or_off(f.active));
            fprintf(pp, "%s    frame type %d\n", imbedstr, f.type);
            fprintf(pp, "%s    frame linestyle %d\n", imbedstr, f.lines);
            fprintf(pp, "%s    frame linewidth %d\n", imbedstr, f.linew);
            fprintf(pp, "%s    frame color %d\n", imbedstr, f.color);
            fprintf(pp, "%s    frame fill %s\n", imbedstr, on_or_off(f.fillbg));
            fprintf(pp, "%s    frame background color %d\n", imbedstr, f.bgcolor);
        }
    }
}

void put_annotation(int, FILE *pp, int imbed)
{
    int i;
    boxtype b;
    linetype l;
    plotstr s;
    char imbedstr[2];

    if (imbed)
    {
        strcpy(imbedstr, "@");
    }
    else
    {
        imbedstr[0] = 0;
    }
    for (i = 0; i < MAXBOXES; i++)
    {
        get_graph_box(i, &b);
        if (b.active == ON)
        {
            fprintf(pp, "%swith box\n", imbedstr);
            fprintf(pp, "%s    box on\n", imbedstr);
            fprintf(pp, "%s    box loctype %s\n", imbedstr, w_or_v(b.loctype));
            if (b.loctype == WORLD)
            {
                fprintf(pp, "%s    box g%1d\n", imbedstr, b.gno);
            }
            fprintf(pp, "%s    box %.12lg, %.12lg, %.12lg, %.12lg\n", imbedstr, b.x1, b.y1, b.x2, b.y2);
            fprintf(pp, "%s    box linestyle %d\n", imbedstr, b.lines);
            fprintf(pp, "%s    box linewidth %d\n", imbedstr, b.linew);
            fprintf(pp, "%s    box color %d\n", imbedstr, b.color);
            switch (b.fill)
            {
            case PLOT_NONE:
                fprintf(pp, "%s    box fill none\n", imbedstr);
                break;
            case COLOR:
                fprintf(pp, "%s    box fill color\n", imbedstr);
                break;
            case PATTERN:
                fprintf(pp, "%s    box fill pattern\n", imbedstr);
                break;
            }
            fprintf(pp, "%s    box fill color %d\n", imbedstr, b.fillcolor);
            fprintf(pp, "%s    box fill pattern %d\n", imbedstr, b.fillpattern);
            fprintf(pp, "%sbox def\n", imbedstr);
        }
    }

    for (i = 0; i < MAXLINES; i++)
    {
        get_graph_line(i, &l);
        if (l.active == ON)
        {
            fprintf(pp, "%swith line\n", imbedstr);
            fprintf(pp, "%s    line on\n", imbedstr);
            fprintf(pp, "%s    line loctype %s\n", imbedstr, w_or_v(l.loctype));
            if (l.loctype == WORLD)
            {
                fprintf(pp, "%s    line g%1d\n", imbedstr, l.gno);
            }
            fprintf(pp, "%s    line %.12lg, %.12lg, %.12lg, %.12lg\n", imbedstr, l.x1, l.y1, l.x2, l.y2);
            fprintf(pp, "%s    line linewidth %d\n", imbedstr, l.linew);
            fprintf(pp, "%s    line linestyle %d\n", imbedstr, l.lines);
            fprintf(pp, "%s    line color %d\n", imbedstr, l.color);
            fprintf(pp, "%s    line arrow %d\n", imbedstr, l.arrow);
            fprintf(pp, "%s    line arrow size %lf\n", imbedstr, l.asize);
            fprintf(pp, "%s    line arrow type %d\n", imbedstr, l.atype);
            fprintf(pp, "%sline def\n", imbedstr);
        }
    }

    for (i = 0; i < MAXSTR; i++)
    {
        get_graph_string(i, &s);
        if (s.active == ON && s.s[0])
        {
            fprintf(pp, "%swith string\n", imbedstr);
            fprintf(pp, "%s    string on\n", imbedstr);
            fprintf(pp, "%s    string loctype %s\n", imbedstr, w_or_v(s.loctype));
            if (s.loctype == WORLD)
            {
                fprintf(pp, "%s    string g%1d\n", imbedstr, s.gno);
            }
            fprintf(pp, "%s    string %.12lg, %.12lg\n", imbedstr, s.x, s.y);
            fprintf(pp, "%s    string linewidth %d\n", imbedstr, s.linew);
            fprintf(pp, "%s    string color %d\n", imbedstr, s.color);
            fprintf(pp, "%s    string rot %d\n", imbedstr, s.rot);
            fprintf(pp, "%s    string font %d\n", imbedstr, s.font);
            fprintf(pp, "%s    string just %d\n", imbedstr, s.just);
            fprintf(pp, "%s    string char size %lf\n", imbedstr, s.charsize);
            fprintf(pp, "%sstring def \"%s\"\n", imbedstr, s.s);
        }
    }
}

void put_region(int, FILE *pp, int imbed)
{
    int i, j;
    char imbedstr[2];

    if (imbed)
    {
        strcpy(imbedstr, "@");
    }
    else
    {
        imbedstr[0] = 0;
    }
    for (i = 0; i < MAXREGION; i++)
    {
        if (rg[i].active == ON)
        {
            fprintf(pp, "%sr%1d ON\n", imbedstr, i);
            switch (rg[i].type)
            {
            case ABOVE:
                fprintf(pp, "%sr%1d type above\n", imbedstr, i);
                break;
            case BELOW:
                fprintf(pp, "%sr%1d type below\n", imbedstr, i);
                break;
            case LEFT:
                fprintf(pp, "%sr%1d type left\n", imbedstr, i);
                break;
            case RIGHT:
                fprintf(pp, "%sr%1d type right\n", imbedstr, i);
                break;
            case POLYI:
                fprintf(pp, "%sr%1d type polyi\n", imbedstr, i);
                break;
            case POLYO:
                fprintf(pp, "%sr%1d type polyo\n", imbedstr, i);
                break;
            }
            fprintf(pp, "%sr%1d linestyle %d\n", imbedstr, i, rg[i].lines);
            fprintf(pp, "%sr%1d linewidth %d\n", imbedstr, i, rg[i].linew);
            fprintf(pp, "%sr%1d color %d\n", imbedstr, i, rg[i].color);
            if (rg[i].type != POLYI && rg[i].type != POLYO)
            {
                fprintf(pp, "%sr%1d line %.12lg, %.12lg, %.12lg, %.12lg\n", imbedstr, i, rg[i].x1, rg[i].y1, rg[i].x2, rg[i].y2);
            }
            else
            {
                if (rg[i].x != NULL)
                {
                    for (j = 0; j < rg[i].n; j++)
                    {
                        fprintf(pp, "%sr%1d xy %.12lg, %.12lg\n", imbedstr, i, rg[i].x[j], rg[i].y[j]);
                    }
                }
            }
            for (j = 0; j < maxgraph; j++)
            {
                if (rg[i].linkto[j] == TRUE)
                {
                    fprintf(pp, "%slink r%1d to g%1d\n", imbedstr, i, j);
                }
            }
        }
    }
}

/*
 * read/write state functions
 */
void putbinary(int /* gno */, FILE * /*pp*/, int /*imbed*/)
{
}

void getbinary(int /* gno */, FILE * /*pp*/, int /*imbed*/)
{
}
