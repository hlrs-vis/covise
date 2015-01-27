/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: xprotos.h,v 1.10 1994/10/30 07:26:14 pturner Exp pturner $
 *
 * Prototypes involving X
 *
 */
#include <ctype.h>

void set_axis_proc(Widget w, XtPointer client_data, XtPointer call_data);
void page_special_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
void accept_symmisc(Widget, XtPointer, XtPointer);
void accept_ledit_proc(Widget w, XtPointer client_data, XtPointer call_data);
void define_boxplot_proc(Widget, XtPointer, XtPointer);
void setall_colors_proc(Widget w, XtPointer client_data, XtPointer call_data);
void setall_sym_proc(Widget w, XtPointer client_data, XtPointer call_data);
void setall_linew_proc(Widget w, XtPointer client_data, XtPointer call_data);
void set_cset_proc(Widget w, XtPointer client_data, XtPointer call_data);
void set_page(Widget w, XtPointer client_data, XtPointer call_data);
void set_actioncb(Widget w, XtPointer client_data, XtPointer call_data);
void bailout(void);
void autoscale_proc(Widget w, XtPointer client_data, XtPointer call_data);
void autoticks_proc(Widget w, XtPointer client_data, XtPointer call_data);
void set_left_footer(const char *s);
void do_main_loop(void);

void drawgraph(void);
void doforce_redraw(void);
void do_hardcopy(void);
void do_zoom(Widget w, XtPointer client_data, XtPointer call_data);
void do_zoomx(Widget w, XtPointer client_data, XtPointer call_data);
void do_zoomy(Widget w, XtPointer client_data, XtPointer call_data);
void do_select_point(Widget w, XtPointer client_data, XtPointer call_data);
void do_clear_point(Widget w, XtPointer client_data, XtPointer call_data);
void refresh(Widget w, XtPointer client_data, XmDrawingAreaCallbackStruct *cbs);
void setpointer(int x, int y);
void getpoints(int x, int y);
void set_stack_message(void);
void select_line(int x1, int y1, int x2, int y2);
void draw_rectangle(int x1, int y1, int x2, int y2);
void select_region(int x1, int y1, int x2, int y2);
void draw_focus(int gno);
void update_text_cursor(char *s, int x, int y);
void set_action(int act);
void do_text_string(int op, int c);
void my_proc(Widget w, caddr_t data, XEvent *event);
void do_select_area(void);
void do_select_peri(void);
void do_select_region(void);
void setbgcolor(int c);
void motion(XMotionEvent *e);
int double_click(XButtonEvent *e);
void switch_current_graph(int gfrom, int gto);

void yesnoCB(Widget w, Boolean *keep_grab, XmAnyCallbackStruct *reason);
int yesno(const char *msg1, const char *msg2, const char *s1, const char *s2);
void errwin(const char *s);

void create_file_popup(Widget wid, XtPointer client_data, XtPointer call_data);
void create_netcdfs_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_rparams_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_wparam_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_block_popup(Widget w, XtPointer client_data, XtPointer call_data);

void create_eblock_frame(Widget w, XtPointer client_data, XtPointer call_data);

void set_printer(int device, char *prstr);
void do_prstr_toggle(Widget w, XtPointer client_data, XtPointer call_data);
void create_printer_setup(Widget w, XtPointer client_data, XtPointer call_data);

void update_draw(void);
void create_draw_frame(Widget w, XtPointer client_data, XtPointer call_data);

void open_command(Widget w, XtPointer client_data, XtPointer call_data);
void close_rhist_popup(Widget w, XtPointer client_data, XtPointer call_data);
void do_rhist_proc(Widget w, XtPointer client_data, XtPointer call_data);
void create_rhist_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_whist_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_comp_frame(Widget w, XtPointer client_data, XtPointer call_data);
void do_pick_compose(Widget w, XtPointer client_data, XtPointer call_data);
void create_eval_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_load_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_histo_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_fourier_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_run_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_reg_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_diff_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_seasonal_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_interp_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_int_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_xcor_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_spline_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_samp_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_digf_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_lconv_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_leval_frame(Widget w, XtPointer client_data, XtPointer call_data);
void execute_pick_compute(int gno, int setno, int function);

void define_setops_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_write_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_hotlinks_popup(Widget w, XtPointer client_data, XtPointer call_data);
void update_hotlinks(void);
void create_saveall_popup(Widget w, XtPointer client_data, XtPointer call_data);

void SetLabel(Widget w, char *buf);
void create_points_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_goto_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_add_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_editp_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_region_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_define_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_clear_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_extract_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_delete_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_extractsets_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_deletesets_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_evalregion_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_area_frame(Widget w, XtPointer client_data, XtPointer call_data);

void update_status(int gno, int itemtype, int itemno);
void update_graph_status(int gno);
void update_region_status(int rno);
void update_set_status(int gno, int setno);
void clear_status(void);
void update_status_popup(Widget w, XtPointer client_data, XtPointer call_data);
void select_set(Widget w, XtPointer calld, XEvent *e);
void define_status_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_about_grtool(Widget w, XtPointer client_data, XtPointer call_data);

void updatesymbols(int gno, int value);
void updatelegendstr(int gno);
void define_symbols_popup(Widget w, XtPointer client_data, XtPointer call_data);
void updatelegends(int gno);
void define_legends_proc(Widget w, XtPointer client_data, XtPointer call_data);
void legend_loc_proc(Widget w, XtPointer client_data, XtPointer call_data);
void legend_load_proc(Widget w, XtPointer client_data, XtPointer call_data);
void define_legend_popup(Widget w, XtPointer client_data, XtPointer call_data);
void updateerrbar(int gno, int value);

void update_ticks(int gno);
void update_ticks_items(int gno);
void create_ticks_frame(Widget w, XtPointer client_data, XtPointer call_data);

void update_graph_items(void);
void create_graph_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_image_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_gtype_frame(Widget w, XtPointer client_data, XtPointer call_data);

void update_world(int gno);
void update_world_proc(void);
void create_world_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_view(int gno);
double fround(double x, double r);
void create_view_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_arrange(void);
void create_arrange_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_overlay_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_autos(int gno);
void create_autos_frame(Widget w, XtPointer client_data, XtPointer call_data);

void boxes_def_proc(Widget w, XtPointer client_data, XtPointer call_data);
void lines_def_proc(Widget w, XtPointer client_data, XtPointer call_data);
void updatestrings(void);
void update_lines(void);
void update_boxes(void);
void define_string_defaults(Widget w, XtPointer client_data, XtPointer call_data);
void define_objects_popup(Widget w, XtPointer client_data, XtPointer call_data);
void define_strings_popup(Widget w, XtPointer client_data, XtPointer call_data);
void define_lines_popup(Widget w, XtPointer client_data, XtPointer call_data);
void define_boxes_popup(Widget w, XtPointer client_data, XtPointer call_data);
void do_object_function(Widget w, XtPointer client_data, XtPointer call_data);

void update_label_proc(void);
void create_label_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_labelprops_proc(void);

void update_locator_items(int gno);
void create_locator_frame(Widget w, XtPointer client_data, XtPointer call_data);

void update_frame_items(int gno);
void create_frame_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_monitor_frame(Widget w, XtPointer client_data, XtPointer call_data);
void stufftext(char *s, int sp);

void create_help_frame(Widget w, XtPointer client_data, XtPointer call_data);

void create_nonl_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_nonl(void);

void create_page_frame(Widget w, XtPointer client_data, XtPointer call_data);
void update_misc_items(void);
void create_misc_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_props_frame(Widget w, XtPointer client_data, XtPointer call_data);
void SetChoice(Widget *w, int value);
int GetChoice(Widget *w);
Widget *CreatePanelChoice(Widget parent, const char *labstr, int nchoices, ...);
Widget *CreatePanelChoice0(Widget parent, const char *labstr, int ncols, int nchoices, ...);
Widget *CreateGraphChoice(Widget parent, const char *labelstr, int ngraphs, int type);
Widget *CreateColorChoice(Widget parent, const char *s, int map);
Widget CreateTextItem2(Widget parent, int len, const char *s);
Widget CreateTextItem4(Widget parent, int len, const char *s);

char *xv_getstr(Widget w);
void xv_setstr(Widget w, const char *s);
void XmGetPos(Widget w, int type, int *x, int *y);
void GetWH(Widget w, Dimension *ww, Dimension *wh);
void GetXY(Widget w, Position *x, Position *y);
void XtFlush(void);
void destroy_dialog(Widget w, XtPointer p);
void handle_close(Widget w);
void XtRaise(Widget w);
Widget *CreateSetChoice(Widget parent, const char *labelstr, int nsets, int type);
void CreateCommandButtons(Widget parent, int n, Widget *buts, char **l);

int save_image_on_disk(Display *disp, Window info_win_id, Pixmap win_id, int x, int y, int width, int height, char *name_of_file, Colormap the_colormap);
XImage *read_image_from_disk(Display *disp, Window win_id, char *name_of_file, int *width, int *height, int *depth);

void get_xlib_dims(int *w, int *h);
void xlibdoublebuffer(int mode);
void xlibfrontbuffer(int mode);
void xlibbackbuffer(int mode);
void xlibswapbuffer(void);
void refresh_from_backpix(void);
void resize_backpix(void);
void set_write_mode(int m);
void xlibsetmode(int mode);
void drawxlib(int x, int y, int mode);
int xconvxlib(double x);
int yconvxlib(double y);
void initialize_cms_data(void);
void xlibinitcmap(void);
void xlibsetcmap(int i, int r, int g, int b);
int xlibsetlinewidth(int c);
int xlibsetlinestyle(int style);
int xlibsetcolor(int c);
void xlibdrawtic(int x, int y, int dir, int updown);
void xlibsetfont(int n);
void xlibsetfont2(char *typeface, int point, int weight, int angle);
void dispstrxlib(int x, int y, int rot, char *s, int just, int fudge);
void xlibinit_tiles(void);
int xlibsetpat(int k);
void xlibfill(int n, int *px, int *py);
void xlibfillcolor(int n, int *px, int *py);
void xlibdrawarc(int x, int y, int r);
void xlibfillarc(int x, int y, int r);
void xlibdrawellipse(int x, int y, int xm, int ym);
void xlibfillellipse(int x, int y, int xm, int ym);
void xlibleavegraphics(void);
int xlibinitgraphics(int dmode);
void set_wait_cursor(void);
void unset_wait_cursor(void);
void set_cursor(int c);
void set_window_cursor(Window xwin, int c);
void init_cursors(void);
