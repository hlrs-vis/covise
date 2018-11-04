%{

/* $Id: pars.yacc,v 1.9 1994/10/09 04:42:11 pturner Exp pturner $
 * 
 * evaluate expressions, commands, parameter files
 * 
 */

#define PARS			/* to overide some defines in defines.h */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <sys/param.h>

#include "globals.h"

#include "extern.h"
// #include <Xm/Xm.h>
// #include "motifinc.h"
// #include "extern2.h"
void set_plotstr_string(plotstr *pstr, char *buf);

#ifndef M_PI
#     define M_PI  3.14159265358979323846
#endif

#ifndef TRUE
#     define TRUE 1
#endif

#ifndef FALSE
#     define FALSE 0
#endif

double result, resx, resy;	/* return value if expression */
double nonl_parms[10];

double drand48(void);
long lrand48(void);
double erf(double arg); /* doesn't seem to be in ANSI C */
double erfc(double arg); /* doesn't seem to be in ANSI C */
double rnorm(double mean, double sdev);
double fx(double x);
double normp(double b, double *s);
double invnorm(double p);
double invt(double p, int n);
double *getvptr(int gno, int setno, int v);
double vmin(double *x, int n);
double vmax(double *x, int n);
void yyerror(const char *s);
#define __my_yylex
int yylex(void);

static int interr;

static double *freelist[100]; 	/* temporary vectors */
static int fcnt;		/* number allocated */

int naxis = 0;	/* current axis */
int curline, curbox, curstring, curleg, curgrid;

int gotbatch, gotparams, gotread; /* these guys attempt to avoid reentrancy problems */
int readtype, readsrc, readxformat;
char batchfile[256], paramfile[256], readfile[256];

static char f_string[512];	/* buffer for string to parse */
static int pos = 0;
static double *aa, *bb, *cc, *dd, *xx, *yy;
static int setindex, lxy, ls;
static int setsetno;
static int whichgraph;
static int whichset;

extern int change_gno;
extern int change_type;

%}

%union {
    double val;
    long ival;
    double *ptr;
    long func;
    long pset;
    char *str;
}

%token <func> ABS 
%token <func> ACOS
%token <func> ASIN
%token <func> ATAN
%token <func> ATAN2
%token <func> CEIL 
%token <func> COS
%token <func> DEG
%token <func> DX
%token <func> DY
%token <func> ERF
%token <func> ERFC
%token <func> EXP
%token <func> FLOOR 
%token <func> HYPOT
%token <func> INDEX
%token <func> INT
%token <func> INVN
%token <func> INVT
%token <func> IRAND
%token <func> LGAMMA 
%token <func> LN
%token <func> LOG 
%token <func> LOGISTIC 
%token <func> MAXP
%token <func> MINP
%token <func> MOD 
%token <func> NORM
%token <func> NORMP
%token <func> PI 
%token <func> RAD
%token <func> RAND
%token <func> RNORM
%token <func> SETNO
%token <func> SIN
%token <func> SQR 
%token <func> SQRT
%token <func> TAN 
%token <ival> INUM
%token <pset> VX1
%token <pset> VX2
%token <pset> VY1
%token <pset> VY2
%token <pset> WX1
%token <pset> WX2
%token <pset> WY1
%token <pset> WY2
%token <pset> DELAYP
%token <pset> DOUBLEBUFFER
%token <pset> DOWN
%token <pset> ABSOLUTE
%token <pset> ABORT
%token <pset> ACTIVATE
%token <pset> ACTIVE
%token <pset> ALT
%token <pset> ALTERNATE
%token <pset> ALTXAXIS
%token <pset> ALTYAXIS
%token <pset> ANGLE
%token <pset> ANNOTATE
%token <pset> APPEND
%token <pset> AREA
%token <pset> ARRANGE
%token <pset> ARROW
%token <pset> AUTO
%token <pset> AUTOSCALE
%token <pset> AUTOTICKS
%token <pset> AVG
%token <pset> AXIS
%token <pset> AXES
%token <pset> BACKBUFFER
%token <pset> BACKGROUND
%token <pset> BAR
%token <pset> BATCH
%token <pset> BLOCK
%token <pset> BIN
%token <pset> BOTH
%token <pset> BOTTOM
%token <pset> BOX
%token <pset> BOXPLOT
%token <pset> CD
%token <pset> CELLS
%token <pset> CENTER
%token <pset> CHAR
%token <pset> CHRSTR
%token <pset> CLEAR
%token <pset> CLICK
%token <pset> CMAP
%token <pset> COLOR
%token <pset> COMMENT
%token <pset> COPY
%token <pset> CYCLE
%token <pset> DECIMAL
%token <pset> DEF
%token <pset> DEFAULT
%token <pset> DELETE
%token <pset> DEVICE
%token <pset> DFT
%token <pset> DIFFERENCE
%token <pset> DISK
%token <pset> DRAW2
%token <pset> DROP
%token <pset> DXDX
%token <pset> DXP
%token <pset> DYDY
%token <pset> DYP
%token <pset> ECHO
%token <pset> EDIT
%token <pset> ELSE
%token <pset> END
%token <pset> ERRORBAR
%token <pset> EXIT
%token <pset> EXPONENTIAL
%token <pset> FALSEP
%token <pset> FFT
%token <pset> FILEP
%token <pset> FILL
%token <pset> FIND
%token <pset> FIXEDPOINT
%token <pset> FLUSH
%token <pset> FOCUS
%token <pset> FOLLOWS
%token <pset> FONTP
%token <pset> FOREGROUND
%token <pset> FORMAT
%token <pset> FRONTBUFFER
%token <pset> FRAMEP
%token <pset> GETP
%token <pset> GRAPH
%token <pset> GRAPHNO
%token <pset> GRAPHS
%token <pset> GRAPHTYPE
%token <pset> GRID
%token <pset> HARDCOPY
%token <pset> HBAR
%token <pset> HBOXPLOT
%token <pset> HGAP
%token <pset> HIDDEN
%token <pset> HORIZONTAL
%token <pset> HPGLL
%token <pset> HPGLP
%token <pset> HISTO
%token <pset> IF
%token <pset> IGNORE
%token <pset> IHL
%token <pset> IMAGE
%token <pset> IN
%token <pset> PLOT_INIT
%token <pset> INOUT
%token <pset> INTEGRATE
%token <pset> INTERP
%token <pset> INVDFT
%token <pset> INVFFT
%token <pset> JUST
%token <pset> KILL
%token <pset> LABEL
%token <pset> LAYOUT
%token <pset> LEAVE
%token <pset> LEFT
%token <pset> LEGEND
%token <pset> LENGTH
%token <pset> LEVEL
%token <pset> LEVELS
%token <pset> LINE
%token <pset> LINESTYLE
%token <pset> LINETO
%token <pset> LINEWIDTH
%token <pset> LINK
%token <pset> LOAD
%token <pset> LOCATOR
%token <pset> LOCTYPE
%token <pset> LOGX
%token <pset> LOGY
%token <pset> LOGXY
%token <pset> MAJOR
%token <pset> MIFL
%token <pset> MIFP
%token <pset> MINOR
%token <pset> MISSINGP
%token <pset> MOVE
%token <pset> MOVE2
%token <pset> MOVETO
%token <pset> NEGATE
%token <pset> NO
%token <pset> PLOT_NONE
%token <pset> NORMAL
%token <pset> NXY
%token <pset> OFF
%token <pset> OFFSETX
%token <pset> OFFSETY
%token <pset> ON
%token <pset> OP
%token <pset> ORIENT
%token <pset> OUT
%token <pset> PAGE
%token <pset> PARA
%token <pset> PARALLEL
%token <pset> PARAMETERS
%token <pset> PARAMS
%token <pset> PATTERN
%token <pset> PERIMETER
%token <pset> PERP
%token <pset> PERPENDICULAR
%token <pset> PIE
%token <pset> PIPE
%token <pset> PLACE
%token <pset> POINT
%token <pset> POLAR
%token <pset> POWER
%token <pset> PREC
%token <pset> PREPEND
%token <pset> PRINT
%token <pset> PS
%token <pset> PSCOLORP
%token <pset> PSMONOP
%token <pset> PSCOLORL
%token <pset> PSMONOL
%token <pset> PUSH
%token <pset> POP
%token <pset> PUTP
%token <pset> RAWSPICE
%token <pset> READ
%token <pset> REDRAW
%token <pset> REGRESS
%token <pset> REGNUM
%token <pset> REGIONS
%token <pset> PLOT_RENDER
%token <pset> REVERSE
%token <pset> RIGHT
%token <pset> RISER
%token <pset> ROT
%token <pset> RUNAVG
%token <pset> RUNMED
%token <pset> RUNSTD
%token <pset> RUNMIN
%token <pset> RUNMAX
%token <pset> SAMPLE
%token <pset> SAVEALL
%token <pset> SCALE
%token <pset> SCIENTIFIC
%token <pset> SET
%token <pset> SETNUM
%token <pset> SETS
%token <pset> SIGN
%token <pset> SIZE
%token <pset> SKIP
%token <pset> SLEEP
%token <pset> SLICE
%token <pset> SOURCE
%token <pset> SPEC
%token <pset> SPECIFIED
%token <pset> SPECTRUM
%token <pset> STACK
%token <pset> STACKEDBAR
%token <pset> STACKEDHBAR
%token <pset> STACKEDLINE
%token <pset> STAGGER
%token <pset> PLOT_START
%token <pset> STARTTYPE
%token <pset> STATUS
%token <pset> STOP
%token <pset> PLOT_STRING
%token <pset> SUBTITLE
%token <pset> SWAPBUFFER
%token <pset> SYMBOL
%token <pset> TICKP
%token <pset> TICKLABEL
%token <pset> TICKMARKS
%token <pset> TITLE
%token <pset> TO
%token <pset> TOP
%token <pset> TRUEP
%token <pset> TYPE
%token <pset> UP
%token <pset> VELOCITY
%token <pset> VERTICAL
%token <pset> VGAP
%token <pset> VIEW
%token <pset> WITH
%token <pset> WORLD
%token <pset> WRITE
%token <pset> PLOT_X
%token <pset> X0
%token <pset> X1
%token <pset> XAXES
%token <pset> XAXIS
%token <pset> XCOR
%token <pset> XMAX
%token <pset> XMIN
%token <pset> XY
%token <pset> XYARC
%token <pset> XYBOX
%token <pset> XYBOXPLOT
%token <pset> XYFIXED
%token <pset> XYHILO
%token <pset> XYRT
%token <pset> XYSEG
%token <pset> XYSTRING
%token <pset> XYDX
%token <pset> XYDY
%token <pset> XYDXDX
%token <pset> XYDYDY
%token <pset> XYDXDY
%token <pset> XYX2Y2
%token <pset> XYXX
%token <pset> XYYY
%token <pset> XYZ
%token <pset> XYZW
%token <pset> XYUV
%token <pset> PLOT_Y
%token <pset> Y0
%token <pset> Y1
%token <pset> Y2
%token <pset> Y3
%token <pset> Y4
%token <pset> Y5
%token <pset> YAXES
%token <pset> YAXIS
%token <pset> YES
%token <pset> YMAX
%token <pset> YMIN
%token <pset> ZEROXAXIS
%token <pset> ZEROYAXIS
%token <pset> ABOVE
%token <pset> BELOW
%token <pset> POLYI
%token <pset> POLYO
%token <pset> GENERAL
%token <pset> DDMMYY
%token <pset> YYMMDD
%token <pset> MMDDYY
%token <pset> MMYY
%token <pset> MMDD
%token <pset> MONTHDAY
%token <pset> DAYMONTH
%token <pset> MONTHS
%token <pset> MONTHL
%token <pset> DAYOFWEEKS
%token <pset> DAYOFWEEKL
%token <pset> DAYOFYEAR
%token <pset> HMS
%token <pset> MMDDHMS
%token <pset> MMDDYYHMS
%token <pset> YYMMDDHMS
%token <pset> DEGREESLON
%token <pset> DEGREESMMLON
%token <pset> DEGREESMMSSLON
%token <pset> MMSSLON
%token <pset> DEGREESLAT
%token <pset> DEGREESMMLAT
%token <pset> DEGREESMMSSLAT
%token <pset> MMSSLAT
%token <pset> DOT
%token <pset> STAR
%token <pset> PLUS
%token <pset> CROSS
%token <pset> CIRCLE
%token <pset> SQUARE
%token <pset> DIAMOND
%token <pset> TRIANGLE1
%token <pset> TRIANGLE2
%token <pset> TRIANGLE3
%token <pset> TRIANGLE4
%token <pset> SPLINE

%token <pset> LANDSCAPE
%token <pset> PORTRAIT
%token <pset> FREE
%token <pset> FIXED
%token <pset> STATUSBAR
%token <pset> LOCATORBAR
%token <pset> TOOLBAR

%token <ptr> VAR 
%token <val> NUMBER
%token <val> FITPARM

%type <val> expr
%type <ptr> vexpr
%type <ptr> asgn
%type <ptr> vasgn
%type <ptr> rasgn
%type <pset> vector
%type <pset> onoff
%type <pset> colpat
%type <pset> torf
%type <pset> worldview
%type <pset> formatchoice
%type <pset> inoutchoice
%type <pset> signchoice
%type <pset> justchoice
%type <pset> opchoice
%type <pset> direction
%type <pset> printer
%type <pset> regionset
%type <pset> regiontype
%type <pset> graphtype
%type <pset> selectsets
%type <pset> setvelocity
%type <pset> parmset
%type <pset> prop
%type <pset> filltype
%type <pset> sourcetype
%type <pset> xytype
%type <pset> runtype
%type <pset> ffttype
%type <pset> extremetype
%type <pset> pagelayout
%right '='
%left OR
%left AND
%nonassoc GT LT LE GE EQ NE
%left '+' '-'
%left '*' '/' '%'
%right '^'
%right UMINUS NOT

%%

list:
	| asgn '\n'
	| vasgn '\n'
	| rasgn '\n'
	| expr '\n' {
	    result = $1;
	}
	| vexpr '\n' {
	    result = *$1;
	}
	| parmset '\n'
	| regionset '\n'
	| setaxis '\n'
	| setvelocity '\n'
	| set_setprop '\n'
	| setprint '\n'
	| error '\n' {
	    return 1;
	}
	;

setprint:
	PRINT printer CHRSTR {
	    if ($2 == FILEP) {
		set_printer(FILEP, (char *)$3);
	    }
	    else {
		set_printer($2, (char *)$3);
	    }
	    free((char *) $3);
	}
	| PRINT TO printer CHRSTR {
	    if ($3 == FILEP) {
		set_printer(FILEP, (char *)$4);
	    }
	    else {
		set_printer($3, (char *)$4);
	    }
	    free((char *) $4);
	}
	| PRINT TO printer {
	    if ($3 == FILEP) {
		set_printer(FILEP, NULL);
	    }
	    else {
		set_printer($3, NULL);
	    }
	}
	| DEVICE NUMBER {
	    tdevice = (int) $2;
	}
	| HARDCOPY DEVICE NUMBER {
	    hdevice = (int) $3;
	}
	| HARDCOPY {
	    do_hardcopy();
	}
	| PS LINEWIDTH NUMBER 
	{ 
	    extern int pslwfactor; /* fudge factor for linewidths in the PS driver */
	    pslwfactor = (int)$3; 
	}
	;

printer:
	PSMONOP { $$ = GR_PS_P; }
	| PSMONOL { $$ = GR_PS_L; }
	| MIFP { $$ = GR_MIF_P; }
	| MIFL { $$ = GR_MIF_L; }
	| HPGLP { $$ = GR_HPGL_P; }
	| HPGLL { $$ = GR_HPGL_L; }
	| HARDCOPY { $$ = hdevice; }
	| FILEP { $$ = FILEP; }
	;

regionset:
	REGNUM onoff {
	    rg[$1].active = $2;
	}
	| REGNUM TYPE regiontype {
	    rg[$1].type = $3;
	}
	| REGNUM COLOR NUMBER {
	    rg[$1].color = checkon(COLOR, rg[$1].color, (int) $3);
	}
	| REGNUM LINESTYLE NUMBER {
	    rg[$1].lines = checkon(LINESTYLE, rg[$1].lines, (int) $3);
	}
	| REGNUM LINEWIDTH NUMBER {
	    rg[$1].linew = checkon(LINEWIDTH, rg[$1].linew, (int) $3);
	}
	| REGNUM LINE expr ',' expr ',' expr ',' expr
	{
	    rg[$1].x1 = $3;
	    rg[$1].y1 = $5;
	    rg[$1].x2 = $7;
	    rg[$1].y2 = $9;
	}
	| REGNUM XY expr ',' expr
	{
	    if (rg[$1].x == NULL || rg[$1].n == 0) {
		rg[$1].n = 0;
		rg[$1].x = (double *) calloc(1, sizeof(double));
		rg[$1].y = (double *) calloc(1, sizeof(double));
	    } else {
		rg[$1].x = (double *) realloc(rg[$1].x, (rg[$1].n + 1) * sizeof(double));
		rg[$1].y = (double *) realloc(rg[$1].y, (rg[$1].n + 1) * sizeof(double));
	    }
	    rg[$1].x[rg[$1].n] = $3;
	    rg[$1].y[rg[$1].n] = $5;
	    rg[$1].n++;
	}
	| LINK REGNUM TO GRAPHNO {
	    rg[$2].linkto[$4] = TRUE;
	}
	;

parmset:
	REDRAW {
	    drawgraph();
	}
	| AUTO REDRAW onoff {
	    auto_redraw = ($3 == ON);
	}
	| CD CHRSTR {
	    char buf[MAXPATHLEN];
	    strcpy(buf, (char *) $2);
	    expand_tilde(buf); 
	    if (chdir(buf) >= 0) {
		strcpy(workingdir, buf);
	    	if (inwin) {
		    set_title(workingdir);
	        }
	    }
	    free((char *) $2);
	}
	| ECHO CHRSTR {
	    if (inwin) {
		set_left_footer((char *)$2);
	    }
	    else {
		printf("%s\n", (char *)$2);
	    }
	    free((char *) $2);
	}
	| BACKGROUND COLOR NUMBER {
	    setbgcolor((int) $3);
	}
	| CMAP NUMBER ',' NUMBER ',' NUMBER ',' NUMBER {
	    xlibsetcmap((int) $2, (int) $4, (int) $6, (int) $8);
	}
	| EXIT {
	    exit(0);
	}
        | PAGE LAYOUT pagelayout
        {
            set_pagelayout($3);
        }
        | STATUSBAR onoff
        {
            set_toolbars($1, $2 == ON);
        }
        | TOOLBAR onoff
        {
            set_toolbars($1, $2 == ON);
        }
        | LOCATORBAR onoff
        {
            set_toolbars($1, $2 == ON);
        }

	| DRAW2 expr ',' expr {
	    if (inwin) {
		my_draw2((double) $2, (double) $4);
		flush_pending();
	    }
	}
	| MOVE2 expr ',' expr {
	    if (inwin) {
		my_move2((double) $2, (double) $4);
	    }
	}
	| SYMBOL expr ',' expr ',' NUMBER {
	    if (inwin) {
		double x = (double) $2;
		double y = (double) $4;
		drawpolysym(&x, &y, 1, (int) $6, 0, 0, 1.0);
		flush_pending();
	    }
	}
	| COLOR NUMBER {
	    if (inwin) {
		setcolor((int) $2);
	    }
	}
	| LINEWIDTH NUMBER {
	    if (inwin) {
		setlinewidth((int) $2);
	    }
	}
	| LINESTYLE NUMBER {
	    if (inwin) {
		setlinestyle((int) $2);
	    }
	}
	| WRITE PLOT_STRING expr ',' expr ',' CHRSTR {
	}

	| PAGE direction
	{
	    switch ($2) {
	    case UP:
		gwindup_proc();
		break;
	    case DOWN:
		gwinddown_proc();
		break;
	    case RIGHT:
		gwindright_proc();
		break;
	    case LEFT:
		gwindleft_proc();
		break;
	    case IN:
		gwindshrink_proc();
		break;
	    case OUT:
		gwindexpand_proc();
		break;
	    }
	}
	| PAGE NUMBER {
	    scroll_proc((int) $2);
	}
	| PAGE INOUT NUMBER {
	    scrollinout_proc((int) $3);
	}
	| LINK PAGE onoff {
	    scrolling_islinked = $3 == ON;
	}
	| DOUBLEBUFFER torf {
	    my_doublebuffer($2 == TRUEP);
	}
	| FRONTBUFFER torf {
	    my_frontbuffer($2 == TRUEP);
	}
	| BACKBUFFER torf {
	    my_backbuffer($2 == TRUEP);
	}
	| SWAPBUFFER {
	    my_swapbuffer();
	}
	| SLEEP NUMBER {
	    sleep((int) $2);
	}
	| DELAYP NUMBER {	/* TODO add delay function */
	}
	| ABORT torf {		/* TODO add abort flag and function */
	}
	| GETP CHRSTR
	{
	    gotparams = TRUE;
	    strcpy(paramfile, (char *) $2);
	    free((char *) $2);
	}
	| PUTP CHRSTR
	{
	    if (!fexists((char *) $2)) {
		FILE *pp = fopen((char *) $2, "w");
		if (pp != NULL) {
		    putparms(cg, pp, 0);
		    fclose(pp);
		} else {
		    errwin("Unable to write parameter file");
		}
	    }
	    free((char *) $2);
	}
	| WITH GRAPHNO {
	    cg = (int) $2;
	    g[cg].parmsread = TRUE;
	    change_gno = cg;
	}
	| WITH SETNUM {
	    curset = (int) $2;
	}

/* Hot links */
	| SETNUM LINK sourcetype CHRSTR {
	    set_hotlink(cg, $1, 1, (char *) $4, (int)$3);
	    free((char *) $4);
	}
	| GRAPHNO '.' SETNUM LINK sourcetype CHRSTR {
	    set_hotlink($1, $3, 1, (char *) $6, $5);
	    free((char *) $6);
	}
	| SETNUM LINK onoff {
	    set_hotlink(cg, $1, $3 == ON, NULL, 0);
	}
	| GRAPHNO '.' SETNUM LINK onoff {
	    set_hotlink($1, $3, $5 == ON, NULL, 0);
	}
	| ACTIVATE SETNUM NUMBER {
	    do_activateset(cg, $2, (int) $3);
	}
	| ACTIVATE GRAPHNO '.' SETNUM NUMBER {
	    do_activateset($2, $4, (int) $5);
	}
	| SETNUM LENGTH NUMBER {
	    setlength(cg, $1, (int) $3);
	    updatesetminmax(cg, $1);
	    update_set_status(cg, $1);
	}
	| GRAPHNO '.' SETNUM LENGTH NUMBER {
	    setlength($1, $3, (int) $5);
	    updatesetminmax($1, $3);
	    update_set_status($1, $3);
	}
	| SETNUM POINT expr ',' expr {
	    add_point(cg, $1, $3, $5, 0.0, 0.0, XY);
	}
	| GRAPHNO '.' SETNUM POINT expr ',' expr {
	    add_point($1, $3, $5, $7, 0.0, 0.0, XY);
	}

	| SETNUM DROP NUMBER ',' NUMBER {
	    int start = (int) $3 - 1;
	    int stop = (int) $5 - 1;
	    int dist = stop - start + 1;
	    if (dist > 0 && start >= 0) {
	        droppoints(cg, $1, start, stop, dist);
	    }
	}
	| GRAPHNO '.' SETNUM DROP NUMBER ',' NUMBER {
	    int start = (int) $5 - 1;
	    int stop = (int) $7 - 1;
	    int dist = stop - start + 1;
	    if (dist > 0 && start >= 0) {
	        droppoints($1, $3, start, stop, dist);
	    }
	}
	| COPY SETNUM TO SETNUM {
	    do_copyset(cg, $2, cg, $4);
	}
	| COPY GRAPHNO '.' SETNUM TO GRAPHNO '.' SETNUM {
	    do_copyset($2, $4, $6, $8);
	}
	| MOVE SETNUM TO SETNUM {
	    do_moveset(cg, $2, cg, $4);
	}
	| MOVE GRAPHNO '.' SETNUM TO GRAPHNO '.' SETNUM {
	    do_moveset($2, $4, $6, $8);
	}
	| KILL SETNUM
	{
	    killset(cg, $2);
	}
	| KILL SETS
	{
	    int i;
	    for (i = 0; i < g[cg].maxplot; i++) {
		killset(cg, i);
	    }
	}
	| KILL SETNUM SAVEALL
	{
	    softkillset(cg, $2);
	}
	| KILL SETS SAVEALL
	{
	    int i;
	    for (i = 0; i < g[cg].maxplot; i++) {
		softkillset(cg, i);
	    }
	}
	| KILL GRAPHNO
	{
	    kill_graph($2);
	}
	| KILL GRAPHS
	{
	    kill_graph(maxgraph);
	}
	| FLUSH
	{
	    wipeout(0);
	}
	| ARRANGE NUMBER ',' NUMBER
	{
	    arrange_graphs((int) $2, (int) $4);
	}
	| LOAD VAR NUMBER ',' expr ',' expr
	{
	    int i;
	    for (i = 0; i < (int) $3; i++) {
		$2[i] = $5 + $7 * i;
	    }
	}
	| REGRESS '(' SETNUM ',' NUMBER ')'
	{
	    int setno = $3, ideg = (int) $5;
	    do_regress(setno, ideg, 0, -1, 0);
	}
	| runtype '(' SETNUM ',' NUMBER ')'
	{
	    do_running_command($1, $3, (int) $5);
	}
	| ffttype '(' SETNUM ',' NUMBER ')'
	{
	    do_fourier_command($1, $3, (int) $5);
	}
	| SPLINE '(' SETNUM ',' expr ',' expr ',' NUMBER ')'
	{
	    do_spline($3, $5, $7, (int) $9);
	}
	| HISTO '(' SETNUM ',' expr ',' expr ',' NUMBER ')'
	{
	    do_histo_command($3, -1, -1, $5, $7, (int) $9);
	}
	| DIFFERENCE '(' SETNUM ',' NUMBER ')'
	{
	    do_differ($3, (int) $5);
	}
	| INT '(' SETNUM ')'
	{
	    do_int($3, 0);
	}
	| AUTOSCALE
	{
	    if (activeset(cg)) {
		defaultgraph(cg);
		default_axis(cg, g[cg].auto_type, X_AXIS);
		default_axis(cg, g[cg].auto_type, ZX_AXIS);
		default_axis(cg, g[cg].auto_type, Y_AXIS);
		default_axis(cg, g[cg].auto_type, ZY_AXIS);
		update_world(cg);
		drawgraph();
	    } else {
		errwin("No active sets!");
	    }
	}
	| AUTOSCALE XAXES
	{
	    if (activeset(cg)) {
		defaultx(cg, -1);
		default_axis(cg, g[cg].auto_type, X_AXIS);
		default_axis(cg, g[cg].auto_type, ZX_AXIS);
		update_world(cg);
		drawgraph();
	    } else {
		errwin("No active sets!");
	    }
	}
	| AUTOSCALE YAXES
	{
	    if (activeset(cg)) {
		defaulty(cg, -1);
		default_axis(cg, g[cg].auto_type, Y_AXIS);
		default_axis(cg, g[cg].auto_type, ZY_AXIS);
		update_world(cg);
		drawgraph();
	    } else {
		errwin("No active sets!");
	    }
	}
	| AUTOSCALE SETNUM
	{
	    if (isactive_set(cg, $2)) {
		defaultsetgraph(cg, $2);
		default_axis(cg, g[cg].auto_type, X_AXIS);
		default_axis(cg, g[cg].auto_type, ZX_AXIS);
		default_axis(cg, g[cg].auto_type, Y_AXIS);
		default_axis(cg, g[cg].auto_type, ZY_AXIS);
		update_world(cg);
		drawgraph();
	    } else {
		errwin("Set not active");
	    }
	}
	| LOCATOR onoff
	{
	    extern int go_locateflag;
	    go_locateflag = ($2 == ON);
	}
	| FOCUS GRAPHNO
	{
	    draw_focus(cg);
	    cg = (int) $2;
	    defineworld(g[cg].w.xg1, g[cg].w.yg1, g[cg].w.xg2, g[cg].w.yg2, 
			islogx(cg), islogy(cg));
	    viewport(g[cg].v.xv1, g[cg].v.yv1, g[cg].v.xv2, g[cg].v.yv2);
	    draw_focus(cg);
	    update_all(cg);
	}
	| FOCUS onoff {
	    draw_focus_flag = $2;
	}
	| FOCUS SET {
	    focus_policy = $2;
	}
	| FOCUS FOLLOWS {
	    focus_policy = $2;
	}
	| FOCUS CLICK {
	    focus_policy = $2;
	}
	| SOURCE sourcetype {
	    cursource = $2;
	}
	| TYPE xytype {
	    curtype = $2;
	    change_type = curtype;
	}
	| FORMAT formatchoice
	{
	    readxformat = $2;
	}
	| READ CHRSTR
	{
	    gotread = TRUE;
	    readtype = curtype;
	    readsrc = cursource;
	    strcpy(readfile, (char *) $2);
	    free((char *) $2);
	}
	| READ BATCH CHRSTR
	{
	    gotbatch = TRUE;
	    strcpy(batchfile, (char *) $3);
	    free((char *) $3);
	}
	| READ BLOCK CHRSTR
	{
	    getdata(cg, (char *) $3, DISK, BLOCK);
	    free((char *) $3);
	}
	| READ BLOCK sourcetype CHRSTR
	{
	    getdata(cg, (char *) $4, $3, BLOCK);
	    free((char *) $4);
	}
	| BLOCK xytype CHRSTR
	{
	    create_set_fromblock(cg, $2, (char *) $3);
	    free((char *) $3);
	}
	| READ xytype CHRSTR
	{
	    gotread = TRUE;
	    readtype = $2;
	    readsrc = cursource;
	    strcpy(readfile, (char *) $3);
	    free((char *) $3);
	}
	| READ xytype sourcetype CHRSTR
	{
	    gotread = TRUE;
	    strcpy(readfile, (char *) $4);
	    readtype = $2;
	    readsrc = $3;
	    free((char *) $4);
	}
	| READ IMAGE CHRSTR
	{
	    read_image((char *) $3);
	    free((char *) $3);
	}
	| WRITE IMAGE CHRSTR
	{
	    write_image((char *) $3);
	    free((char *) $3);
	}
	| IMAGE XY NUMBER ',' NUMBER
	{
	    imagex = (int) $3;
	    imagey = (int) $5;
	}
	| WRITE SETNUM
	{
	    outputset(cg, $2, (char *) NULL, (char *) NULL);
	}
	| WRITE SETNUM FORMAT CHRSTR
	{
	    outputset(cg, $2, (char *) NULL, (char *) $4);
	    free((char *) $4);
	}
	| WRITE SETNUM FILEP CHRSTR
	{
	    outputset(cg, $2, (char *) $4, (char *) NULL);
	    free((char *) $4);
	}
	| WRITE SETNUM FILEP CHRSTR FORMAT CHRSTR
	{
	    outputset(cg, $2, (char *) $4, (char *) $6);
	    free((char *) $4);
	    free((char *) $6);
	}
        | SAVEALL CHRSTR
        {
            extern char sformat[];
            do_writesets(maxgraph, -1, 1, (char *) $2, sformat);
            free((char *) $2);
        }
	| PUSH {
	    push_world();
	}
	| POP {
	    pop_world();
	}
	| CYCLE {
	    cycle_world_stack();
	}
	| STACK NUMBER {
	    if ((int) $2 > 0)
		show_world_stack((int) $2 - 1);
	}
	| STACK WORLD expr ',' expr ',' expr ',' expr TICKP expr ',' expr ',' expr ',' expr
	{
	    add_world(cg, $3, $5, $7, $9, $11, $13, $15, $17);
	}
	| CLEAR STACK {
	    clear_world_stack();
	}
	| CLEAR BOX {
	    do_clear_boxes();
	}
	| WITH BOX {
	    curbox = next_box();
	}
	| WITH BOX NUMBER {
	    curbox = (int) $3;
	}
	| BOX onoff {
	    boxes[curbox].active = $2;
	}
	| BOX GRAPHNO {
	    boxes[curbox].gno = $2;
	}
	| BOX expr ',' expr ',' expr ',' expr
	{
	    if (curbox >= 0 && curbox < MAXBOXES) {
		boxes[curbox].x1 = $2;
		boxes[curbox].y1 = $4;
		boxes[curbox].x2 = $6;
		boxes[curbox].y2 = $8;
	    }
	}
	| BOX LOCTYPE worldview {
	    box_loctype = $3;
	}
	| BOX LINESTYLE NUMBER {
	    box_lines = checkon(LINESTYLE, box_lines, (int) $3);
	}
	| BOX LINEWIDTH NUMBER {
	    box_linew = checkon(LINEWIDTH, box_linew, (int) $3);
	}
	| BOX COLOR NUMBER {
	    box_color = checkon(COLOR, box_color, (int) $3);
	}
	| BOX FILL filltype {
	    box_fill = $3;
	}
	| BOX FILL COLOR NUMBER {
	    box_fillcolor = checkon(COLOR, box_fillcolor, (int) $4);
	}
	| BOX FILL PATTERN NUMBER {
	    box_fillpat = checkon(PATTERN, box_fillpat, (int) $4);
	}
	| BOX DEF
	{
	    if (curbox >= 0 && curbox < MAXBOXES) {
		boxes[curbox].lines = box_lines;
		boxes[curbox].linew = box_linew;
		boxes[curbox].color = box_color;
		boxes[curbox].fill = box_fill;
		boxes[curbox].fillcolor = box_fillcolor;
		boxes[curbox].fillpattern = box_fillpat;
		boxes[curbox].loctype = box_loctype;
	    }
	}
	| WITH LINE {
	    curline = next_line();
	}
	| WITH LINE NUMBER {
	    curline = (int) $3;
	}
	| CLEAR LINE {
	    do_clear_lines();
	}
	| LINE onoff {
	    lines[curline].active = $2;
	}
	| LINE GRAPHNO {
	    lines[curline].gno = $2;
	}
	| LINE expr ',' expr ',' expr ',' expr
	{
	    lines[curline].x1 = $2;
	    lines[curline].y1 = $4;
	    lines[curline].x2 = $6;
	    lines[curline].y2 = $8;
	}
	| LINE LOCTYPE worldview {
	    line_loctype = $3;
	}
	| LINE LINEWIDTH NUMBER {
	    line_linew = checkon(LINEWIDTH, line_linew, (int) $3);
	}
	| LINE LINESTYLE NUMBER {
	    line_lines = checkon(LINESTYLE, line_lines, (int) $3);
	}
	| LINE COLOR NUMBER {
	    line_color = checkon(COLOR, line_color, (int) $3);
	}
	| LINE ARROW NUMBER {
	    line_arrow = checkon(ARROW, line_arrow, (int) $3);
	}
	| LINE ARROW SIZE NUMBER {
	    line_asize = $4;
	}
	| LINE ARROW TYPE NUMBER {
	    line_atype = (int) $4;
	}
	| LINE DEF
	{
	    if (curline >= 0 && curline < MAXLINES) {
		lines[curline].lines = line_lines;
		lines[curline].linew = line_linew;
		lines[curline].color = line_color;
		lines[curline].arrow = line_arrow;
		lines[curline].asize = line_asize;
		lines[curline].atype = line_atype;
		lines[curline].loctype = line_loctype;
	    }
	}
	| CLEAR PLOT_STRING {
	    do_clear_text();
	}
	| WITH PLOT_STRING {
	    curstring = next_string();
	}
	| WITH PLOT_STRING NUMBER {
	    curstring = (int) $3;
	}
	| PLOT_STRING onoff {
	    pstr[curstring].active = $2;
	}
	| PLOT_STRING GRAPHNO {
	    pstr[curstring].gno = $2;
	}
	| PLOT_STRING expr ',' expr
	{
	    pstr[curstring].x = $2;
	    pstr[curstring].y = $4;
	}
	| PLOT_STRING LOCTYPE worldview {
	    string_loctype = $3;
	}
	| PLOT_STRING LINEWIDTH NUMBER {
	    string_linew = checkon(LINEWIDTH, string_linew, (int) $3);
	}
	| PLOT_STRING COLOR NUMBER {
	    string_color = checkon(COLOR, string_color, (int) $3);
	}
	| PLOT_STRING ROT NUMBER {
	    string_rot = (int) $3;
	}
	| PLOT_STRING FONTP NUMBER {
	    string_font = checkon(FONTP, string_font, (int) $3);
	}
	| PLOT_STRING JUST NUMBER {
	    string_just = checkon(JUST, string_just, (int) $3);
	}
	| PLOT_STRING CHAR SIZE NUMBER {
	    string_size = $4;
	}
	| PLOT_STRING DEF CHRSTR
	{
	    set_plotstr_string(&pstr[curstring], (char *) $3);
	    pstr[curstring].linew = string_linew;
	    pstr[curstring].color = string_color;
	    pstr[curstring].font = string_font;
	    pstr[curstring].just = string_just;
	    pstr[curstring].loctype = string_loctype;
	    pstr[curstring].rot = string_rot;
	    pstr[curstring].charsize = string_size;
	    free((char *) $3);
	}
	| DEFAULT LINESTYLE NUMBER {
	    grdefaults.lines = (int) $3;
	}
	| DEFAULT LINEWIDTH NUMBER {
	    grdefaults.linew = (int) $3;
	}
	| DEFAULT COLOR NUMBER {
	    grdefaults.color = (int) $3;
	}
	| DEFAULT CHAR SIZE NUMBER {
	    grdefaults.charsize = $4;
	}
	| DEFAULT FONTP NUMBER {
	    grdefaults.font = (int) $3;
	}
	| DEFAULT FONTP SOURCE NUMBER {
	    grdefaults.fontsrc = (int) $4;
	}
	| DEFAULT SYMBOL SIZE NUMBER {
	    grdefaults.symsize = $4;
	}
	| WORLD expr ',' expr ',' expr ',' expr
	{
	    g[cg].w.xg1 = $2;
	    g[cg].w.yg1 = $4;
	    g[cg].w.xg2 = $6;
	    g[cg].w.yg2 = $8;
	}
	| WORLD XMIN expr {
	    g[cg].w.xg1 = $3;
	}
	| WORLD XMAX expr {
	    g[cg].w.xg2 = $3;
	}
	| WORLD YMIN expr {
	    g[cg].w.yg1 = $3;
	}
	| WORLD YMAX expr {
	    g[cg].w.yg2 = $3;
	}
	| VIEW expr ',' expr ',' expr ',' expr
	{
	    g[cg].v.xv1 = $2;
	    g[cg].v.yv1 = $4;
	    g[cg].v.xv2 = $6;
	    g[cg].v.yv2 = $8;
	}
	| VIEW XMIN NUMBER {
	    g[cg].v.xv1 = $3;
	}
	| VIEW XMAX NUMBER {
	    g[cg].v.xv2 = $3;
	}
	| VIEW YMIN NUMBER {
	    g[cg].v.yv1 = $3;
	}
	| VIEW YMAX NUMBER {
	    g[cg].v.yv2 = $3;
	}
	| TITLE CHRSTR {
	    set_plotstr_string(&g[cg].labs.title, (char *) $2);
	    free((char *) $2);
	}
	| TITLE FONTP NUMBER {
	    g[cg].labs.title.font = checkon(FONTP, g[cg].labs.title.font, (int) $3);
	}
	| TITLE SIZE NUMBER {
	    g[cg].labs.title.charsize = $3;
	}
	| TITLE COLOR NUMBER {
	    g[cg].labs.title.color = checkon(COLOR, g[cg].labs.title.color, (int) $3);
	}
	| TITLE LINEWIDTH NUMBER
	{
	    g[cg].labs.title.linew = checkon(LINEWIDTH, g[cg].labs.title.linew, (int) $3);
	}
	| SUBTITLE CHRSTR {
	    set_plotstr_string(&g[cg].labs.stitle, (char *) $2);
	    free((char *) $2);
	}
	| SUBTITLE FONTP NUMBER
	{
	    g[cg].labs.stitle.font = checkon(FONTP, g[cg].labs.stitle.font, (int) $3);
	}
	| SUBTITLE SIZE NUMBER {
	    g[cg].labs.stitle.charsize = $3;
	}
	| SUBTITLE COLOR NUMBER
	{
	    g[cg].labs.stitle.color = checkon(COLOR, g[cg].labs.stitle.color, (int) $3);
	}
	| SUBTITLE LINEWIDTH NUMBER
	{
	    g[cg].labs.stitle.linew = checkon(LINEWIDTH, g[cg].labs.stitle.color, (int) $3);
	}
	| GRAPHS MAXP SETS NUMBER {
	    realloc_plots((int) $4);
	}
	| LEGEND onoff {
	    g[cg].l.active = $2;
	}
	| LEGEND LOCTYPE worldview {
	    g[cg].l.loctype = $3;
	}
	| LEGEND LAYOUT NUMBER {
	    g[cg].l.layout = (int) $3;
	}
	| LEGEND VGAP NUMBER {
	    g[cg].l.vgap = (int) $3;
	}
	| LEGEND HGAP NUMBER {
	    g[cg].l.hgap = (int) $3;
	}
	| LEGEND LENGTH NUMBER {
	    g[cg].l.len = (int) $3;
	}
	| LEGEND BOX onoff {
	    g[cg].l.box = $3;
	}
	| LEGEND BOX FILL onoff {
	    g[cg].l.boxfill = $4;
	}
	| LEGEND BOX FILL WITH colpat {
	    g[cg].l.boxfillusing = $5;
	}
	| LEGEND BOX FILL colpat NUMBER
	{
	    if ($4 == COLOR) {
		g[cg].l.boxfillcolor = (int) $5;
	    } else {
		g[cg].l.boxfillpat = (int) $5;
	    }
	}
	| LEGEND BOX COLOR NUMBER {
	    g[cg].l.boxlcolor = checkon(COLOR, g[cg].l.boxlcolor, (int) $4);
	}
	| LEGEND BOX LINESTYLE NUMBER {
	    g[cg].l.boxlines = checkon(LINESTYLE, g[cg].l.boxlines, (int) $4);
	}
	| LEGEND BOX LINEWIDTH NUMBER {
	    g[cg].l.boxlinew = checkon(LINEWIDTH, g[cg].l.boxlinew, (int) $4);
	}
	| LEGEND expr ',' expr {
	    g[cg].l.legx = $2;
	    g[cg].l.legy = $4;
	}
	| LEGEND X1 expr {
	    g[cg].l.legx = $3;
	}
	| LEGEND Y1 expr {
	    g[cg].l.legy = $3;
	}
	| LEGEND CHAR SIZE NUMBER {
	    g[cg].l.charsize = $4;
	}
	| LEGEND FONTP NUMBER {
	    g[cg].l.font = checkon(FONTP, g[cg].l.font, (int) $3);
	}
	| LEGEND LINESTYLE NUMBER {
	    g[cg].l.lines = checkon(LINESTYLE, g[cg].l.lines, (int) $3);
	}
	| LEGEND LINEWIDTH NUMBER {
	    g[cg].l.linew = checkon(LINEWIDTH, g[cg].l.linew, (int) $3);
	}
	| LEGEND COLOR NUMBER {
	    g[cg].l.color = checkon(COLOR, g[cg].l.color, (int) $3);
	}
	| LEGEND PLOT_STRING NUMBER CHRSTR {
	    set_plotstr_string(&g[cg].l.str[(int) $3], (char *) $4);
	    free((char *) $4);
	}
	| FRAMEP onoff {
	    g[cg].f.active = $2;
	}
	| FRAMEP TYPE NUMBER {
	    g[cg].f.type = (int) $3;
	}
	| FRAMEP LINESTYLE NUMBER {
	    g[cg].f.lines = checkon(LINESTYLE, g[cg].f.lines, (int) $3);
	}
	| FRAMEP LINEWIDTH NUMBER {
	    g[cg].f.linew = checkon(LINEWIDTH, g[cg].f.linew, (int) $3);
	}
	| FRAMEP COLOR NUMBER {
	    g[cg].f.color = checkon(COLOR, g[cg].f.color, (int) $3);
	}
	| FRAMEP FILL onoff {
	    g[cg].f.fillbg = $3;
	}
	| FRAMEP BACKGROUND COLOR NUMBER {
	    g[cg].f.bgcolor = (int) $4;
	}
	| GRAPHNO onoff {
	    g[$1].active = $2;
	}
	| GRAPHNO LABEL onoff {
	    g[$1].label = $3;
	}
	| GRAPHNO AUTOSCALE TYPE AUTO {
	    g[$1].auto_type = $4;
	}
	| GRAPHNO AUTOSCALE TYPE SPEC {
	    g[$1].auto_type = $4;
	}
	| GRAPHNO AUTOSCALE torf {
	    g[$1].parmsread = ($3 == FALSEP);
	}
	| GRAPHNO HIDDEN torf {
	    g[$1].hidden = ($3 == TRUEP);
	}
	| GRAPHNO TYPE graphtype {
	    g[$1].type = $3;
	}
	| GRAPHNO FIXEDPOINT onoff {
	    g[$1].pointset = ($3 == ON);
	}
	| GRAPHNO FIXEDPOINT FORMAT formatchoice formatchoice
	{
	    g[$1].fx = $4;
	    g[$1].fy = $5;
	}
	| GRAPHNO FIXEDPOINT PREC NUMBER ',' NUMBER
	{
	    g[$1].px = (int)$4;
	    g[$1].py = (int)$6;
	}
	| GRAPHNO FIXEDPOINT XY expr ',' expr
	{
	    g[$1].dsx = $4;
	    g[$1].dsy = $6;
	}
	| GRAPHNO FIXEDPOINT TYPE NUMBER {
	    g[$1].pt_type = (int) $4;
	}
	| GRAPHNO MAXP SETS NUMBER {
	    realloc_graph_plots($1, (int) $4);
	}
	;

setvelocity:
        VELOCITY SCALE NUMBER { g[cg].vp.vscale = $3; }
        | VELOCITY LEGEND onoff { g[cg].vp.active = $3; }
        | VELOCITY LINEWIDTH NUMBER { g[cg].vp.linew = (int) $3; }
        | VELOCITY LINESTYLE NUMBER { g[cg].vp.lines = (int) $3; }
        | VELOCITY COLOR NUMBER { g[cg].vp.color = (int) $3; }
        | VELOCITY LOCTYPE worldview { g[cg].vp.loctype = $3; }
        | VELOCITY ARROW TYPE NUMBER { g[cg].vp.arrowtype = (int) $4; }
        ;

xytype:
	XY {
	    $$ = XY;
	}
	| XYARC {
	    $$ = XYARC;
	}
	| XYBOX {
	    $$ = XYBOX;
	}
	| XYBOXPLOT {
	    $$ = XYBOXPLOT;
	}
	| XYHILO {
	    $$ = XYHILO;
	}
	| XYRT {
	    $$ = XYRT;
	}
	| XYSEG {
	    $$ = XYSEG;
	}
	| XYSTRING {
	    $$ = XYSTRING;
	}
	| XYDX {
	    $$ = XYDX;
	}
	| XYDY {
	    $$ = XYDY;
	}
	| XYDXDX {
	    $$ = XYDXDX;
	}
	| XYDYDY {
	    $$ = XYDYDY;
	}
	| XYDXDY {
	    $$ = XYDXDY;
	}
	| XYX2Y2 {
	    $$ = XYX2Y2;
	}
	| XYXX {
	    $$ = XYXX;
	}
	| XYYY {
	    $$ = XYYY;
	}
	| XYZ {
	    $$ = XYZ;
	}
	| XYZW {
	    $$ = XYZW;
	}
	| XYUV {
	    $$ = XYUV;
	}
	| NXY {
	    $$ = NXY;
	}
	| BIN {
	    $$ = BIN;
	}
	| RAWSPICE {
	    $$ = RAWSPICE;
	}
	;

graphtype:
	XY {
	    $$ = $1;
	}
	| LOGX {
	    $$ = $1;
	}
	| LOGY {
	    $$ = $1;
	}
	| LOGXY {
	    $$ = $1;
	}
	| BAR {
	    $$ = $1;
	}
	| HBAR {
	    $$ = $1;
	}
	| STACKEDBAR {
	    $$ = $1;
	}
	| STACKEDHBAR {
	    $$ = $1;
	}
	| POLAR {
	    $$ = XY;		/* not active */
	}
	| XYFIXED {
	    $$ = XY;		/* not active */
	}
	| BOXPLOT {
	    $$ = $1;
	}
	| HBOXPLOT {
	    $$ = $1;
	}
	;

pagelayout:
        FREE {
            $$ = FREE;
        }
        | LANDSCAPE {
            $$ = LANDSCAPE;
        }
        | PORTRAIT {
            $$ = PORTRAIT;
        }
        | FIXED {
            $$ = FIXED;
        }
        ;

regiontype:
	ABOVE
	|  BELOW
	|  LEFT
	|  RIGHT
	|  POLYI
	|  POLYO
	;

set_setprop:
	selectsets setprop
/* for ranges - yet to be implemented
	|  SETNUM '-' SETNUM {
	    printf("%d-%d\n", $1, $3);
	}
	| GRAPHNO SETNUM '-' SETNUM {
	    printf("Graph %d %d-%d\n", $1, $2, $4);
	}
	|  GRAPHS SETNUM '-' SETNUM {
	    printf("ALL graphs %d-%d\n", $2, $4);
	}
*/
	;

setprop:
	onoff {
	    set_prop(whichgraph, SET, SETNUM, whichset, ACTIVE, $1, 0);
	}
	| IGNORE {
	    set_prop(whichgraph, SET, SETNUM, whichset, ACTIVE, $1, 0);
	}
	| TYPE xytype {
	    set_prop(whichgraph, SET, SETNUM, whichset, TYPE, $2, 0);
	}
	| MISSINGP expr {
	    set_prop(whichgraph, SET, SETNUM, whichset, MISSINGP, $2, 0);
	}
	| PREC NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, PREC, (int) $2, 0);
	}
	| FORMAT formatchoice {
	    set_prop(whichgraph, SET, SETNUM, whichset, FORMAT, $2, 0);
	}
	| SYMBOL expr {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, TYPE, (int) $2, 0);
	}
	| SYMBOL FILL NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, FILL, (int) $3, 0);
	}
	| SYMBOL CENTER torf {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, CENTER, ($3 == TRUEP), 0);
	}
	| SYMBOL SIZE expr {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, SIZE, $3, 0);
	}
	| SYMBOL CHAR NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, CHAR, (int) $3, 0);
	}
	| SYMBOL SKIP NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, SKIP, (int) $3, 0);
	}
	| SYMBOL COLOR expr {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, COLOR, (int) $3, 0);
	}
	| SYMBOL LINEWIDTH NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, LINEWIDTH, (int) $3, 0);
	}
	| SYMBOL LINESTYLE NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SYMBOL, LINESTYLE, (int) $3, 0);
	}
	| prop NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, $1, (int) $2, 0);
	}
	| FILL NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, FILL, TYPE, (int) $2, 0);
	}
	| FILL WITH colpat {
	    set_prop(whichgraph, SET, SETNUM, whichset, FILL, WITH, $3, 0);
	}
	| FILL colpat NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, FILL, $2, (int) $3, 0);
	}
	| SKIP NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, SKIP, (int) $2, 0);
	}
	| ERRORBAR LENGTH NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, LENGTH, $3, 0);
	}
	| ERRORBAR TYPE opchoice {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, TYPE, $3, 0);
	}
	| ERRORBAR LINEWIDTH NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, LINEWIDTH, (int) $3, 0);
	}
	| ERRORBAR LINESTYLE NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, LINESTYLE, (int) $3, 0);
	}
	| ERRORBAR RISER onoff {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, RISER, ACTIVE, $3, 0);
	}
	| ERRORBAR RISER LINEWIDTH NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, RISER, LINEWIDTH, (int) $4, 0);
	}
	| ERRORBAR RISER LINESTYLE NUMBER {
	    set_prop(whichgraph, SET, SETNUM, whichset, ERRORBAR, RISER, LINESTYLE, (int) $4, 0);
	}
	| XYZ expr ',' expr {
	    set_prop(whichgraph, SET, SETNUM, whichset, XYZ, $2, $4, 0);
	}
	| COMMENT CHRSTR {
	    set_prop(whichgraph, SET, SETNUM, whichset, COMMENT, (char *) $2, 0);
	    free((char *) $2);
	}
	;

setaxis:
	axis
	|  axis axisfeature
	|  allaxes
	|  GRAPHS axis
	|  GRAPHS axis axisfeature
	|  GRAPHS allaxes
	;

axis:
	XAXIS
	|  YAXIS
	|  ALTXAXIS
	|  ALTYAXIS
	|  ZEROXAXIS
	|  ZEROYAXIS
	;

allaxes:
	AXES axesprops
	|  XAXES axesprops
	|  YAXES axesprops
	;

axesprops:
	onoff {
	    set_axis_prop(whichgraph, naxis, $1, 0.0);
	}
	| COLOR NUMBER {
	    set_axis_prop(whichgraph, naxis, $1, $2);
	}
	| LINEWIDTH NUMBER {
	    set_axis_prop(whichgraph, naxis, $1, $2);
	}
	| LINESTYLE NUMBER {
	    set_axis_prop(whichgraph, naxis, $1, $2);
	}
	| FONTP NUMBER {
	    set_axis_prop(whichgraph, naxis, $1, $2);
	}
	| CHAR SIZE NUMBER {
	    set_axis_prop(whichgraph, naxis, $1, $3);
	}
	| GRID onoff {
	    set_axis_prop(whichgraph, naxis, $1, $2);
	}
	;

axisfeature:
	TICKP tickdesc
	|  TICKLABEL ticklabeldesc
	|  LABEL axislabeldesc
	|  BAR axisbardesc
	|  onoff {
	    g[cg].t[naxis].active = $1;
	}
	;

tickdesc:
	tickattr
	|  tickdesc tickattr
	;

tickattr:
	onoff
	{
	    g[cg].t[naxis].t_flag = $1;
	    g[cg].t[naxis].t_mflag = $1;
	}
	| MAJOR onoff {
	    g[cg].t[naxis].t_flag = $2;
	}
	| MINOR onoff {
	    g[cg].t[naxis].t_mflag = $2;
	}
	| MAJOR expr {
	    g[cg].t[naxis].tmajor = $2;
	}
	| MINOR expr {
	    g[cg].t[naxis].tminor = $2;
	}
	| OFFSETX expr {
	    g[cg].t[naxis].offsx = $2;
	}
	| OFFSETY expr {
	    g[cg].t[naxis].offsy = $2;
	}
	| ALT onoff {
	    g[cg].t[naxis].alt = $2;
	}
	| MINP expr {
	    g[cg].t[naxis].tmin = $2;
	}
	| MAXP expr {
	    g[cg].t[naxis].tmax = $2;
	}
	| DEFAULT NUMBER {
	    g[cg].t[naxis].t_num = (int) $2;
	}
	| inoutchoice {
	    g[cg].t[naxis].t_inout = $1;
	}
	| LOG onoff {
	    g[cg].t[naxis].t_log = $2;
	}
	| SIZE NUMBER {
	    g[cg].t[naxis].t_size = $2;
	}
	| MAJOR SIZE NUMBER {
	    g[cg].t[naxis].t_size = $3;
	}
	| MINOR SIZE NUMBER {
	    g[cg].t[naxis].t_msize = $3;
	}
	| COLOR NUMBER {
	    g[cg].t[naxis].t_color = g[cg].t[naxis].t_mcolor = (int) $2;
	}
	| LINEWIDTH NUMBER {
	    g[cg].t[naxis].t_linew = g[cg].t[naxis].t_mlinew = (int) $2;
	}
	| MAJOR COLOR NUMBER {
	    g[cg].t[naxis].t_color = (int) $3;
	}
	| MINOR COLOR NUMBER {
	    g[cg].t[naxis].t_mcolor = (int) $3;
	}
	| MAJOR LINEWIDTH NUMBER {
	    g[cg].t[naxis].t_linew = (int) $3;
	}
	| MINOR LINEWIDTH NUMBER {
	    g[cg].t[naxis].t_mlinew = (int) $3;
	}
	| MAJOR LINESTYLE NUMBER {
	    g[cg].t[naxis].t_lines = (int) $3;
	}
	| MINOR LINESTYLE NUMBER {
	    g[cg].t[naxis].t_mlines = (int) $3;
	}
	| MAJOR GRID onoff {
	    g[cg].t[naxis].t_gridflag = $3;
	}
	| MINOR GRID onoff {
	    g[cg].t[naxis].t_mgridflag = $3;
	}
	| OP opchoice {
	    g[cg].t[naxis].t_op = $2;
	}
	| TYPE AUTO {
	    g[cg].t[naxis].t_type = AUTO;
	}
	| TYPE SPEC {
	    g[cg].t[naxis].t_type = SPEC;
	}
	| SPEC NUMBER {
	    g[cg].t[naxis].t_spec = (int) $2;
	}
	| NUMBER ',' expr {
	    g[cg].t[naxis].t_specloc[(int) $1] = $3;
	}
	;

ticklabeldesc:
	ticklabelattr
	|  ticklabeldesc ticklabelattr
	;

ticklabelattr:
	onoff {
	    g[cg].t[naxis].tl_flag = $1;
	}
	| TYPE AUTO {
	    g[cg].t[naxis].tl_type = AUTO;
	}
	| TYPE SPEC {
	    g[cg].t[naxis].tl_type = SPEC;
	}
	| PREC NUMBER {
	    g[cg].t[naxis].tl_prec = (int) $2;
	}
	| FORMAT formatchoice {
	    g[cg].t[naxis].tl_format = $2;
	}
	| FORMAT NUMBER {
	    g[cg].t[naxis].tl_format = (int)$2;
	}
	| APPEND CHRSTR {
	    strcpy(g[cg].t[naxis].tl_appstr, (char *) $2);
	    free((char *) $2);
	}
	| PREPEND CHRSTR {
	    strcpy(g[cg].t[naxis].tl_prestr, (char *) $2);
	    free((char *) $2);
	}
	| LAYOUT HORIZONTAL {
	    g[cg].t[naxis].tl_layout = HORIZONTAL;
	}
	| LAYOUT VERTICAL {
	    g[cg].t[naxis].tl_layout = VERTICAL;
	}
	| LAYOUT SPEC {
	    g[cg].t[naxis].tl_layout = SPEC;
	}
	| ANGLE NUMBER {
	    g[cg].t[naxis].tl_angle = (int) $2;
	}
	| JUST justchoice {
	    g[cg].t[naxis].tl_just = (int) $2;
	}
	| SKIP NUMBER {
	    g[cg].t[naxis].tl_skip = (int) $2;
	}
	| STAGGER NUMBER {
	    g[cg].t[naxis].tl_staggered = (int) $2;
	}
	| OP opchoice {
	    g[cg].t[naxis].tl_op = $2;
	}
	| SIGN signchoice {
	    g[cg].t[naxis].tl_sign = $2;
	}
	| PLOT_START expr {
	    g[cg].t[naxis].tl_start = $2;
	}
	| STOP expr {
	    g[cg].t[naxis].tl_stop = $2;
	}
	| PLOT_START TYPE SPEC {
	    g[cg].t[naxis].tl_starttype = (int) $3;
	}
	| PLOT_START TYPE AUTO {
	    g[cg].t[naxis].tl_starttype = (int) $3;
	}
	| STOP TYPE SPEC {
	    g[cg].t[naxis].tl_stoptype = (int) $3;
	}
	| STOP TYPE AUTO {
	    g[cg].t[naxis].tl_stoptype = (int) $3;
	}
	| VGAP NUMBER {
	    g[cg].t[naxis].tl_vgap = $2;
	}
	| HGAP NUMBER {
	    g[cg].t[naxis].tl_hgap = $2;
	}
	| CHAR SIZE NUMBER {
	    g[cg].t[naxis].tl_charsize = $3;
	}
	| FONTP NUMBER {
	    g[cg].t[naxis].tl_font = (int) $2;
	}
	| COLOR NUMBER {
	    g[cg].t[naxis].tl_color = (int) $2;
	}
	| LINEWIDTH NUMBER {
	    g[cg].t[naxis].tl_linew = (int) $2;
	}
	| NUMBER ',' CHRSTR {
	    set_plotstr_string(&g[cg].t[naxis].t_speclab[(int) $1], (char *) $3);
	    free((char *) $3);
	}
	;

axislabeldesc:
	CHRSTR {
	    set_plotstr_string(&g[cg].t[naxis].label, (char *) $1);
	    free((char *) $1);
	}
	| LAYOUT PERP {
	    g[cg].t[naxis].label_layout = PERP;
	}
	| LAYOUT PARA {
	    g[cg].t[naxis].label_layout = PARA;
	}
	| PLACE AUTO {
	    g[cg].t[naxis].label_place = $2;
	}
	| PLACE SPEC {
	    g[cg].t[naxis].label_place = $2;
	}
	| PLACE NUMBER ',' NUMBER {
	    g[cg].t[naxis].label.x = $2;
	    g[cg].t[naxis].label.y = $4;
	}
	| JUST justchoice {
	    g[cg].t[naxis].label.just = (int) $2;
	}
	| CHAR SIZE NUMBER {
	    g[cg].t[naxis].label.charsize = $3;
	}
	| FONTP NUMBER {
	    g[cg].t[naxis].label.font = (int) $2;
	}
	| COLOR NUMBER {
	    g[cg].t[naxis].label.color = (int) $2;
	}
	| LINEWIDTH NUMBER {
	    g[cg].t[naxis].label.linew = (int) $2;
	}
	;

axisbardesc:
	onoff {
	    g[cg].t[naxis].t_drawbar = $1;
	}
	| COLOR NUMBER {
	    g[cg].t[naxis].t_drawbarcolor = (int) $2;
	}
	| LINESTYLE NUMBER {
	    g[cg].t[naxis].t_drawbarlines = (int) $2;
	}
	| LINEWIDTH NUMBER {
	    g[cg].t[naxis].t_drawbarlinew = (int) $2;
	}
	;

selectsets:
	GRAPHNO '.' SETNUM
	{
	    whichgraph = $1;
	    whichset = $3;
	}
	| SETNUM
	{
	    whichgraph = cg;
	    whichset = $1;
	}
	|  SETS
	{
	    whichgraph = cg;
	    whichset = $1;
	}
	| GRAPHNO SETS
	{
	    whichgraph = $1;
	    whichset = $2;
	}
	|  GRAPHS SETS
	{
	    whichgraph = $1;
	    whichset = $2;
	}
	|  GRAPHS SETNUM
	{
	    whichgraph = $1;
	    whichset = $2;
	}
	;

prop:
        LINESTYLE {
            $$ = $1;
        }
        | LINEWIDTH {
            $$ = $1;
        }
        | FONTP {
            $$ = $1;
        }
        | COLOR {
            $$ = $1;
        }
        | SIZE {
            $$ = $1;
        }
        ;

onoff:
	ON {
	    $$ = ON;
	}
	| OFF {
	    $$ = OFF;
	}
	;

colpat:
	COLOR {
	    $$ = COLOR;
	}
	| PATTERN {
	    $$ = PATTERN;
	}
	;

runtype:
	RUNAVG {
	    $$ = RUNAVG;
	}
	| RUNSTD {
	    $$ = RUNSTD;
	}
	| RUNMED {
	    $$ = RUNMED;
	}
	| RUNMAX {
	    $$ = RUNMAX;
	}
	| RUNMIN {
	    $$ = RUNMIN;
	}
	;

ffttype:
	DFT {
	    $$ = DFT;
	}
	| FFT {
	    $$ = FFT;
	}
	| INVDFT {
	    $$ = INVDFT;
	}
	| INVFFT {
	    $$ = INVFFT;
	}
	;

sourcetype:
	DISK {
	    $$ = DISK;
	}
	| PIPE {
	    $$ = PIPE;
	}
	;

filltype:
	PATTERN {
	    $$ = PATTERN;
	}
	| COLOR {
	    $$ = COLOR;
	}
	| PLOT_NONE {
	    $$ = PLOT_NONE;
	}
	;

opchoice:
	TOP {
	    $$ = TOP;
	}
	| BOTTOM {
	    $$ = BOTTOM;
	}
	| LEFT {
	    $$ = LEFT;
	}
	| RIGHT {
	    $$ = RIGHT;
	}
	| BOTH {
	    $$ = BOTH;
	}
	;

justchoice:
	RIGHT {
	    $$ = RIGHT;
	}
	| LEFT {
	    $$ = LEFT;
	}
	| CENTER {
	    $$ = CENTER;
	}
	;

extremetype:
	MINP {
	    $$ = MINP;
	}
	| MAXP {
	    $$ = MAXP;
	}
	;

torf:
	TRUEP {
	    $$ = TRUEP;
	}
	| FALSEP {
	    $$ = FALSEP;
	}
	;

inoutchoice:
	IN {
	    $$ = IN;
	}
	| OUT {
	    $$ = OUT;
	}
	| BOTH {
	    $$ = BOTH;
	}
	;

formatchoice:
	DECIMAL {
	    $$ = DECIMAL;
	}
	| EXPONENTIAL {
	    $$ = EXPONENTIAL;
	}
	| POWER {
	    $$ = POWER;
	}
	| GENERAL {
	    $$ = GENERAL;
	}
	| DDMMYY {
	    $$ = DDMMYY;
	}
	| MMDDYY {
	    $$ = MMDDYY;
	}
	| YYMMDD {
	    $$ = YYMMDD;
	}
	| MMYY {
	    $$ = MMYY;
	}
	| MMDD {
	    $$ = MMDD;
	}
	| MONTHDAY {
	    $$ = MONTHDAY;
	}
	| DAYMONTH {
	    $$ = DAYMONTH;
	}
	| MONTHS {
	    $$ = MONTHS;
	}
	| MONTHL {
	    $$ = MONTHL;
	}
	| DAYOFWEEKS {
	    $$ = DAYOFWEEKS;
	}
	| DAYOFWEEKL {
	    $$ = DAYOFWEEKL;
	}
	| DAYOFYEAR {
	    $$ = DAYOFYEAR;
	}
	| HMS {
	    $$ = HMS;
	}
	| MMDDHMS {
	    $$ = MMDDHMS;
	}
	| MMDDYYHMS {
	    $$ = MMDDYYHMS;
	}
	| YYMMDDHMS {
	    $$ = YYMMDDHMS;
	}
	| DEGREESLON {
	    $$ = DEGREESLON;
	}
	| DEGREESMMLON {
	    $$ = DEGREESMMLON;
	}
	| DEGREESMMSSLON {
	    $$ = DEGREESMMSSLON;
	}
	| MMSSLON {
	    $$ = MMSSLON;
	}
	| DEGREESLAT {
	    $$ = DEGREESLAT;
	}
	| DEGREESMMLAT {
	    $$ = DEGREESMMLAT;
	}
	| DEGREESMMSSLAT {
	    $$ = DEGREESMMSSLAT;
	}
	| MMSSLAT {
	    $$ = MMSSLAT;
	}
	;

signchoice:
	NORMAL {
	    $$ = NORMAL;
	}
	| ABSOLUTE {
	    $$ = ABSOLUTE;
	}
	| NEGATE {
	    $$ = NEGATE;
	}
	;

direction:
	UP {
	    $$ = UP;
	}
	| DOWN {
	    $$ = DOWN;
	}
	| RIGHT {
	    $$ = RIGHT;
	}
	| LEFT {
	    $$ = LEFT;
	}
	| IN {
	    $$ = IN;
	}
	| OUT {
	    $$ = OUT;
	}
	;

worldview:
	WORLD {
	    $$ = WORLD;
	}
	| VIEW {
	    $$ = VIEW;
	}
	;

vector:
	PLOT_X { $$ = PLOT_X; }
	| PLOT_Y { $$ = PLOT_Y; }
	| X0 { $$ = X0; }
	| Y0 { $$ = Y0; }
	| Y1 { $$ = Y1; }
	| Y2 { $$ = Y2; }
	| Y3 { $$ = Y3; }
	| Y4 { $$ = Y4; }
	| Y5 { $$ = Y5; }
	;

asgn:
	VAR '[' expr ']' '=' expr
	{
	    int itmp = (int) $3 - 1;
	    if (itmp >= ls) {
		yyerror("Subscript out of range");
		return 1;
	    } else {
		$1[itmp] = $6;
		result = $6;
	    }
	}
	| vector '[' expr ']' '=' expr
	{
	    int itmp = (int) $3 - 1;
	    double *ptr = getvptr(cg, curset, (int)$3);
	    if (ptr != NULL) {
	        ptr[itmp] = $6;
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    result = $6;
	    updatesetminmax(cg, curset);
	    update_set_status(cg, curset);
	}
	| SETNUM '.' vector '[' expr ']' '=' expr
	{
	    int itmp = (int) $5 - 1;
	    double *ptr = getvptr(cg, $1, $3);
	    if (ptr != NULL) {
	        ptr[itmp] = $8;
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    result = $8;
	    updatesetminmax(cg, $1);
	    update_set_status(cg, $1);
	}
	| SETNUM '.' SYMBOL '=' expr
	{
	    set_prop(cg, SET, SETNUM, $1, SYMBOL, TYPE, (int) $5, 0);
	    result = 0;
	}
/*
	| SETS '.' SYMBOL '=' expr
	{
	    set_prop(cg, SET, SETNUM, $1, SYMBOL, TYPE, (int) $5, 0);
	    result = 0;
	}
*/
	| GRAPHNO '.' SETNUM '.' vector '[' expr ']' '=' expr
	{
	    int itmp = (int) $7 - 1;
	    double *ptr = getvptr($1, $3, $5);
	    if (ptr != NULL) {
	        ptr[itmp] = $10;
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    result = $10;
	    updatesetminmax($1, $3);
	    update_set_status($1, $3);
	}
	;

rasgn:
	REGNUM '.' vector '=' expr
	{
	    if ($3 == PLOT_X) {
		*xx = $5;
	    } else {
		*yy = $5;
	    }
	}
	;

vasgn:
	VAR '=' vexpr
	{
	    int i;
	    for (i = 0; i < lxy; i++) {
		$1[i] = $3[i];
	    }
	    result = $3[0];
	}
	| vector '=' vexpr
	{
	    int i;
	    double *ptr;
	    if (!isactive_set(cg, curset)) {
		activateset(cg, curset);
		setlength(cg, curset, lxy);
		setcomment(cg, curset, "Created");
	    }
	    ptr = getvptr(cg, curset, $1);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $3[i];
	        }
	        result = $3[0];
	        updatesetminmax(cg, curset);
	        update_set_status(cg, curset);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| SETNUM '.' vector '=' vexpr {
	    int i;
	    double *ptr;
	    if (!isactive_set(cg, $1)) {
		activateset(cg, $1);
		setlength(cg, $1, lxy);
		setcomment(cg, $1, "Created");
	    }
	    ptr = getvptr(cg, $1, $3);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $5[i];
	        }
	        result = $5[0];
	        updatesetminmax(cg, $1);
	        update_set_status(cg, $1);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| GRAPHNO '.' SETNUM '.' vector '=' vexpr
	{
	    int i;
	    double *ptr;
	    if (!isactive_set($1, $3)) {
		activateset($1, $3);
		setlength($1, $3, lxy);
		setcomment($1, $3, "Created");
	    }
	    ptr = getvptr($1, $3, $5);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $7[i];
	        }
	        result = $7[0];
	        updatesetminmax($1, $3);
	        update_set_status($1, $3);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| VAR '=' expr
	{
	    int i;
	    for (i = 0; i < lxy; i++) {
		$1[i] = $3;
	    }
	    result = $3;
	}
	| vector '=' expr
	{
	    int i;
	    double *ptr;
	    if (!isactive_set(cg, curset)) {
		activateset(cg, curset);
		setlength(cg, curset, lxy);
		setcomment(cg, curset, "Created");
	    }
	    ptr = getvptr(cg, curset, $1);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $3;
	        }
	        result = $3;
	        updatesetminmax(cg, curset);
	        update_set_status(cg, curset);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| SETNUM '.' vector '=' expr
	{
	    int i;
	    double *ptr;
	    if (!isactive_set(cg, $1)) {
		activateset(cg, $1);
		setlength(cg, $1, lxy);
		setcomment(cg, $1, "Created");
	    }
	    ptr = getvptr(cg, $1, $3);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $5;
	        }
	        result = $5;
	        updatesetminmax(cg, $1);
	        update_set_status(cg, $1);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| GRAPHNO '.' SETNUM '.' vector '=' expr
	{
	    int i;
	    double *ptr;
	    if (!isactive_set($1, $3)) {
		activateset($1, $3);
		setlength($1, $3, lxy);
		setcomment($1, $3, "Created");
	    }
	    ptr = getvptr($1, $3, $5);
	    if (ptr != NULL) {
	        for (i = 0; i < lxy; i++) {
		    ptr[i] = $7;
	        }
	        result = $7;
	        updatesetminmax($1, $3);
	        update_set_status($1, $3);
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	;

vexpr:
	VAR
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i];
	    }
	}
	| vector
	{
	    int i;
	    double *ptr = getvptr(cg, curset, $1);
	    if (ptr == NULL) {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = ptr[i];
	    }
	}
	| SETNUM '.' vector
	{
	    int i;
	    double *ptr = getvptr(cg, $1, $3);
	    if (ptr == NULL) {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = ptr[i];
	    }
	}
	| GRAPHNO '.' SETNUM '.' vector
	{
	    int i;
	    double *ptr = getvptr($1, $3, $5);
	    if (ptr == NULL) {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = ptr[i];
	    }
	}
	| expr '+' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 + $3;
	    }
	}
	| vexpr '+' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] + $3[i];
	    }
	}
	| expr '+' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 + $3[i];
	    }
	}
	| vexpr '+' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] + $3;
	    }
	}
	| expr '-' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 - $3;
	    }
	}
	| vexpr '-' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] - $3[i];
	    }
	}
	| expr '-' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 - $3[i];
	    }
	}
	| vexpr '-' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] - $3;
	    }
	}
	| expr '*' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 * $3;
	    }
	}
	| vexpr '*' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] * $3[i];
	    }
	}
	| expr '*' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 * $3[i];
	    }
	}
	| vexpr '*' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] * $3;
	    }
	}
	| expr '/' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    if ($3 == 0.0) {
		yyerror("Divide by Zero");
		return 1;
	    }
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 / $3;
	    }
	}
	| vexpr '/' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		if ($3[i] == 0.0) {
		    yyerror("Divide by Zero");
		    return 1;
		}
	    }
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] / $3[i];
	    }
	}
	| expr '/' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		if ($3[i] == 0.0) {
		    yyerror("Divide by Zero");
		    return 1;
		}
	    }
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1 / $3[i];
	    }
	}
	| vexpr '/' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    if ($3 == 0.0) {
		yyerror("Divide by Zero");
		return 1;
	    }
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] / $3;
	    }
	}
	| expr '^' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = pow($1, $3);
	    }
	}
	| expr '^' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = pow($1, $3[i]);
	    }
	}
	| vexpr '^' expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = pow($1[i], $3);
	    }
	}
	| vexpr '^' vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = pow($1[i], $3[i]);
	    }
	}
	| ABS '(' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = fabs($3);
	    }
	}
	| ABS '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = fabs($3[i]);
	    }
	}
	| ACOS '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = acos($3[i]);
	    }
	}
	| ASIN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = asin($3[i]);
	    }
	}
	| ATAN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = atan($3[i]);
	    }
	}
	| ATAN2 '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = atan2($3[i], $5[i]);
	    }
	}
	| CEIL '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = ceil($3[i]);
	    }
	}
	| COS '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = cos($3[i]);
	    }
	}
	| expr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1;
	    }
	}
	| DEG
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] *= M_PI / 180.0;
	    }
	}
	| ERF '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = erf($3[i]);
	    }
	}
	| ERFC '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = erfc($3[i]);
	    }
	}
	| EXP '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = exp($3[i]);
	    }
	}
	| FLOOR '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = floor($3[i]);
	    }
	}
	| HYPOT '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = my_hypot($3[i], $5[i]);
	    }
	}
	| HYPOT '(' expr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = my_hypot($3, $5[i]);
	    }
	}
	| HYPOT '(' vexpr ',' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = my_hypot($3[i], $5);
	    }
	}
	| HYPOT '(' expr ',' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = my_hypot($3, $5);
	    }
	}
	| INDEX
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = i + 1;
	    }
	}
	| SETNO
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1;
	    }
	}
	| INT '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = (int) $3[i];
	    }
	}
	| INVN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = invnorm($3[i]);
	    }
	}
	| INVT '(' expr ',' NUMBER ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = invt($3, (int) $5);
	    }
	}
	| INVT '(' vexpr ',' NUMBER ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = invt($3[i], (int) $5);
	    }
	}
	| IRAND '(' NUMBER ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = lrand48() % (long) ($3);
	    }
	}
	| LGAMMA '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = lgamma($3[i]);
	    }
	}
	| LN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = log($3[i]);
	    }
	}
	| LOG '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = log10($3[i]);
	    }
	}
	| LOGISTIC '(' vexpr ',' expr ',' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = 1.0 / (1.0 + exp(-($3[i] - $5)/ $7));
	    }
	}
	| MAXP '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $3[i] >= $5[i] ? $3[i] : $5[i];
	    }
	}
	| MINP '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $3[i] <= $5[i] ? $3[i] : $5[i];
	    }
	}
	| MOD '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = fmod($3[i], $5[i]);
	    }
	}
	| NORM '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = fx($3[i]);
	    }
	}
	| NORMP '(' vexpr ')'
	{
	    int i;
	    double tmp;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = normp($3[i], &tmp);
	    }
	}
	| PI
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = M_PI;
	    }
	}
	| RAD
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = M_PI / 180.0;
	    }
	}
	| RAND
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = (double) drand48();
	    }
	}
	| RNORM '(' vexpr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = rnorm($3[i], $5[i]);
	    }
	}
	| RNORM '(' expr ',' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = rnorm($3, $5[i]);
	    }
	}
	| RNORM '(' vexpr ',' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = rnorm($3[i], $5);
	    }
	}
	| RNORM '(' expr ',' expr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = rnorm($3, $5);
	    }
	}
	| SIN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = sin($3[i]);
	    }
	}
	| SQR '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $3[i] * $3[i];
	    }
	}
	| SQRT '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = sqrt($3[i]);
	    }
	}
	| TAN '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = tan($3[i]);
	    }
	}
	| vexpr '?' vexpr ':' vexpr {
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
	        if ((int) $1[i]) {
		    $$[i] = $3[i];
	        } else {
		    $$[i] = $5[i];
	        }
	    }
	}
	| vexpr GT vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] > $3[i];
	    }
	}
	| vexpr LT vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] < $3[i];
	    }
	}
	| vexpr LE vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] <= $3[i];
	    }
	}
	| vexpr GE vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] >= $3[i];
	    }
	}
	| vexpr EQ vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] == $3[i];
	    }
	}
	| vexpr NE vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] != $3[i];
	    }
	}
	| vexpr AND vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] && $3[i];
	    }
	}
	| vexpr OR vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $1[i] || $3[i];
	    }
	}
	| NOT vexpr
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = !($2[i]);
	    }
	}
	| '(' vexpr ')'
	{
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = $2[i];
	    }
	}
	| '-' vexpr %prec UMINUS {
	    int i;
	    $$ = (double *) calloc(lxy, sizeof(double));
	    freelist[fcnt++] = $$;
	    for (i = 0; i < lxy; i++) {
		$$[i] = -$2[i];
	    }
	}
	;

expr:	NUMBER
	|  FITPARM {
	    $$ = $1;
	}
	|  VAR '[' expr ']' {
	    $$ = $1[(int) $3];
	}
	| REGNUM '.' vector {
	    $$ = ($3 == PLOT_X) ? *xx : *yy;
	}
	| vector '[' expr ']' {
	    double *ptr = getvptr(cg, curset, $1);
	    if (ptr != NULL) {
		$$ = ptr[(int) $3 - 1];
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| SETNUM '.' vector '[' expr ']' {
	    double *ptr = getvptr(cg, $1, $3);
	    if (ptr != NULL) {
		$$ = ptr[(int) $5 - 1];
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| GRAPHNO '.' SETNUM '.' vector '[' expr ']' {
	    double *ptr = getvptr($1, $3, $5);
	    if (ptr != NULL) {
		$$ = ptr[(int) $7 - 1];
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| SETNUM '.' vector '.' extremetype {
	    double *ptr = getvptr(cg, $1, $3);
	    if (ptr == NULL) {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    switch ($5) {
	    case MINP:
		$$ = vmin(ptr, g[cg].p[$1].len);
		break;
	    case MAXP:
		$$ = vmax(ptr, g[cg].p[$1].len);
		break;
	    }
	}
	| GRAPHNO '.' SETNUM '.' vector '.' extremetype {
	    double *ptr = getvptr($1, $3, $5);
	    if (ptr == NULL) {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	    switch ($7) {
	    case MINP:
		$$ = vmin(ptr, g[$1].p[$3].len);
		break;
	    case MAXP:
		$$ = vmax(ptr, g[$1].p[$3].len);
		break;
	    }
	}
	| SETNUM '.' LENGTH {
	    $$ = g[cg].p[$1].len;
	}
	| SETNUM '.' vector '.' AVG
	{
	    double bar, sd;
	    double *ptr = getvptr(cg, $1, $3);
	    if (ptr != NULL) {
		stasum(ptr, getsetlength(cg, $1), &bar, &sd, 0);
	        $$ = bar;
	    }
	    else {
		yyerror("NULL variable, check set type");
		return 1;
	    }
	}
	| expr '+' expr {
	    $$ = $1 + $3;
	}
	| expr '-' expr {
	    $$ = $1 - $3;
	}
	| expr '*' expr {
	    $$ = $1 * $3;
	}
	| expr '/' expr
	{
	    if ($3 != 0.0) {
		$$ = $1 / $3;
	    } else {
		yyerror("Divide by Zero");
		return 1;
	    }
	}
	| expr '%' expr {
	    $$ = fmod($1, $3);
	}
	| expr '^' expr {
	    $$ = pow($1, $3);
	}
	| ABS '(' expr ')' {
	    $$ = fabs($3);
	}
	| ACOS '(' expr ')' {
	    $$ = acos($3);
	}
	| ASIN '(' expr ')' {
	    $$ = asin($3);
	}
	| ATAN '(' expr ')' {
	    $$ = atan($3);
	}
	| ATAN2 '(' expr ',' expr ')' {
	    $$ = atan2($3, $5);
	}
	| CEIL '(' expr ')' {
	    $$ = ceil($3);
	}
	| COS '(' expr ')' {
	    $$ = cos($3);
	}
	| DEG {
	    $$ = 180.0 / M_PI;
	}
	| ERF '(' expr ')' {
	    $$ = erf($3);
	}
	| ERFC '(' expr ')' {
	    $$ = erfc($3);
	}
	| EXP '(' expr ')' {
	    $$ = exp($3);
	}
	| FLOOR '(' expr ')' {
	    $$ = floor($3);
	}
	| HYPOT '(' expr ',' expr ')' {
	    $$ = my_hypot($3, $5);
	}
	| GRAPHNO '.' VX1 {
	    $$ = g[$1].v.xv1;
	}
	| GRAPHNO '.' VX2 {
	    $$ = g[$1].v.xv2;
	}
	| GRAPHNO '.' VY1 {
	    $$ = g[$1].v.yv1;
	}
	| GRAPHNO '.' VY2 {
	    $$ = g[$1].v.yv2;
	}
	| GRAPHNO '.' WX1 {
	    $$ = g[$1].w.xg1;
	}
	| GRAPHNO '.' WX2 {
	    $$ = g[$1].w.xg2;
	}
	| GRAPHNO '.' WY1 {
	    $$ = g[$1].w.yg1;
	}
	| GRAPHNO '.' WY2 {
	    $$ = g[$1].w.yg2;
	}
	| VX1 {
	    $$ = g[cg].v.xv1;
	}
	| VX2 {
	    $$ = g[cg].v.xv2;
	}
	| VY1 {
	    $$ = g[cg].v.yv1;
	}
	| VY2 {
	    $$ = g[cg].v.yv2;
	}
	| WX1 {
	    $$ = g[cg].w.xg1;
	}
	| WX2 {
	    $$ = g[cg].w.xg2;
	}
	| WY1 {
	    $$ = g[cg].w.yg1;
	}
	| WY2 {
	    $$ = g[cg].w.yg2;
	}
	| DX {
	    $$ = g[cg].w.xg2 - g[cg].w.xg1;
	}
	| DY {
	    $$ = g[cg].w.yg2 - g[cg].w.yg1;
	}
	| INDEX {
	    $$ = setindex;
	}
	| SETNO {
	    $$ = setsetno;
	}
	| INT '(' expr ')' {
	    $$ = (long) $3;
	}
	| INVN '(' expr ')' {
	    $$ = invnorm($3);
	}
	| INVT '(' expr ',' NUMBER ')' {
	    $$ = invt($3, (int) $5);
	}
	| IRAND '(' NUMBER ')' {
	    $$ = lrand48() % (long) ($3);
	}
	| LGAMMA '(' expr ')' {
	    $$ = lgamma($3);
	}
	| LN '(' expr ')' {
	    $$ = log($3);
	}
	| LOG '(' expr ')' {
	    $$ = log10($3);
	}
	| LOGISTIC '(' expr ',' expr ',' expr ')'
	{
	    $$ = 1.0 / (1.0 + exp(-($3 - $5)/ $7));
	}
	| MAXP '(' expr ',' expr ')' {
	    $$ = $3 >= $5 ? $3 : $5;
	}
	| MINP '(' expr ',' expr ')' {
	    $$ = $3 <= $5 ? $3 : $5;
	}
	| MOD '(' expr ',' expr ')' {
	    $$ = fmod($3, $5);
	}
	| NORM '(' expr ')' {
	    $$ = fx($3);
	}
	| NORMP '(' expr ')' {
	    double tmp;
	    $$ = normp($3, &tmp);
	}
	| PI {
	    $$ = M_PI;
	}
	| RAD {
	    $$ = M_PI / 180.0;
	}
	| RAND {
	    $$ = (double) drand48();
	}
	| RNORM '(' expr ',' expr ')' {
	    $$ = rnorm($3, $5);
	}
	| SIN '(' expr ')' {
	    $$ = sin($3);
	}
	| SQR '(' expr ')' {
	    $$ = pow($3, 2.0);
	}
	| SQRT '(' expr ')' {
	    $$ = sqrt($3);
	}
	| TAN '(' expr ')' {
	    $$ = tan($3);
	}
	| expr '?' expr ':' expr {
	    if ((int) $1) {
		$$ = $3;
	    } else {
		$$ = $5;
	    }
	}
	| expr GT expr {
	    $$ = $1 > $3;
	}
	| expr LT expr {
	    $$ = $1 < $3;
	}
	| expr LE expr {
	    $$ = $1 <= $3;
	}
	| expr GE expr {
	    $$ = $1 >= $3;
	}
	| expr EQ expr {
	    $$ = $1 == $3;
	}
	| expr NE expr {
	    $$ = $1 != $3;
	}
	| expr AND expr {
	    $$ = $1 && $3;
	}
	| expr OR expr {
	    $$ = $1 || $3;
	}
	| NOT expr {
	    $$ = !($2);
	}
	| '(' expr ')' {
	    $$ = $2;
	}
	| '-' expr %prec UMINUS {
	    $$ = -$2;
	}
	;
%%

void fixupstr(char *val)
{
    int vl = strlen(val);
    lowtoupper(val);
    val[vl + 1] = 0;
    val[vl] = '\n';
}

void scanner(char *s, double *x, double *y, int len, double *a, double *b, double *c, double *d, int lenscr, int i, int setno, int *errpos)
{
    interr = 0;
    whichgraph = cg;
    whichset = setno;
    if (s[0] == '#') {
	return;
    }
    pos = 0;
    aa = a;
    bb = b;
    cc = c;
    dd = d;
    xx = x;
    yy = y;
    lxy = len;
    ls = lenscr;
    setindex = i + 1;
    curset = setsetno = setno;
    strcpy(f_string, s);
    fcnt = 0;
    log_results(s);
    yyparse();
    *errpos = interr;
    for (i = 0; i < fcnt; i++) {
	free(freelist[i]);
	freelist[i] = NULL;
    }
}

void runbatch(char *bfile)
{
    double x, y;
    int errpos;
    char stext[256];
    FILE *fp;
    if (strcmp("stdin", bfile)) {
	fp = fopen(bfile, "r");
    }
    else {
	fp = stdin;
    }
    if (fp == NULL) {
        fprintf(stderr, "Error opening batch file \"%s\"\n", bfile);
        exit(1);
    }
    while(fgets(stext, 255, fp) != NULL) {
        if (stext[0] == '#') {
            continue;
        }
	if (strlen(stext) == 0) {
	    continue;
	}
        lowtoupper(stext);
/* TODO check on 0, 0 here for index and setno */
        scanner(stext, &x, &y, 1, ax, bx, cx, dx, 1, 0, 0, &errpos);
        stext[0] = 0;
        if (gotparams && paramfile[0]) {
            if (!getparms(cg, paramfile)) {
            }
            gotparams = 0;
        } else if (gotread && readfile[0]) {
            if (getdata(cg, readfile, readsrc, readtype)) {
                drawgraph();
            }
            gotread = 0;
        }
    }
    if (fp != stdin) {
	fclose(fp);
    }
}

#define MAXFUN 384

int maxparms = MAXFUN;
int maxfunc = MAXFUN;

symtab_entry key[] = {
	{"A", VAR},
	{"A0", FITPARM},
	{"A1", FITPARM},
	{"A2", FITPARM},
	{"A3", FITPARM},
	{"A4", FITPARM},
	{"A5", FITPARM},
	{"A6", FITPARM},
	{"A7", FITPARM},
	{"A8", FITPARM},
	{"A9", FITPARM},
	{"ABORT", ABORT},
	{"ABOVE", ABOVE},
	{"ABS", ABS},
	{"ABSOLUTE", ABSOLUTE},
	{"ACOS", ACOS},
	{"ACTIVATE", ACTIVATE},
	{"ACTIVE", ACTIVE},
	{"ALT", ALT},
	{"ALTERNATE", ALTERNATE},
	{"ALTYAXIS", ALTYAXIS},
	{"AND", AND},
	{"ANGLE", ANGLE},
	{"ANNOTATE", ANNOTATE},
	{"ARROW", ARROW},
	{"ASIN", ASIN},
	{"ATAN", ATAN},
	{"AUTOSCALE", AUTOSCALE},
	{"AVG", AVG},
	{"AXES", AXES},
	{"AXIS", AXIS},
	{"B", VAR},
	{"BACKBUFFER", BACKBUFFER},
	{"BAR", BAR},
	{"BATCH", BATCH},
	{"BELOW", BELOW},
	{"BIN", BIN},
	{"BOTH", BOTH},
	{"BOTTOM", BOTTOM},
	{"BOX", BOX},
	{"BOXPLOT", BOXPLOT},
	{"C", VAR},
	{"CD", CD},
	{"CEIL", CEIL},
	{"CELLS", CELLS},
	{"CENTER", CENTER},
	{"CHAR", CHAR},
	{"CHRSTR", CHRSTR},
	{"CLEAR", CLEAR},
	{"CLICK", CLICK},
	{"CMAP", CMAP},
	{"CO", COLOR},
	{"COLOR", COLOR},
	{"COMMENT", COMMENT},
	{"COPY", COPY},
	{"COS", COS},
	{"CYCLE", CYCLE},
	{"D", VAR},
	{"DAYMONTH", DAYMONTH},
	{"DAYOFWEEKL", DAYOFWEEKL},
	{"DAYOFWEEKS", DAYOFWEEKS},
	{"DAYOFYEAR", DAYOFYEAR},
	{"DDMMYY", DDMMYY},
	{"DECIMAL", DECIMAL},
	{"DEF", DEF},
	{"DEG", DEG},
	{"DEGREESLAT", DEGREESLAT},
	{"DEGREESLON", DEGREESLON},
	{"DEGREESMMLAT", DEGREESMMLAT},
	{"DEGREESMMLON", DEGREESMMLON},
	{"DEGREESMMSSLAT", DEGREESMMSSLAT},
	{"DEGREESMMSSLON", DEGREESMMSSLON},
	{"DELAY", DELAYP},
	{"DELETE", DELETE},
	{"DEVICE", DEVICE},
	{"DIFF", DIFFERENCE},
	{"DIFFERENCE", DIFFERENCE},
	{"DISK", DISK},
	{"DOUBLEBUFFER", DOUBLEBUFFER},
	{"DOWN", DOWN},
	{"DRAW2", DRAW2},
	{"DROP", DROP},
	{"DX", DX},
	{"DXDX", DXDX},
	{"DY", DY},
	{"ECHO", ECHO},
	{"EDIT", EDIT},
	{"END", END},
	{"EQ", EQ},
	{"ER", ERRORBAR},
	{"ERF", ERF},
	{"ERFC", ERFC},
	{"ERRORBAR", ERRORBAR},
	{"EXIT", EXIT},
	{"EXP", EXP},
	{"EXPONENTIAL", EXPONENTIAL},
	{"FALSE", FALSEP},
	{"FFT", FFT},
	{"FILE", FILEP},
	{"FILL", FILL},
	{"FIND", FIND},
	{"FIXED", FIXED},
	{"FIXEDPOINT", FIXEDPOINT},
	{"FLOOR", FLOOR},
	{"FLUSH", FLUSH},
	{"FOCUS", FOCUS},
	{"FOLLOWS", FOLLOWS},
	{"FONT", FONTP},
	{"FOREGROUND", FOREGROUND},
	{"FORMAT", FORMAT},
	{"FRAME", FRAMEP},
	{"FREE", FREE},
	{"FRONTBUFFER", FRONTBUFFER},
	{"GE", GE},
	{"GENERAL", GENERAL},
	{"GETP", GETP},
	{"GRAPH", GRAPH},
	{"GRAPHS", GRAPHS},
	{"GRAPHTYPE", GRAPHTYPE},
	{"GRID", GRID},
	{"GT", GT},
	{"HARDCOPY", HARDCOPY},
	{"HBAR", HBAR},
	{"HBOXPLOT", HBOXPLOT},
	{"HGAP", HGAP},
	{"HIDDEN", HIDDEN},
	{"HISTO", HISTO},
	{"HMS", HMS},
	{"HORIZONTAL", HORIZONTAL},
	{"HPGLL", HPGLL},
	{"HPGLP", HPGLP},
	{"HYPOT", HYPOT},
	{"IF", IF},
	{"IGNORE", IGNORE},
	{"IHL", IHL},
	{"IMAGE", IMAGE},
	{"IN", IN},
	{"INDEX", INDEX},
	{"INIT", PLOT_INIT},
	{"INOUT", INOUT},
	{"INT", INT},
	{"INTEGRATE", INT},
	{"INTERP", INTERP},
	{"INUM", INUM},
	{"INVDFT", INVDFT},
	{"INVFFT", INVFFT},
	{"INVN", INVN},
	{"INVT", INVT},
	{"IRAND", IRAND},
	{"JUST", JUST},
	{"KILL", KILL},
	{"LABEL", LABEL},
	{"LANDSCAPE", LANDSCAPE},
	{"LAYOUT", LAYOUT},
	{"LE", LE},
	{"LEAVE", LEAVE},
	{"LEFT", LEFT},
	{"LEGEND", LEGEND},
	{"LENGTH", LENGTH},
	{"LEVEL", LEVEL},
	{"LEVELS", LEVELS},
	{"LGAMMA", LGAMMA},
	{"LINE", LINE},
	{"LINESTYLE", LINESTYLE},
	{"LINETO", LINETO},
	{"LINEWIDTH", LINEWIDTH},
	{"LINK", LINK},
	{"LN", LN},
	{"LOAD", LOAD},
	{"LOCATOR", LOCATOR},
	{"LOCATORBAR", LOCATORBAR},
	{"LOCTYPE", LOCTYPE},
	{"LOG", LOG},
	{"LOGISTIC", LOGISTIC},
	{"LOGX", LOGX},
	{"LOGXY", LOGXY},
	{"LOGY", LOGY},
	{"LS", LINESTYLE},
	{"LT", LT},
	{"LW", LINEWIDTH},
	{"MAJOR", MAJOR},
	{"MAX", MAXP},
	{"MIFL", MIFL},
	{"MIFP", MIFP},
	{"MIN", MINP},
	{"MINOR", MINOR},
	{"MISSING", MISSINGP},
	{"MMDD", MMDD},
	{"MMDDHMS", MMDDHMS},
	{"MMDDYY", MMDDYY},
	{"MMDDYYHMS", MMDDYYHMS},
	{"MMSSLAT", MMSSLAT},
	{"MMSSLON", MMSSLON},
	{"MMYY", MMYY},
	{"MOD", MOD},
	{"MONTHDAY", MONTHDAY},
	{"MONTHL", MONTHL},
	{"MONTHS", MONTHS},
	{"MOVE", MOVE},
	{"MOVE2", MOVE2},
	{"MOVETO", MOVETO},
	{"NE", NE},
	{"NEGATE", NEGATE},
	{"NO", NO},
	{"NONE", PLOT_NONE},
	{"NORM", NORM},
	{"NORMAL", NORMAL},
	{"NORMP", NORMP},
	{"NOT", NOT},
	{"NUMBER", NUMBER},
	{"NXY", NXY},
	{"OFF", OFF},
	{"OFFSETX", OFFSETX},
	{"OFFSETY", OFFSETY},
	{"ON", ON},
	{"OP", OP},
	{"OR", OR},
	{"ORIENT", ORIENT},
	{"OUT", OUT},
	{"PAGE", PAGE},
	{"PARA", PARA},
	{"PARALLEL", PARALLEL},
	{"PARAMETERS", PARAMETERS},
	{"PARAMS", PARAMS},
	{"PATTERN", PATTERN},
	{"PERIMETER", PERIMETER},
	{"PERP", PERP},
	{"PERPENDICULAR", PERPENDICULAR},
	{"PI", PI},
	{"PIE", PIE},
	{"PIPE", PIPE},
	{"PLACE", PLACE},
	{"POINT", POINT},
	{"POLAR", POLAR},
	{"POLYI", POLYI},
	{"POLYO", POLYO},
	{"POP", POP},
	{"PORTRAIT", PORTRAIT},
	{"POWER", POWER},
	{"PREC", PREC},
	{"PREPEND", PREPEND},
	{"PRINT", PRINT},
	{"PS", PS},
	{"PSCOLORL", PSCOLORL},
	{"PSCOLORP", PSCOLORP},
	{"PSMONOL", PSMONOL},
	{"PSMONOP", PSMONOP},
	{"PUSH", PUSH},
	{"PUTP", PUTP},
	{"RAND", RAND},
	{"RAWSPICE", RAWSPICE},
	{"READ", READ},
	{"REDRAW", REDRAW},
	{"RENDER", PLOT_RENDER},
	{"REVERSE", REVERSE},
	{"RIGHT", RIGHT},
	{"RNORM", RNORM},
	{"ROT", ROT},
	{"RUNAVG", RUNAVG},
	{"RUNMAX", RUNMAX},
	{"RUNMED", RUNMED},
	{"RUNMIN", RUNMIN},
	{"RUNSTD", RUNSTD},
	{"SAMPLE", SAMPLE},
	{"SAVEALL", SAVEALL},
	{"SCALE", SCALE},
	{"SCIENTIFIC", SCIENTIFIC},
	{"SET", SET},
	{"SETNO", SETNO},
	{"SETS", SETS},
	{"SIGN", SIGN},
	{"SIN", SIN},
	{"SIZE", SIZE},
	{"SKIP", SKIP},
	{"SLEEP", SLEEP},
	{"SLICE", SLICE},
	{"SOURCE", SOURCE},
	{"SPEC", SPEC},
	{"SPECIFIED", SPECIFIED},
	{"SPECTRUM", SPECTRUM},
	{"SPLINE", SPLINE},
	{"SQR", SQR},
	{"SQRT", SQRT},
	{"STACK", STACK},
	{"STACKEDBAR", STACKEDBAR},
	{"STACKEDHBAR", STACKEDHBAR},
	{"STACKEDLINE", STACKEDLINE},
	{"STAGGER", STAGGER},
	{"START", PLOT_START},
	{"STARTTYPE", STARTTYPE},
	{"STATUSBAR", STATUSBAR},
	{"STOP", STOP},
	{"STRING", PLOT_STRING},
	{"SUBTITLE", SUBTITLE},
	{"SWAPBUFFER", SWAPBUFFER},
	{"SYMBOL", SYMBOL},
	{"TAN", TAN},
	{"TICK", TICKP},
	{"TICKLABEL", TICKLABEL},
	{"TICKMARKS", TICKMARKS},
	{"TITLE", TITLE},
	{"TO", TO},
	{"TOOLBAR", TOOLBAR},
	{"TOP", TOP},
	{"TRUE", TRUEP},
	{"TYPE", TYPE},
	{"UP", UP},
	{"VAR", VAR},
	{"VERTICAL", VERTICAL},
	{"VGAP", VGAP},
	{"VIEW", VIEW},
	{"VX1", VX1},
	{"VX2", VX2},
	{"VY1", VY1},
	{"VY2", VY2},
	{"WITH", WITH},
	{"WORLD", WORLD},
	{"WRITE", WRITE},
	{"WX1", WX1},
	{"WX2", WX2},
	{"WY1", WY1},
	{"WY2", WY2},
	{"X", PLOT_X},
	{"X0", X0},
	{"X1", X1},
	{"XAXES", XAXES},
	{"XAXIS", XAXIS},
	{"XCOR", XCOR},
	{"XMAX", XMAX},
    {"XMIN", XMIN},
	{"XY", XY},
	{"XYARC", XYARC},
	{"XYBOX", XYBOX},
	{"XYBOXPLOT", XYBOXPLOT},
	{"XYDX", XYDX},
	{"XYDXDX", XYDXDX},
	{"XYDXDY", XYDXDY},
	{"XYDY", XYDY},
	{"XYDYDY", XYDYDY},
	{"XYFIXED", XYFIXED},
	{"XYHILO", XYHILO},
	{"XYRT", XYRT},
	{"XYSEG", XYSEG},
	{"XYSTRING", XYSTRING},
	{"XYUV", XYUV},
	{"XYX2Y2", XYX2Y2},
	{"XYXX", XYXX},
	{"XYYY", XYYY},
	{"XYZ", XYZ},
	{"XYZW", XYZW},
	{"Y", PLOT_Y},
	{"Y0", Y0},
	{"Y1", Y1},
	{"Y2", Y2},
	{"Y3", Y3},
	{"Y4", Y4},
	{"Y5", Y5},
	{"YAXES", YAXES},
	{"YAXIS", YAXIS},
	{"YES", YES},
	{"YMAX", YMAX},
	{"YMIN", YMIN},
	{"YYMMDD", YYMMDD},
	{"YYMMDDHMS", YYMMDDHMS},
	{"ZEROXAXIS", ZEROXAXIS},
	{"ZEROYAXIS", ZEROYAXIS}
};

int findf(symtab_entry *key, char *s, int tlen)
{

    int low, high, mid;

    low = 0;
    high = tlen - 1;
    while (low <= high) {
	mid = (low + high) / 2;
	if (strcmp(s, key[mid].s) < 0) {
	    high = mid - 1;
	} else {
	    if (strcmp(s, key[mid].s) > 0) {
		low = mid + 1;
	    } else {
		return (mid);
	    }
	}
    }
    return (-1);
}

int getcharstr(void)
{
    if (pos >= strlen(f_string))
	 return EOF;
    return (f_string[pos++]);
}

void ungetchstr(void)
{
    if (pos > 0)
	pos--;
}

int yylex(void)
{
    int c, i;
    int found;
    static char s[256];
    char sbuf[256];
    char *str;

    while ((c = getcharstr()) == ' ' || c == '\t')
       ;
    if (c == EOF) {
	return (0);
    }
    if (c == '"') {
	i = 0;
	while ((c = getcharstr()) != '"' && c != EOF) {
	    if (c == '\\') {
		int ctmp;
		ctmp = getcharstr();
		if (ctmp != '"') {
		    ungetchstr();
		}
		else {
		    c = ctmp;
		}
	    }
	    s[i] = c;
	    i++;
	}
	if (c == EOF) {
	    yyerror("Nonterminating string");
	    return 0;
	}
	s[i] = '\0';
	str = (char *) malloc(strlen(s) + 1);
	strcpy(str, s);
	yylval.str = str;
	return CHRSTR;
    }
    if (c == '.' || isdigit(c)) {
	char stmp[80];
	double d;
	int i, gotdot = 0;

	i = 0;
	while (c == '.' || isdigit(c)) {
	    if (c == '.') {
		if (gotdot) {
		    yyerror("Reading number, too many dots");
	    	    return 0;
		} else {
		    gotdot = 1;
		}
	    }
	    stmp[i++] = c;
	    c = getcharstr();
	}
	if (c == 'E' || c == 'e') {
	    stmp[i++] = c;
	    c = getcharstr();
	    if (c == '+' || c == '-') {
		stmp[i++] = c;
		c = getcharstr();
	    }
	    while (isdigit(c)) {
		stmp[i++] = c;
		c = getcharstr();
	    }
	}
	if (gotdot && i == 1) {
	    ungetchstr();
	    return '.';
	}
	stmp[i] = '\0';
	ungetchstr();
	sscanf(stmp, "%lf", &d);
	yylval.val = d;
	return NUMBER;
    }
/* graphs, sets, regions resp. */
    if (c == 'G' || c == 'S' || c == 'R') {
	char stmp[80];
	int i = 0, ctmp = c, gn, sn, rn;
	c = getcharstr();
	while (isdigit(c)) {
	    stmp[i++] = c;
	    c = getcharstr();
	}
	if (i == 0) {
	    c = ctmp;
	    ungetchstr();
	} else {
	    ungetchstr();
	    if (ctmp == 'G') {
	        stmp[i] = '\0';
		gn = atoi(stmp);
		if (gn >= 0 && gn < maxgraph) {
		    yylval.ival = gn;
		    whichgraph = gn;
		    return GRAPHNO;
		}
	    } else if (ctmp == 'S') {
	        stmp[i] = '\0';
		sn = atoi(stmp);
		if (sn >= 0 && sn < g[cg].maxplot) {
		    lxy = getsetlength(cg, sn);
		    yylval.ival = sn;
		    whichset = sn;
		    return SETNUM;
		}
	    } else if (ctmp == 'R') {
	        stmp[i] = '\0';
		rn = atoi(stmp);
		if (rn >= 0 && rn < MAXREGION) {
		    yylval.ival = rn;
		    return REGNUM;
		}
	    }
	}
    }
    if (isalpha(c)) {
	char *p = sbuf;
	do {
	    *p++ = c;
	} while ((c = getcharstr()) != EOF && isalnum(c));
	ungetchstr();
	*p = '\0';
        if (debuglevel == 2) {
	    printf("->%s<-\n", sbuf);
	}
	if ((found = findf(key, sbuf, MAXFUN)) >= 0) {
	    if (key[found].type == VAR) {
		switch (sbuf[0]) {
		case 'A':
		    yylval.ptr = aa;
		    return VAR;
		case 'B':
		    yylval.ptr = bb;
		    return VAR;
		case 'C':
		    yylval.ptr = cc;
		    return VAR;
		case 'D':
		    yylval.ptr = dd;
		    return VAR;
		}
	    }
	    else if (key[found].type == FITPARM) {
		int index = sbuf[1] - '0';
		yylval.val = nonl_parms[index];
		return FITPARM;
	    }
	    else { /* set up special cases */
		switch (key[found].type) {
		case XAXIS:
		    naxis = 0;
		    break;
		case YAXIS:
		    naxis = 1;
		    break;
		case ZEROXAXIS:
		    naxis = 2;
		    break;
		case ZEROYAXIS:
		    naxis = 3;
		    break;
		case ALTXAXIS:
		    naxis = 4;
		    break;
		case ALTYAXIS:
		    naxis = 5;
		    break;
		case AXES:
		    naxis = 6;
		    break;
		case XAXES:
		    naxis = 7;
		    break;
		case YAXES:
		    naxis = 8;
		    break;
		case GRAPHS:
		    yylval.ival = -1;
		    whichgraph = -1;
		    return GRAPHS;
		    break;
		case SETS:
		    yylval.ival = -1;
		    whichset = -1;
		    return SETS;
		    break;
		default:
		    break;
		}
	    }
	    yylval.func = key[found].type;
	    return key[found].type;
	} else {
	    strcat(sbuf, ": No such function or variable");
	    yyerror(sbuf);
	    return 0;
	}
    }
    switch (c) {
    case '>':
	return follow('=', GE, GT);
    case '<':
	return follow('=', LE, LT);
    case '=':
	return follow('=', EQ, '=');
    case '!':
	return follow('=', NE, NOT);
    case '|':
	return follow('|', OR, '|');
    case '&':
	return follow('&', AND, '&');
    case '\n':
	return '\n';
    default:
	return c;
    }
}

int follow(int expect, int ifyes, int ifno)
{
    int c = getcharstr();

    if (c == expect) {
	return ifyes;
    }
    ungetchstr();
    return ifno;
}

void yyerror(char *s)
{
 yyerror((const char *)s);
}
void yyerror(const char *s)
{
    int i;
    char buf[256];
    sprintf(buf, "%s: %s", s, f_string);
    i = strlen(buf);
    buf[i - 1] = 0;
    errwin(buf);
    interr = 1;
}

#define C1 0.1978977093962766
#define C2 0.1352915131768107

double rnorm(double mean, double sdev)
{
    double u = drand48();

    return mean + sdev * (pow(u, C2) - pow(1.0 - u, C2)) / C1;
}

double fx(double x)
{
    return 1.0 / sqrt(2.0 * M_PI) * exp(-x * x * 0.5);
}

double normp(double b, double *s)
{
    double sum, dx, a = -8.0, fx(double x);
    int i, n = 48;

    sum = fx(a) + fx(b);
    dx = (b - a) / n;
    for (i = 1; i <= ((n - 1) / 2); i++)
	sum = sum + 4.0 * fx(a + (2.0 * i - 1.0) * dx) + 2.0 * fx(a + 2.0 * i * dx);
    sum = sum + 4.0 * fx(b - dx);
    *s = fx(b);
    return sum * dx / 3.0;
}

double invnorm(double p)
{
    double s, x, z, normp(double b, double *s);

    if (p > 0.5)
	x = 1.0 - p;
    else
	x = p;
    s = sqrt(-2.0 * log(x));
    x = ((-7.49101 * s - 448.047) * s - 1266.846);
    x = x / (((s + 109.8371) * s + 748.189) * s + 498.003) + s;
    if (p < 0.5)
	x = -x;
    z = p - normp(x, &s);
    z = z / s;
    s = x * x;
    return (((((((((720.0 * s + 2556.0) * s + 1740.0) * s + 127.0) * z / 7.0 +
		 ((120.0 * s + 326.0) * s + 127.0) * x) * z / 6.0 + (24 * s + 46.0) * s + 7.0) * z / 40.0 +
	       (0.75 * s + 0.875) * x) * z + s + 0.5) * z / 3.0 + x * 0.5) * z + 1.0) * z + x + 0.832e-24 * x;
}

double invt(double p, int n)
{
    double sign, temp, a, b, c, d, x, y;

    sign = 1.0;
    if (p < 0.5) {
	p = 1.0 - p;
	sign = -1.0;
    }
    p = (1 - p) * 2;
    if (n == 2) {
	temp = sqrt(2.0 / (p * (2.0 - p)) - 2.0);
	temp = sign * temp;
	return temp;
    } else if (n == 1) {
	p = p * M_PI / 2.0;
	return sign * cos(p) / sin(p);
    } else {
	a = 1.0 / (n - 0.5);
	b = 48.0 / (a * a);
	c = ((20700 * a / b - 98.0) * a - 16.0) * a + 96.36;
	d = ((94.5 / (b + c) - 3.0) / b + 1.0) * sqrt(a * M_PI / 2.0) * n;
	x = d * p;
	y = exp((2.0 / n) * log(x));
	if (y > (0.05 + a)) {
	    x = invnorm(p * 0.5);
	    y = x * x;
	    if (n < 5)
		c = c + 0.3 * (n - 4.5) * (x + 0.6);
	    c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
	    y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
	    y = a * y * y;
	    if (y > 0.002)
		y = exp(y) - 1.0;
	    else
		y = 0.5 * y * y + y;
	} else
	    y = ((1.0 / (((n + 0.6) / (n * y) - 0.089 * d - 0.822) * (n + 2.0) * 3.0) + 0.5 / (n + 4.0)) * y - 1.0) * (n + 1.0) / (n + 2.0) + 1.0 / y;
	return sign * sqrt(n * y);
    }
}

/*
 * return a pointer to the array given by v
 */
double *getvptr(int gno, int setno, int v)
{
    switch (v) {
    case PLOT_X:
    case X0:
	return g[gno].p[setno].ex[0];
	break;
    case PLOT_Y:
    case Y0:
	return g[gno].p[setno].ex[1];
	break;
    case Y1:
	return g[gno].p[setno].ex[2];
	break;
    case Y2:
	return g[gno].p[setno].ex[3];
	break;
    case Y3:
	return g[gno].p[setno].ex[4];
	break;
    case Y4:
	return g[gno].p[setno].ex[5];
	break;
    case Y5:
	return g[gno].p[setno].ex[6];
	break;
    }
    return NULL;
}
