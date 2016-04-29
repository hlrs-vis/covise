double abstand (double, double, double, double);

void GERADE(double *x1, double *y1,
double *x2, double *y2,
int seed,
int m,
double L12,
double *x, double *y);

void MESHSEED(double *XK, double *YK,
int MO,
double AA, double AE,
int PA,  int PE,
double VX, double VY, double VT,
int M,
double L12,
int anz_naca,
int NL,
double *XS, double *YS);

void RECHNE_EINTEILUNG_L12(int *NL,
double *TBEG, double *TEND,
int   mo,
double L12,
double *TAB);

void DREIECK(double *fixbord1, double *fiybord1,
double *fixbord2, double *fiybord2,
double *fixbord3, double *fiybord3,
int ase,
double *fixr12,
double *fiyr12);

void BEZIER(double *x1,
double *y1,
double phi_start,
double *x2,
double *y2,
double phi_ende,
int modus,
int *anz_pkt,
int m,
double *L12,
double *x, double *y);

void RECHNE_EINLAUF(int schnitt,
double *fixb32,
double *yo, double *delta,
double *fixplo,   double *fiyplo,
double *fixpro,   double *fiypro);

void RECHNE_MITTELPUNKTE(double *fixplo, double *fixpro,
double *yo, int *dat1,  int *dat2, int *dat3,
double *fixpmlo, double *fiypmlo,
double *fixpmro, double *fiypmro);

void DOPPELTE_WEG (double *fixr, double *fiyr,
int seed, double TOL, int *knot_nr, int *randpunkt,
double *neux, double *neuy,
int *anz_doppelt, int *ersatz_nr);

void GERADE3D(double *x1,  double *y1, double *z1,
double *x2, double *y2, double *z2,
int anz,
int seed,
int m,
double L12,
double *x,  double *y, double*z);

void AUSGABE_3D_GEO(char *geo_pfad, double *NETZX, double *NETZY , double *NETZZ,
int anz_schnitte,
int seed,
int *el_liste,
int anz_elemente);

void RECHNE_RB(int anz_schnitte, int anz_elemente, int *ersatz_nr,
int seed, int ase[16][2], int anz_kmark, int *kmark, int anz_wrb_ls,
int *wrb_ls, int anz_elmark, int anz_elmark_einlauf,
int anz_elmark_eli, int anz_elmark_ere, int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re, int *elmark_15li,
int *elmark_15re, int *elmark_auslauf);

void AUSGABE_3D_RB(char *rb_pfad, int anz_schnitte, int seed, int anz_wrb_ls, int *wrb_ls,
int anz_elemente, int *el_liste, int anz_grenz, int anz_elmark_einlauf,
int anz_elmark_eli,  int anz_elmark_ere,  int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re,
int *elmark_15li, int *elmark_15re, int *elmark_auslauf,
int anz_kmark, int *kmark, int ase[16][2], int start3,
int start5, int end6, int end7, int start10, int start11, int start15, double *p2);

void RECHNE_3D_RB(struct ggrid *gg, int anz_schnitte, int seed, int anz_wrb_ls, int *wrb_ls,
int anz_elemente, int *el_liste, int anz_grenz, int anz_elmark_einlauf,
int anz_elmark_eli,  int anz_elmark_ere,  int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re,
int *elmark_15li, int *elmark_15re, int *elmark_auslauf,
int anz_kmark, int *kmark, int ase[16][2], int start3,
int start5, int end6, int end7, int start10, int start11, int
start15, double *p2);
