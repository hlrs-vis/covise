      PARAMETER NPANZ=17, NPROF=150, NPMAX=200, NECKM=10, NKANT=20,
     *          NMEKO=100, NB = 10, Np2 = 50, anznk=30
      PARAMETER      (npla = 11)
      PARAMETER (IMAX=19 ,NMAX=24)
      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
      CHARACTER*10 TYPN(NECKM), TYPG(NECKM), NAME
      CHARACTER*2  AUSWAHL(NB)
      CHARACTER*1  PAPIER, ANTWORT
      CHARACTER*200 datei_prototyp, datei_zentral_file, datei_daten,
     .              datei_dicke, datei_profil, datei_steuerfile,
     .              datei_profil_neu,datei_output,
     .              datei_zwischen_file,
     .              datei_meko,datei_spezifisch,datei_La_Winkel,
     .              datei_Stroe_Winkel,datei_Ge_Dreiecke,
     .              datei_schaufel_xyz,datei_zwischen_prof,
     .              datei_Batrekhy_prof,datei_ax_stf,
     .              datei_randbedingung,
     .               datei_leitprofil
      LOGICAL  BISPLINE, GUTE_VERTEILUNG, ZENTRALFILE, NEUES_FORMAT,
     .         ZUSPITZEN, EX 
      INTEGER lesenok   
      REAL      theta_sch(npla), theta_p(IMAX), theta_s
      DIMENSION NLAUF(NPANZ), NUN(NECKM), NUG(NECKM), NPARA(NPANZ)
      REAL XEIN(NPROF,NPANZ), YEIN(NPROF,NPANZ), ZEIN(NPROF,NPANZ),
     *     MUE(NPROF,NPANZ), PHI(NPROF,NPANZ),
     *     XABW(NPROF,NPANZ), YABW(NPROF,NPANZ), PHIABW(NPROF,NPANZ),
     *     IVKANTE(NPANZ),
     *     XABWP(NPROF,NPANZ),YABWP(NPROF,NPANZ),PHIABWP(NPROF,NPANZ),
     *     IVKANTEP(NPANZ),
     *     XABWB(NPROF,NPANZ),YABWB(NPROF,NPANZ),PHIABWB(NPROF,NPANZ),
     *     IVKANTEB(NPANZ),
     *     XABWS(NPROF,NPANZ),YABWS(NPROF,NPANZ),PHIABWS(NPROF,NPANZ),
     *     XPROF(NPROF,NPANZ), YPROF(NPROF,NPANZ), ZPROF(NPROF,NPANZ),
     *     SEHNE(NPROF),
     *     U(NPROF), UDS(NPROF),
     *     XPROFP(NPMAX,NPANZ),YPROFP(NPMAX,NPANZ),ZPROFP(NPMAX,NPANZ),
     *     XPROFB(NPMAX,NPANZ),YPROFB(NPMAX,NPANZ),ZPROFB(NPMAX,NPANZ),
     *     X_SKELET(NPMAX/2,NPANZ), Y_SKELET(NPMAX/2,NPANZ),
     *     Z_SKELET(NPMAX/2,NPANZ),
     *     S_SKELET(NPMAX/2,NPANZ), DB_SKELET(NPMAX/2,NPANZ),
     *     DB(NPMAX,NPANZ),
     *     XGIT(NPMAX), YGIT(NPMAX),
     *     ZNABE(NMEKO), RNABE(NMEKO), ZGEH(NMEKO), RGEH(NMEKO),
     *     XECKN(NECKM), YECKN(NECKM), XECKG(NECKM), YECKG(NECKM),
     *     XMITN(NECKM), YMITN(NECKM), XMITG(NECKM), YMITG(NECKM),
     *     RMITN(NECKM), RMITG(NECKM),
     *     XVK(NKANT), YVK(NKANT), XHK(NKANT), YHK(NKANT),
     *     ZVK(NPANZ), RVK(NPANZ), 
     *     ZHKDS(NPANZ), RHKDS(NPANZ), ZHKSS(NPANZ), RHKSS(NPANZ)
      REAL  X_ek(Np2), Y_ek(Np2), Z_ek(Np2), R_ek(Np2),
     .      X_akd(Np2), Y_akd(Np2), Z_akd(Np2), R_akd(Np2),
     .      X_aks(Np2), Y_aks(Np2), Z_aks(Np2), R_aks(Np2)
      REAL  KCM2, KU2, N, NM, N1str, Nq
      REAL  x_ds(IMAX,npla), y_ds(IMAX,npla), z_ds(IMAX,npla),
     .      x_ss(IMAX,npla), y_ss(IMAX,npla), z_ss(IMAX,npla)     
      REAL  beta1_sch(npla), beta2_sch(npla),
     .      delta_beta1(npla), delta_beta2(npla), d_2(npla)
      REAL  DL_basprof(IMAX), XL_basprof(IMAX), profmax
      REAL  DL_prof(npla,IMAX), XL_prof(npla,IMAX)
      REAL  XWRL, BETA1, BETA2, DS, AL, Z, dzul, sdpktzul
      REAL  dzul_r(imax), d_theta2(imax), d_theta1(imax)
      REAL  d2, di_da, n_z2, D1, D2Z, z_axe_la
      REAL  r_nabe(anznk), z_nabe(anznk)
      REAL  r_kranz(anznk), z_kranz(anznk)
      REAL  P_XI(IMAX), P_ETA(IMAX), BGSEHNE(IMAX), BGLAENGE(IMAX)
      REAL  XI_ALT, ETA_ALT, GESLAENGE, UEBER
      REAL  umschlingung, max1, max2, pe1i, pe1a, pe2i, pe2a
      REAL  dicke_i, dicke_a, d_strich_a 
      REAL  r_j
      REAL  b0,d0,l0, leit_dr,D0i,D0a
      REAL  x_leit_prof,y_leit_prof
      REAL  d2_kugel
      REAL  versch_i,versch_a
      INTEGER ZDP
      INTEGER nlschaufel,Nleit,  Nprofile
      INTEGER anznabe,anzkranz,posnabe,poskranz,numnabe
      INTEGER lese, schreibe, fehler, WRSPEZIFISCH, WRBETRIEBSPUNKT,
     .        WRMEKO, WROUTPUT, WRPROFILE_NEU, WRZWISCHEN_FILE,
     .        WRLA_WINKEL, WRSTROE_WINKEL, WRGE_DREIECKE,
     .        WRZWISCHEN_PROF, WRSCHAUFEL_XYZ, WRBATREKHY,
     .        WRPROTOTYP, success
      COMMON /LAUFR/ NLAUF, XEIN, YEIN, ZEIN
      COMMON /KONFO/ PHI, MUE, PHITEMP, ALPHA_AB
      COMMON /ABWIC/ XABW, YABW, PHIABW, IVKANTE
      COMMON /ABWICP/XABWP, YABWP, PHIABWP, IVKANTEP
      COMMON /ABWICB/XABWB, YABWB, PHIABWB, IVKANTEB
      COMMON /ABWICS/XABWS, YABWS, PHIABWS
      COMMON /NABE/  NECKN, XECKN, YECKN, TYPN, NUN, XMITN,YMITN,RMITN
      COMMON /GEHAE/ NECKG, XECKG, YECKG, TYPG, NUG, XMITG,YMITG,RMITG 
      COMMON /MEKO/  NPNABE, ZNABE, RNABE, NPGEH, ZGEH, RGEH
      COMMON /KANTE/ NVK, XVK, YVK, NHK, XHK, YHK
      COMMON /KANBE/ ZVK, RVK, ZHKDS, RHKDS, ZHKSS, RHKSS
      COMMON /EK_ORI/ X_ek, Y_ek, Z_ek, R_ek, N_kante
      COMMON /AK_DS/  X_akd, Y_akd, Z_akd, R_akd
      COMMON /AK_SS/  X_aks, Y_aks, Z_aks, R_aks
      COMMON /TRANS/ XPROF, YPROF, ZPROF, SEHNE, U
      COMMON /NETZA/ NPARA, XPROFP, YPROFP, ZPROFP, UDS
      COMMON /NETZB/ XPROFB, YPROFB, ZPROFB
      COMMON /SKELE/ N_SKELET, X_SKELET, Y_SKELET, Z_SKELET
      COMMON /EINSK/ DB_HK_KRANZ, SE_KRANZ, HE_HK
      COMMON /BIEGE/ S_SKELET, DB_SKELET, DB
      COMMON /GITTR/ XGIT, YGIT
      COMMON /GRAPH/ NSCHAUFEL, ALPHAG, BETAG
      COMMON /HAUPT/ B1,n_z0,  phi0
      COMMON /BILDR/ AUSWAHL, PAPIER, ANTWORT      
      COMMON /GRIFF/ IPEN1, IPEN2, IPEN3, IPEN4, IPEN5, IPEN6, IPEN7
      COMMON /io/    lese, schreibe, fehler, WRSPEZIFISCH, 
     .               WRBETRIEBSPUNKT, WRMEKO, WROUTPUT, WRPROFILE_NEU,
     .               WRZWISCHEN_FILE, WRLA_WINKEL, WRSTROE_WINKEL,
     .               WRGE_DREIECKE, WRZWISCHEN_PROF, WRSCHAUFEL_XYZ,
     .               WRBATREKHY, WRPROTOTYP, success
      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT
      COMMON /daten/ XWRL, BETA1, BETA2, DS, AL, Z, dzul, sdpktzul,
     .               NAME
      COMMON /BASIS/ DL_basprof(IMAX), XL_basprof(IMAX), profmax
      COMMON /PROFI/ DL_prof(npla,IMAX), XL_prof(npla,IMAX)
      COMMON /DICKE/ dzul_r(imax), d_theta2(imax), d_theta1(imax),
     .               theta_sch(npla), theta_s
      COMMON /abwik/ x_sl_ab(IMAX, npla), y_sl_ab(IMAX, npla),
     .               x_ds_ab(IMAX, npla), y_ds_ab(IMAX, npla),
     .               x_ss_ab(IMAX, npla), y_ss_ab(IMAX, npla)
      COMMON /LAUF/  d_1(npla), d_2(npla), beta_1(npla), beta_2(npla)
      COMMON /WINK/  beta1_sch(npla), beta2_sch(npla),
     .               delta_beta1(npla), delta_beta2(npla)
      COMMON /xyz/   x_ds(IMAX,npla), y_ds(IMAX,npla), z_ds(IMAX,npla),
     .               x_ss(IMAX,npla), y_ss(IMAX,npla), z_ss(IMAX,npla),
     .               x_sl(IMAX,npla), y_sl(IMAX,npla), z_sl(IMAX,npla)
      COMMON /PROFI/ ZDP, DL(IMAX), XL(IMAX)
      COMMON /DREHP/ x_sl_dpkt(npla), y_sl_dpkt(npla)
      COMMON /DSSSL/ SLX(IMAX),SLY(IMAX),DSX(IMAX),DSY(IMAX),
     .               SSX(IMAX),SSY(IMAX), SLX_dpkt, SLY_dpkt
      COMMON /absol/ Q, H, N, ETA, VAUBEZ
      COMMON /leitg/ Nleit,b0,d0,l0,leit_dr,D0i,D0a
      COMMON /MODELL/NM, HM
      COMMON /zwis/  d2, di_da, nlschaufel, n_z2, D1, D2Z,
     .               z_axe_la
      COMMON /RELAT/ KCM2, KU2, Nq, N1str, Q1str
      COMMON /GEOME/ D2M,  D2i, D2a
      COMMON /anzah/ Nprofile
      COMMON /fona/  anznabe,r_nabe(anznk), z_nabe(anznk),d2_kugel
      COMMON /fokr/  anzkranz, r_kranz(anznk), z_kranz(anznk),
     #               posnabe,poskranz,numnabe
      COMMON /kont/  umschlingung, max1, max2, pe1i, pe1a, pe2i, pe2a
      COMMON /verz/  datei_prototyp, datei_zentral_file, datei_daten,
     .               datei_dicke, datei_profil, datei_steuerfile,
     .               datei_profil_neu,datei_output,
     .               datei_zwischen_file,
     .               datei_meko,datei_spezifisch,datei_La_Winkel,
     .               datei_Stroe_Winkel,datei_Ge_Dreiecke,
     .               datei_schaufel_xyz,datei_zwischen_prof,
     .               datei_Batrekhy_prof,datei_randbedingung,
     .               datei_leitprofil
      COMMON /ver2/  datei_ax_stf
      COMMON /lesn/  lesenok
      COMMON /dick/  dicke_i ,dicke_a, d_strich_a
      COMMON /radi/  r_j 
      COMMON /wust/  db2i, db2a, db1i, db1a
      COMMON /leits/ x_leit_prof(38),y_leit_prof(38)
      COMMON /STROE/ cm1, cm2, cm0i, cu0i, c0i, u0i, alpha_0i,
     .               wu0i, w0i, beta_0i
      COMMON /prof/  versch_i,versch_a   
