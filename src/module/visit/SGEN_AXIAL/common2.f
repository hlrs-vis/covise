      PARAMETER      npla = 11
      REAL           KCM2, KU2, N, NM, N1str, Nq, PI
      REAL           d2, di_da, n_z2, D1, D2Z, z_axe_la
      REAL           db2i, db2a, db1i, db1a
      REAL           mwi,mwa
      REAL           umschlingung, max1, max2, pe1i, pe1a, pe2i, pe2a
      REAL           dicke_i, dicke_a, d_strich_a 
      REAL           ETA_H, ETA_EL, ETA_M
      REAL           PHI, ALPHA_AB
      REAL           b0,d0,l0, leit_dr,D0i,D0a
      INTEGER        nlschaufel, Nleit,Nprofile
      CHARACTER*20   HYDRAULIK      
      CHARACTER*10   DATUM
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
      LOGICAL        EX
      INTEGER lesenok    
      COMMON /io/    lese, schreibe, fehler, WRSPEZIFISCH, 
     .               WRBETRIEBSPUNKT, WRMEKO, WROUTPUT, WRPROFILE_NEU,
     .               WRZWISCHEN_FILE, WRLA_WINKEL, WRSTROE_WINKEL,
     .               WRGE_DREIECKE, WRZWISCHEN_PROF, WRSCHAUFEL_XYZ,
     .               WRBATREKHY, WRPROTOTYP,success
      COMMON /absol/ Q, H, N, ETA, VAUBEZ
      COMMON /WIRKU/ ETA_H, ETA_EL, ETA_M
      COMMON /RELAT/ KCM2, KU2, Nq, N1str, Q1str
      COMMON /GEOME/ D2M, D2i, D2a
      COMMON /LAUF/  d_1(npla), d_2(npla) , beta_1(npla), beta_2(npla)    
      COMMON /leitg/ Nleit,b0,d0,l0, leit_dr,D0i,D0a
      COMMON /MODELL/NM, HM
      COMMON /WINKL/ PHI, ALPHA_AB
      COMMON /anzah/ Nprofile
      COMMON /STROE/ cm1, cm2, cm0i, cu0i, c0i, u0i, alpha_0i,
     .               wu0i, w0i, beta_0i
      COMMON /DREIE/ u1(npla), cu1(npla), wu1(npla), c1(npla), w1(npla),
     .               u2(npla), cu2(npla), wu2(npla), c2(npla), w2(npla),
     .               alpha1(npla), alpha2(npla)
      COMMON /zwis/  d2, di_da, nlschaufel, n_z2, D1, D2Z,
     .               z_axe_la
      COMMON /wust/  db2i, db2a, db1i, db1a
      COMMON /maxw/  mwi,mwa
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
      COMMON /lesn/  lesenok
      COMMON /dick/  dicke_i ,dicke_a, d_strich_a
 

