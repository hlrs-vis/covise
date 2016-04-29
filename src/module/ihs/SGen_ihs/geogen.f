C  Geogen                  **************************
C                          *                        *
C                          *    Programm NETZGEN    *
C                          *                        *
C                          **************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Dieses Programm bereitet die Rohdaten des Schaufelprofils, wie sie  I
C    I  in der Profilzeichnung stehen, auf fuer das parametrisierte Rechen- I
C    I  netz, wie es das EULER-Programm benoetigt.                          I
C    I                                                                      I
C    I          - ZENTRAL_FILE.DAT    (IN)                                  I
C    I          - PROJEKT.DAT         (IN)                                  I
C    I          - BILDER.DAT          (IN)                                  I
C    I                                                                      I
C    I          - PROTOTYP.DAT        (IN)                                  I
C    I                                                                      I
C    I          - ZWISCHEN_FILE.DAT   (OUT)                                 I
C    I          - MEKO.DAT            (OUT)                                 I
C    I          - OUTPUT.DAT          (OUT)                                 I
C    I          - PROFILE_NEU.DAT     (OUT)                                 I
C    I                                                                      I
C    I          - LA-WINKEL.DAT       (OUT)                                 I
C    I          - STROE-WINKEL.DAT    (OUT)                                 I
C    I          - GE-DREIECKE.DAT     (OUT)                                 I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  17. 8. 98

      

      SUBROUTINE geogen()

      INCLUDE 'common.f'
      CHARACTER*8  AXEN1, AXEN2, AXEN3, MARK1, MARK2, SCHAUF1, SCHAUF2,
     .             SCHAUF3
      CHARACTER*8  SKEL1, SKEL2, BIEG1, BIEG2
      CHARACTER*10 TEXT1, TEXT3, TEXT4, TEXT5, TEXT
      CHARACTER*15 TEXT2, TEXT6
      LOGICAL ORIGINAL_PROFILE, INTERPOLIERTE_PROFILE, GEBOGENE_PROFILE,
     .        MERIDIAN_SCHNITT, DREI_D_ANSICHT, PROFIL_VERGLEICH,
     .        VORDER_HINTERKANTE, PARAMETRISIERUNG, KONFORME_ABBILDUNG,
     .        ABWICKLUNG_DER_PROFILE, MIT_PUNKTEN, TAPETE, ROLLE
      INTEGER NPARAM
      TEXT1 = 'ORIGINAL'
      TEXT6 = 'GUT_VERTEILT'
      MARK1 = 'MIT'
      MARK2 = 'OHNE'
      SCHAUF1 = 'MIT'
      SCHAUF2 = 'OHNE'
      SCHAUF3 = 'NEU'
      AXEN1 = 'YMINUSX'
      AXEN2 = 'YZ'
      AXEN3 = 'XMINUSZ'
      SKEL1 = 'MIT'
      SKEL2 = 'OHNE'
      BIEG1 = 'MIT'
      BIEG2 = 'OHNE'
      PI = 3.14159
      LESEN=lese
      lesenok=1
C      IF (LESEN .EQ. 1) PRINT*,'lese=1'
    5 FORMAT('')
  
C     ------ Definitionen aus dialog.f -----
      BISPLINE = .false.    
      GUTE_VERTEILUNG = .true.
      Nprofile = 11
      NPARAM = 21
      ALPHAG = 5.
      BETAG = 4.
      NSCHAUFEL = 4
C     MBECKER DEBUG ITEXT = 'N'     
      TAPETE = .true.
C     ------ Ende Definitionen ------



C      WRITE(6,5)


      
C ------ Folgende Prozeduren stammen aus netzgen.f ----------
      IF (LESEN .EQ. 1)            CALL EINGABE_STEUERFILE
      IF (LESEN .EQ. 1)            CALL EINLESEN_ZENTRALFILE
      
C      IF (lesenok.EQ.1) then
C  	  CALL TRANSFOR
C  	  CALL KONFORM
C  	  CALL ABWICKLUNG ('ORIGINAL')
C  	  CALL PARAMET (TEXT1)
C         CALL PARAMET (TEXT6)
C 	  CALL SCALA
C      END IF      

C      IF (schreibe .EQ. 1) then
C         IF (WRPROFILE_NEU .EQ. 1)    CALL AUSGABE_PROFILE_NEU        
C         IF (WROUTPUT .EQ. 1)         CALL AUSGABE_OUTPUT
C         IF (WRZWISCHEN_FILE .EQ. 1)  CALL AUSGABE_ZWISCHEN_FILE
C         IF (WRMEKO .EQ. 1)           CALL AUSGABE_MEKO_FILE
C      END IF     
C ------ Ende des ersten Pfades ------------------------------



C ------ Folgende Prozeduren stammen aus Eulerglg.f ----------

      IF (LESEN .EQ. 1)            CALL EINGABE_Q_H_N
   
   
      IF (lesenok.EQ.1) then 
C      	 CALL RECHNE_KCM_KU
C      	 CALL UMRECHNEN_AUF_MODELL
         CALL RECHNE_UMLENKUNG_NABE_KRANZ
         CALL rechne_kugel
      	 CALL RECHNE_WINKEL_STROE
      	 CALL LAUF_WINKEL_UEBERTREIBUNG
      	 CALL RECHNE_LEITRAD_EK_AK
      	 CALL RECHNE_WINKEL_LEIT
      	 CALL RECHNE_GE_DREIECKE
      END IF  
      
      IF (schreibe .EQ. 1) then
         call schreibe_randbedingung
         IF (WRPROTOTYP .EQ. 1)       CALL schreibe_prototyp
         IF (WRLA_WINKEL .EQ. 1)      CALL AUSGABE_LAUFRADWINKEL
         IF (WRSTROE_WINKEL .EQ. 1)   CALL AUSGABE_EULERWINKEL
         IF (WRGE_DREIECKE .EQ. 1)    CALL AUSGABE_DREIECKE_LAUF
      END IF

     
C ------ Ende des zweiten Pfades -----------------------------
C      WRITE(6,5)
C ------ Folgende Prozeduren stammen aus schaufel.f ----------     
C      nlauf = 3  
    
    
      IF (LESEN .EQ. 1)         call EINGABE_DATEN
      IF (LESEN .EQ. 1)         call EINGABE_BASISPROFIL
      IF (LESEN .EQ. 1)         call EINGABE_LEITPROFIL
C      IF (LESEN .EQ. 1)         call EINGABE_PROFILDICKE

      IF (lesenok.EQ.1) then
 	 call RECHNE_SCHAUFELKONTUR
 	 call RECHNE_SCHAUFELDICKEN
 	 call RECHNE_SCHAUFELPROFILVERSCHIEBUNG
 	 call RECHNE_HINTERKANTE
 	 call RECHNE_ABWICKLUNG
 	 call RECHNE_XYZ_KOORDINATEN
      END IF

      IF (schreibe .EQ. 1) then
         IF (WRZWISCHEN_PROF .EQ. 1)    call AUSGABE_ABWICKLUNG
         IF (WRSCHAUFEL_XYZ .EQ. 1)     call AUSGABE_XYZ_KOORDINATEN
         IF (WRBATREKHY .EQ. 1)         call AUSGABE_BATREKHY
      END IF

C ------ Ende des dritten Pfades -----------------------------

C      WRITE(6,5)


      RETURN
   
      END

