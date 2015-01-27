C                      **************************************
C                      *                                    *
C                      *        AUSGABE_PROFIL_NEU          *
C                      *                                    *
C                      **************************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Hier wird die Profilkoordinaten ausgegeben, die sich nach der Para- I
C    I  metrisierung ergeben haben. Die Verteilung der Punkte ist:          I
C    I                                                                      I
C    I    - gleiche Anzahl Punkte fuer jedes Profil,                        I
C    I    - gleich viel Punkte auf Druck- und Saugseite des Profils.        I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  H824,  24. 2. 94

      SUBROUTINE AUSGABE_PROFILE_NEU
  
      INCLUDE  'common.f'
   
    5 FORMAT ('   profil_neu.dat ausgegeben')
   10 FORMAT (X, 'Dieses File enthaelt die Profilkoordinaten der ',
     .           'Laufschaufel gut verteilt')
   20 FORMAT (X, '-----------------------------------------------',
     .           '------', /)
   30 FORMAT (X, 'Anzahl Profile: ', /, I5)
   40 FORMAT (X, 'Anzahl Punkte pro Profil: ', /, I5, /)
   50 FORMAT (X, '***** Die Profilkoordinaten laufen von HK DS --> HK ',
     .           'SS *****' )
   60 FORMAT (/, X, 'Profil ', I2, /)
   70 FORMAT (X, '  X [mm]     Y [mm]     Z [mm] ', /)
   80 FORMAT (3(3X, F10.3))


      INQUIRE(FILE=datei_profil_neu,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_profil_neu, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF

 
      OPEN (3, FILE=datei_profil_neu, STATUS='OLD')

C      NPUNKTE = NPARA(1)
C      WRITE (3, 10)
C      WRITE (3, 20)
C      WRITE (3, 30) Nprofile
C      WRITE (3, 40) NPUNKTE
C      WRITE (3, 50)
C      DO J=Nprofile,1,-1
C         WRITE (3, 60) J
C         WRITE (3, 70)
C         DO I=1,NPUNKTE
C            WRITE (3, 80) XPROFP(I,J), YPROFP(I,J), ZPROFP(I,J)
C         END DO
C      END DO         
  
      CLOSE (3)
  
C      WRITE(6,5)

      RETURN
      END





C  Ausgabe                *********************************
C                         *                               *
C                         *        AUSGABE_OUTPUT         *
C                         *                               *
C                         *********************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Hier wird das File fuer den Netzgenerator erstellt.                 I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  EPFL-SEWZ,  7. 1. 87
      SUBROUTINE AUSGABE_OUTPUT
  
      INCLUDE  'common.f'
      DIMENSION VDS(NPANZ)
      LOGICAL   GEBOGEN
C      CHARACTER*(*) TEXT

    5 FORMAT ('   output.dat ausgegeben')
   10 FORMAT (I5, 10X, 'Schaufel-Vorderkante  11 Pkte (-Z; R) ') 
   20 FORMAT (I5, 10X, 'Schaufel-HK  SaugS u. DruckS  (-Z; R) ') 
   30 FORMAT (I5, 10X, 'Meridiankontur, Nabe u. Kranz (-z; R) ')
   40 FORMAT (5X, 2F10.3)
   50 FORMAT (5X, 4F10.3)
   60 FORMAT (2I5)
   70 FORMAT (5F15.8)
      
 

      INQUIRE(FILE=datei_output,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_output, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF

      OPEN (3, FILE=datei_output, STATUS='NEW')

C      GEBOGEN = TEXT .EQ. 'BIEGEN'

      Z0 = 0.
   
      WRITE (3, 10) Nprofile
      DO J=Nprofile,1,-1
         ZVK_T = -(Z0-ZVK(J))
         WRITE (3,40) ZVK_T, RVK(J)
      END DO
  
      WRITE (3, 30) NPNABE
      DO I=1,NPNABE
         ZNABE_T = -(Z0-ZNABE(I))
         ZGEH_T  = -(Z0-ZGEH(I))
         WRITE (3,50) ZNABE_T, RNABE(I), ZGEH_T, RGEH(I)
      END DO
  
      WRITE (3, 20) Nprofile
      DO J=Nprofile,1,-1
         ZHKDS_T = -(Z0-ZHKDS(J))
         ZHKSS_T = -(Z0-ZHKSS(J))
         WRITE (3,50) ZHKDS_T, RHKDS(J), ZHKSS_T, RHKSS(J)
      END DO

      UDS(1)        = -1.0
      UDS(NPARA(1)) =  1.0
      DO I=1,Nprofile
         NSCH = Nprofile
         IF (Nprofile .EQ. 1) NSCH = 2
         VDS(I) = REAL(I-1)/REAL(NSCH-1)
      END DO
      WRITE (3,60) NPARA(1), Nprofile
      WRITE (3,70) (UDS(I), I=1,NPARA(1))
      WRITE (3,70) (VDS(I), I=1,Nprofile)
  
      IF (GEBOGEN) THEN
         DO J=Nprofile,1,-1
            DO I=1,NPARA(J)
               WRITE (3,70) -XPROFB(I,J), YPROFB(I,J), ZPROFB(I,J)
            END DO
         END DO
      ELSE
         DO J=Nprofile,1,-1
            DO I=1,NPARA(J)
               WRITE (3,70) -XPROFP(I,J), YPROFP(I,J), ZPROFP(I,J)
            END DO
         END DO
      END IF
  
      CLOSE (3)

C      WRITE(6,5)

      RETURN
      END



C                       *******************************
C                       *                             *
C                       *    AUSGABE_ZWISCHEN_FILE    *
C                       *                             *
C                       *******************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Hier wird die Profilkoordinaten ausgegeben, die fuer die graphi-    I
C    I  schen Darstellungen notwendig sind.                                 I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  24. 10. 95

      SUBROUTINE AUSGABE_ZWISCHEN_FILE
  
      INCLUDE  'common.f'
      DIMENSION VDS(NPANZ)
      LOGICAL   GEBOGEN
      REAL PI


    5 FORMAT ('   zwischen_file.dat ausgegeben')
   10 FORMAT (X, 'Dieses File enthaelt die Koordinaten des Kanals ',
     .           ' - zunaechst - ')
   20 FORMAT (X, '------------------------------------------------',
     .           '---------------', /)
   30 FORMAT (X, 'Anzahl Profile: ', /, I5, /)
   40 FORMAT (/, X, 'Anzahl Punkte pro Kanalkontur: ', /, I5, /)
   50 FORMAT (X, '***** Die Profilkoordinaten laufen von HK DS --> HK ',
     .           'SS *****' )
   60 FORMAT (/, X, 'Profil ', I2, /)
   70 FORMAT (X, '  X [mm]     Y [mm]     Z [mm] ', /)
   80 FORMAT (3(3X, F10.3))
  110 FORMAT (X, 'Schaufel-Vorderkante  Z [mm]   R [mm] ') 
  120 FORMAT (/, X, 'Schaufel-Hinterkante ZDS [mm]  RDS [mm] ',
     .           ' ZSS [mm]  RSS [mm] ') 
  130 FORMAT (X, 'Meridiankontur, Nabe u. Kranz  ')
  135 FORMAT (20X, '  ZN [mm]  RN [mm]   ZK [mm]   RK [mm] ')
  140 FORMAT (2X, 2F10.3)
  145 FORMAT (2X, 4F10.3)
  150 FORMAT (2X, 3F10.3)
  160 FORMAT (2I5)
  170 FORMAT (5F15.8)
  180 FORMAT (2X, I5, 5F10.3)
   

      INQUIRE(FILE=datei_zwischen_file,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_zwischen_file, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN (3, FILE=datei_zwischen_file, STATUS='NEW')



      PI = 3.14159
      Z0 = 0.
    
      WRITE (3, 180) n_z2, B0, D1, d2, D2Z, z_axe_la
      WRITE (3, 160) NPNABE
      DO I=1,NPNABE
         ZNABE_T = -(Z0-ZNABE(I))
         ZGEH_T  = -(Z0-ZGEH(I))
         WRITE (3,145) RNABE(I), ZNABE_T, RGEH(I), ZGEH_T
      END DO


      WRITE (3, 160) N_kante
      DO i=1,N_kante
         ZVK_T = -(Z0-Z_ek(i))
         WRITE (3,140) R_ek(i), ZVK_T
      END DO
      DO i=1,N_kante
         ZHKDS_T = -(Z0-Z_akd(i))
         ZHKSS_T = -(Z0-Z_aks(i))
         WRITE (3,145) R_akd(i), ZHKDS_T, R_aks(i), ZHKSS_T
      END DO
  
      NPUNKTE = NPARA(1)
      WRITE (3, 160) Nprofile
      WRITE (3, 160) NPUNKTE
      DO J=Nprofile,1,-1
         DO I=1,NPARA(J)
            WRITE (3,150) XPROFP(I,J), YPROFP(I,J), ZPROFP(I,J)
         END DO
      END DO
      WRITE (3, 180) n_z0, B0, D0, phi0

       
     
      CLOSE (3)

C      WRITE(6,5)

      RETURN
      END





C                         ***************************
C                         *                         *
C                         *       AUSGABE_MEKO      *
C                         *                         *
C                         ***************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I           Ausgabe der Meridiankontur fuer diverse Zwecke.            I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                               E. Goede  IHS,  24. 6. 96

      SUBROUTINE AUSGABE_MEKO_FILE
  
      INCLUDE  'common.f'

    5 FORMAT ('   meko.dat ausgegeben')
   80 FORMAT (3(3X, F10.3))
  140 FORMAT (2X, 2F11.3)
  145 FORMAT (3X, 4F11.3)
  150 FORMAT (2X, 3F10.3)
  160 FORMAT (2I5)
  170 FORMAT (5F15.8)
  180 FORMAT (2X, I5, 4F10.3)
   

      INQUIRE(FILE=datei_meko,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_meko, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN (3, FILE=datei_meko, STATUS='NEW')
    
      faktor = 1.0
      WRITE (3, 160) NPNABE
      DO I=1,NPNABE
         r_aus = RNABE(I)*faktor
         z_aus = ZNABE(I)*faktor
         WRITE (3,140) r_aus, z_aus
      END DO
      DO I=1,NPNABE
         r_aus = RGEH(I)*faktor
         z_aus = ZGEH(I)*faktor
         WRITE (3,140) r_aus, z_aus
      END DO
      WRITE (3, 160) N_kante
      DO i=1,N_kante
         x_aus = X_ek(i)*faktor
         y_aus = Y_ek(i)*faktor
         z_aus = Z_ek(i)*faktor
         r_aus = R_ek(i)*faktor
         WRITE (3,145) x_aus, y_aus, z_aus, r_aus
      END DO
      DO i=1,N_kante
         x_aus = X_akd(i)*faktor
         y_aus = Y_akd(i)*faktor
         z_aus = Z_akd(i)*faktor
         r_aus = R_akd(i)*faktor
         WRITE (3,145) x_aus, y_aus, z_aus, r_aus
      END DO
      DO i=1,N_kante
         x_aus = X_aks(i)*faktor
         y_aus = Y_aks(i)*faktor
         z_aus = Z_aks(i)*faktor
         r_aus = R_aks(i)*faktor
         WRITE (3,145) x_aus, y_aus, z_aus, r_aus
      END DO
     
      CLOSE (3)

C      WRITE(6,5)

      RETURN
      END


C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Prototypdaten in spezifischer Darstellung.         I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  6. 6. 96

      SUBROUTINE AUSGABE_SPEZIFISCH

      INCLUDE 'common2.f'

    5 FORMAT ('   spezifisch.dat ausgegeben')
   10 FORMAT (' -----------------------------------------  ')
   20 FORMAT (' Prototypdaten in spezifischer Darstellung: '  )
   30 FORMAT (' -----------------------------------------  ',/)
   40 FORMAT (' Laufrad-Nr., Projekt .................. : ', A25      )
   50 FORMAT (' Betriebspunkt ......................... : ', A20, /   )
   60 FORMAT (' Spezifischer Durchfluss .......... Kcm2 =', F8.4      )
   70 FORMAT (' Spezifische Umfangsgeschwindigkeit Ku2  =', F8.4      )
   80 FORMAT (' Spezifische Drehzahl ............. nq   =', F7.1, 
     .        ' U/min')
   90 FORMAT (' Einheitsdrehzahl ................. n11  =', F7.1,
     .        ' U/min')
  110 FORMAT (' Einheitsdurchfluss ............... Q11  =',F7.3,' m3/s')


      INQUIRE(FILE=datei_spezifisch,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_spezifisch, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN  (3, FILE=datei_spezifisch, STATUS='NEW')

      WRITE (3, 10)
      WRITE (3, 20)
      WRITE (3, 30)
      WRITE (3, 40) PROJEKT
      WRITE (3, 50) KOMMENTAR
      WRITE (3, 60) KCM2
      WRITE (3, 70) KU2
      WRITE (3, 80) Nq
      WRITE (3, 90) N1str
      WRITE (3,110) Q1str
      CLOSE (3)
  
C      WRITE (6,5)
  
      RETURN
      END





C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Ein- und Austrittswinkel des Laufrades.            I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  7. 6. 96

      SUBROUTINE AUSGABE_LAUFRADWINKEL

      
      INCLUDE 'common2.f'

      

C      REAL  beta1_sch(npla), beta2_sch(npla),
C     .      delta_beta1(npla), delta_beta2(npla)

      common /WINK/  beta1_sch(npla), beta2_sch(npla),
     .               delta_beta1(npla), delta_beta2(npla)


      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT
      

      PI = 3.14159

    5 FORMAT ('   La-Winkel.dat ausgegeben')
   10 FORMAT ('    -----------------------------------------  ')
   20 FORMAT ('    Ein- und Austrittswinkel der Laufschaufel : '  )
   30 FORMAT ('    -----------------------------------------  ',/)
   40 FORMAT ('    Laufrad-Nr., Projekt ....... : ', A25 ,//)
   50 FORMAT ('    Profil  Durchmesser   Beta1       Beta2 ',
     .        '     Delta B1    Delta B2    ')
   55 FORMAT ('                [m]        [o]         [o] ',
     .        '         [o]         [o]     ', /)
   60 FORMAT (I8,1X,F10.4,1X,F10.3,1X,F10.3,1X,F10.3,1X,F10.3,1X,F10.3)
  

      INQUIRE(FILE=datei_La_Winkel,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_La_Winkel, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN  (3, FILE=datei_La_Winkel, STATUS='NEW')

      grad = 180./PI
      WRITE (3, 10)
      WRITE (3, 20)
      WRITE (3, 30)
      WRITE (3, 40) PROJEKT
      WRITE (3, 60) Nprofile 
      WRITE (3, 50) 
      WRITE (3, 55) 
      do i=1,Nprofile
         WRITE (3, 60) i, (d_2(i)/1000), beta1_sch(i), 
     .                            beta2_sch(i),
     .                            delta_beta1(i),
     .                            delta_beta2(i),
     .                 beta1_sch(i)-beta2_sch(i)
      end do
      CLOSE (3)
  
C      WRITE (6,5)
  
      RETURN
      END




C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Stroemungswinkel (Eulerwinkel) an Ein- und Aus-    I
C    I       tritt des Laufrades.                                           I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  7. 6. 96

      SUBROUTINE AUSGABE_EULERWINKEL

      INCLUDE 'common2.f'

      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT

      PI = 3.14159

    5 FORMAT ('   Stroe-Winkel.dat ausgegeben')  
   10 FORMAT ('    ------------------------------------    ')
   20 FORMAT ('    Stroemungswinkel an der Laufschaufel :  ')
   30 FORMAT ('    ------------------------------------  ',/)
   40 FORMAT ('    Laufrad-Nr., Projekt ....... : ', A25 ,//)
   50 FORMAT ('    Profil  Durchmesser   Beta1       Beta2 ')
   55 FORMAT ('                [m]        [o]         [o] ', /)
   60 FORMAT (I8, F12.4, 4F12.3)
 

      INQUIRE(FILE=datei_Stroe_Winkel,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_Stroe_Winkel, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN  (3, FILE=datei_Stroe_Winkel, STATUS='NEW')

      grad = 180./PI
      WRITE (3, 10)
      WRITE (3, 20)
      WRITE (3, 30)
      WRITE (3, 40) PROJEKT
      WRITE (3, 50) 
      WRITE (3, 55) 
      do i=1,Nprofile
         WRITE (3, 60) i, (d_2(i)/1000), beta_1(i)*grad, beta_2(i)*grad
      end do
      CLOSE (3)
C      WRITE (6,5)
  
      RETURN
      END




C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Geschwindigkeitsdreiecke an Ein- und Aus-          I
C    I       tritt des Laufrades.                                           I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  14. 11. 96

      SUBROUTINE AUSGABE_DREIECKE_LAUF

      INCLUDE 'common2.f'


      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT

      PI = 3.14159

    5 FORMAT ('   Ge-Dreiecke.dat ausgegeben')
   10 FORMAT ('    --------------------------------------------    ')
   20 FORMAT ('    Geschwindigkeitsdreiecke an der Laufschaufel :  ')
   30 FORMAT ('    --------------------------------------------  '  )
   40 FORMAT (/, '    Laufrad-Nr., Projekt ....... : ', A25 ,//)
   50 FORMAT (//,'  Profil  D1     c1    cm1    cu1     u1  ',
     .        ' alpha1    w1     wu1   Beta1 ')
   55 FORMAT ('         [m]   [m/s]  [m/s]  [m/s]   [m/s]  [o] ',
     .        '   [m/s]   [m/s]  [o]', /)
   70 FORMAT (//,'  Profil  D2     c2    cm2    cu2     u2  ',
     .        ' alpha2    w2     wu2   Beta2 ')
   80 FORMAT ('  Profil  D0     c0    cm0    cu0     u0   alpha0 ',
     .        '   w0    wu0   Beta0 ')
   60 FORMAT (I5, F8.3, 3F7.3, F8.3, F6.1, 2F8.3, F7.2)
   65 FORMAT (I5, F8.3, 3F7.3, F8.3, F6.1, 2F8.3, F6.2, ///)
  110 FORMAT ('    -------------------------------------------    ')
  120 FORMAT ('    Geschwindigkeitsdreiecke am Leitradaustritt :  ')
  130 FORMAT ('    -------------------------------------------  ',/)
 

      INQUIRE(FILE=datei_Ge_Dreiecke,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_Ge_Dreiecke, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      OPEN  (3, FILE=datei_Ge_Dreiecke, STATUS='NEW')

      grad = 180./PI
      WRITE (3, 40) PROJEKT
      WRITE (3, 110)
      WRITE (3, 120)
      WRITE (3, 130)
C     --------------------- Austritt Leitrad -------------------
      WRITE (3, 80)
      WRITE (3, 55) 
      i = 1
      d_ = D0i
      beta = beta_0i*grad
      vmer = cm0i
      um   = u0i
      wu   = wu0i
      wrel = w0i
      cu   = cu0i
      vabs = c0i
      alfa = alpha_0i*grad
      WRITE (3, 65) i, d_, vabs, vmer, cu, um, alfa, 
     .                     wrel, wu, beta
C
C     --------------------- Eintritt Laufrad -------------------
      WRITE (3, 10)
      WRITE (3, 20)
      WRITE (3, 30)
      WRITE (3, 50) 
      WRITE (3, 55) 
      do i=1,Nprofile
         d_ = d_1(i)
         beta = beta_1(i)*grad
         vmer = cm1
         um   = u1(i)
         wu   = wu1(i)
         wrel = w1(i)
         cu   = cu1(i)
         vabs = c1(i)
         alfa = alpha1(i)*grad
         WRITE (3, 60) i, d_, vabs, vmer, cu, um, alfa, 
     .                        wrel, wu, beta
      end do
C     --------------------- Austritt Laufrad -------------------
      WRITE (3, 70) 
      WRITE (3, 55) 
      do i=1,Nprofile
         d_ = d_2(i)/1000.
         beta = beta_2(i)*grad
         vmer = cm2
         um   = u2(i)
         wu   = wu2(i)
         wrel = w2(i)
         cu   = cu2(i)
         vabs = c2(i)
         alfa = alpha2(i)*grad
         WRITE (3, 60) i, d_, vabs, vmer, cu, um, alfa, 
     .                        wrel, wu, beta
      end do
      CLOSE (3)
C      WRITE (6,5)
  
      RETURN
      END









C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Schaufelprofile in karthesischen Koordinaten.      I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  24. 9. 96

      SUBROUTINE AUSGABE_XYZ_KOORDINATEN

      INCLUDE 'common.f'

C      REAL x_ds(IMAX,npla), y_ds(IMAX,npla), z_ds(IMAX,npla),
C     .     x_ss(IMAX,npla), y_ss(IMAX,npla), z_ss(IMAX,npla)


   10 FORMAT ('   schaufel_xyz.dat ausgegeben '  )
   40 FORMAT(1X,2I5)
   50 FORMAT(4X,6(F9.3,1X))


      INQUIRE(FILE=datei_schaufel_xyz,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_schaufel_xyz, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      open (20,file=datei_schaufel_xyz,status='new')

      WRITE(20,40) ZDP, Nprofile
      do j=1,Nprofile
         do i=1,ZDP
            WRITE(20,50) x_ds(i,j), y_ds(i,j), z_ds(i,j),
     .                   x_ss(i,j), y_ss(i,j), z_ss(i,j)
C            print*, x_ds(i,j)
         end do
      end do
      close (20)
  
C      WRITE (6, 10)
  
      RETURN
      END

C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Schaufelprofile in der Abwicklung.                 I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  22. 9. 96

      SUBROUTINE AUSGABE_ABWICKLUNG

      INCLUDE 'common.f'
   10 FORMAT ('   zwischen_prof.dat ausgegeben '  )
   40 FORMAT(1X,2I5)
   50 FORMAT(4X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3)


      INQUIRE(FILE=datei_zwischen_prof,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_zwischen_prof, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      open (20,file=datei_zwischen_prof,status='unknow')

      WRITE(20,40) ZDP, Nprofile
      do j=1,Nprofile
         do i=1,ZDP
            WRITE(20,50) x_sl_ab(i,j), y_sl_ab(i,j),
     .                   x_ds_ab(i,j), y_ds_ab(i,j),
     .                   x_ss_ab(i,j), y_ss_ab(i,j)
         end do
         WRITE(20,50) x_sl_dpkt(j), y_sl_dpkt(j)     
      end do
      close (20)
  
C      WRITE (6, 10)
  
      RETURN
      END




C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I   Ausgabe der Schaufelprofile in der Abwicklung fuer Hr. Batrekhy.   I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  4. 10. 96

      SUBROUTINE AUSGABE_BATREKHY

      INCLUDE 'common.f'
   10 FORMAT ('   Batrekhy_prof.dat ausgegeben '  )
   40 FORMAT(1X,2I5)
   50 FORMAT(4X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3,1X,F8.3)


      INQUIRE(FILE=datei_Batrekhy_prof,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_Batrekhy_prof, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      open (20,file=datei_Batrekhy_prof,status='unknow')

      WRITE(20,40) ZDP, Nprofile
      do j=1,Nprofile
         do i=1,ZDP
            WRITE(20,50) x_ds_ab(i,j), y_ds_ab(i,j)
         end do
         do i=1,ZDP
            WRITE(20,50) x_ss_ab(i,j), y_ss_ab(i,j)
         end do
      end do
      close (20)
  
C      WRITE (6, 10)
  
      RETURN
      END
      
      
      
      
      
      
      
      
      
      
      

      SUBROUTINE schreibe_prototyp

      INCLUDE 'common.f'

c      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
c      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT
      COMMON /maxw/  mwi,mwa
      REAL           mwi,mwa


      INTEGER  UNIT
    5 FORMAT ('')
   10 FORMAT ('Dieses File enthaelt die Betriebspunktdaten fuer den Prot
     .otyp')
   11 FORMAT ('---------------------------------------------------------
     .----')
   12 FORMAT (A25)
   13 FORMAT ('Wassermenge ........................... Q =', F8.2 ,
     .' m3/s')
   14 FORMAT ('Fallhoehe ............................. H =', F8.2 ,' m')
   15 FORMAT ('Drehzahl .............................. n =', F8.2 ,
     .' U/min')
   16 FORMAT ('Bezugsdurchmesser .................... D2 =', F8.2 ,' m')
   17 FORMAT ('Durchmesserverhaeltnis ............ D1/D2 =', F8.4 )
   18 FORMAT ('Anzahl La-Profile .............. Nprofile =', I8	)
   19 FORMAT ('Anzahl Laufschaufeln............... Nlauf =', I8	)
   20 FORMAT ('Umschlingung ............................ =', F8.2 )
   21 FORMAT ('Lage des Maximums Austritt ......... max2 =', F8.2 )
   22 FORMAT ('Profileinschnuerung Austritt innen . pe2i =', F8.2 )
   23 FORMAT ('Profileinschnuerung Austritt aussen  pe2a =', F8.2 )
   24 FORMAT ('Lage des Maximums Eintritt ......... max1 =', F8.2 )
   25 FORMAT ('Profileinschnuerung Eintritt innen . pe1i =', F8.2 )
   26 FORMAT ('Profileinschnuerung Eintritt aussen  pe1a =', F8.2 )
   27 FORMAT ('Profildicke an der Nabe ......... dicke_i =', F8.1 )
   28 FORMAT ('Profildicke am Kranz ............ dicke_a =', F8.1 )
   29 FORMAT ('norm. Stg. Profildicke Kranz . d_strich_a =', F8.1 )
   30 FORMAT ('Winkeluebertreibung ................ db2i =', F8.1 )
   31 FORMAT ('Winkeluebertreibung ................ db2a =', F8.1 )
   32 FORMAT ('Winkeluebertreibung ................ db1i =', F8.1 )
   33 FORMAT ('Winkeluebertreibung ................ db1a =', F8.1 )
   34 FORMAT ('Maximale Woelbung innen.............. mwi =', F8.2 )
   35 FORMAT ('Maximale Woelbung aussen............. mwa =', F8.2 )
   36 FORMAT ('Anzahl Leitschaufeln............... Nleit =', I8	)
   37 FORMAT ('Leitradmitte ......................... D0 =', F8.2 ,' m')
   38 FORMAT ('Leitradhoehe ......................... B0 =', F8.2 ,' m')
   39 FORMAT ('Leitradlaenge Drehachse-Hinterkante       =', F8.2 ,' m')
   40 FORMAT ('Kommentar zur Charakterisierung des Betriebspunktes (<20 
     .Char.):')
   41 FORMAT (A25)

      INQUIRE(FILE=datei_prototyp,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_prototyp, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      


      OPEN  (3, FILE=datei_prototyp, STATUS='new')

      WRITE (3, 10) 
      WRITE (3, 11)       
      WRITE (3, 5)
      WRITE (3, 12) PROJEKT 
      WRITE (3, 5)
      WRITE (3, 13) Q
      WRITE (3, 14) H
      WRITE (3, 15) n
      WRITE (3, 16) d2
      WRITE (3, 17) di_da
      WRITE (3, 5)
      WRITE (3, 18) Nprofile
      WRITE (3, 19) nlschaufel
      WRITE (3, 5)
      WRITE (3, 20) umschlingung
      WRITE (3, 5)
      WRITE (3, 21) max2
      WRITE (3, 22) pe2i
      WRITE (3, 23) pe2a
      WRITE (3, 5)
      WRITE (3, 24) max1
      WRITE (3, 25) pe1i
      WRITE (3, 26) pe1a
      WRITE (3, 5)
      WRITE (3, 27) dicke_i
      WRITE (3, 28) dicke_a
      WRITE (3, 29) d_strich_a
      WRITE (3, 5)
      WRITE (3, 30) db2i
      WRITE (3, 31) db2a
      WRITE (3, 32) db1i
      WRITE (3, 33) db1a
      WRITE (3, 5)
      WRITE (3, 34) mwi
      WRITE (3, 35) mwa
      WRITE (3, 5)
      WRITE (3, 36) Nleit
      WRITE (3, 37) d0
      WRITE (3, 38) b0
      WRITE (3, 39) leit_ax_hi
      WRITE (3, 5)
      WRITE (3, 40)
      WRITE (3, 41) KOMMENTAR

      
      CLOSE (3)
 
      

  
      RETURN
      END






      SUBROUTINE schreibe_randbedingung

      INCLUDE 'common.f'




      PI = 3.14159
      grad = 180./PI





    5 FORMAT ('')
   10 FORMAT ('RB-typ (Ausl/Last) ...............rb_type = Ausl')
   11 FORMAT ('Profil-typ (Block/Turb) ..........pr_type = Block')
   12 FORMAT (A25)
   13 FORMAT ('Volumenstrom gemaess Auslegung .... Q_aus =', F8.2 ,
     .' m3/s')
   14 FORMAT ('Lastpunkt (Q=Lp*Q_aus)................ Lp =    1.00')
   15 FORMAT ('Fallhoehe ............................. H =', F8.2 ,' m')
   16 FORMAT ('Drehzahl .............................. n =', F8.2 ,
     .' U/min')
   17 FORMAT ('Bezugsdurchmesser .................... D2 =', F8.2 ,' m')
   18 FORMAT ('Durchmesserverhaeltnis ............ D1/D2 =', F8.4 )

   20 FORMAT ('Leitradmitte ......................... D0 =', F8.2 ,' m')
   21 FORMAT ('Leitradhoehe ......................... B0 =', F8.2 ,' m')
   22 FORMAT ('Leitradlaenge Drehachse-Hinterkante       =', F8.2 ,' m')

   30 FORMAT ('Anzahl La-Profile .............. Nprofile =', I8	)

   40 FORMAT ('    ------------------------------------    ')
   41 FORMAT ('    Stroemungs- und Schaufelwinkel:  ')
   42 FORMAT ('    ------------------------------------  ',/)
   43 FORMAT ('    Profil  Durchmesser   Beta1       Beta2     ',
     #        'Beta1_sch   Beta2_sch')
   44 FORMAT ('                [m]        [o]         [o]      ',
     #        '   [o]         [o]', /)
   45 FORMAT (I8,1X,F11.6,1X,F11.6,1X,F11.6,1X,F11.6,1X,F11.6)






      INQUIRE(FILE=datei_randbedingung,EXIST=EX)
      if (EX .EQ. .TRUE.) then
          OPEN (3, FILE=datei_randbedingung, STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF


      


      OPEN  (3, FILE=datei_randbedingung, STATUS='new')


      WRITE (3, 5)
      WRITE (3, 12) PROJEKT 
      WRITE (3, 5)
      WRITE (3, 10)
      WRITE (3, 5)
      WRITE (3, 13) Q
      WRITE (3, 14) 
      WRITE (3, 5)
      WRITE (3, 15) H
      WRITE (3, 16) n
      WRITE (3, 5)
      WRITE (3, 17) d2
      WRITE (3, 18) di_da
      WRITE (3, 5)
      WRITE (3, 20) d0
      WRITE (3, 21) b0
      WRITE (3, 22) leit_ax_hi
      WRITE (3, 5)
      WRITE (3, 30) Nprofile
      WRITE (3, 5)
      WRITE (3, 40)
      WRITE (3, 41)
      WRITE (3, 42)
      WRITE (3, 43)
      WRITE (3, 44)
      do i=1,Nprofile
         WRITE (3, 45) i,(d_2(i)/1000),beta_1(i)*grad, beta_2(i)*grad,
     #                 beta1_sch(i),beta2_sch(i)
      end do

      CLOSE (3)
 
      

  
      RETURN
      END
