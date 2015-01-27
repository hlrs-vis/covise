C                      *****************************
C                      *                           *
C                      *    EINGABE_STEUERFILE     *
C                      *                           *
C                      *****************************
    
    
C    +-----------------------------------------------------------------+
C    I                                                                 I
C    I  Liest die Angaben fuer die Auswahl der Bilder vom File         I
C    I  BILDER.DAT .                                                   I
C    I                                                                 I
C    +-----------------------------------------------------------------+
C                                        E. Goede SEWZ-EPFL, 28. 10. 87

      SUBROUTINE EINGABE_STEUERFILE
   
      
      INCLUDE     'common.f'
  
 
    
   10 FORMAT ( ' steuerfile.dat eingelesen ')
   20 FORMAT ( 33X, I3 )
   30 FORMAT ( A20  )
      

      INQUIRE(FILE=datei_steuerfile,EXIST=EX)
      if (EX .EQ. .TRUE.) then


 	 OPEN  (3, FILE=datei_steuerfile, STATUS='OLD')
       
 	 CALL ZEILE (3, 3)
 	 READ (3, 20) WRSPEZIFISCH
 	 READ (3, 20) WRBETRIEBSPUNKT
 	 READ (3, 20) WRMEKO
 	 READ (3, 20) WROUTPUT
 	 READ (3, 20) WRPROFILE_NEU
 	 READ (3, 20) WRZWISCHEN_FILE
 	 READ (3, 20) WRLA_WINKEL
 	 READ (3, 20) WRSTROE_WINKEL
 	 READ (3, 20) WRGE_DREIECKE
 	 READ (3, 20) WRZWISCHEN_PROF
 	 READ (3, 20) WRSCHAUFEL_XYZ
 	 READ (3, 20) WRBATREKHY
 	 CLOSE (3)
  
      ELSE  
         lesenok=0
      ENDIF
      
   
      RETURN
      END






C                          ******************************
C                          *                            *
C                          *    EINLESEN_ZENTRALFILE    *
C                          *                            *
C                          ******************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Einlesen des Zentralfiles mit den Koordinaten des Laufrades.        I
C    I                                                                      I
C    I  Filename:  ZENTRAL_FILE.DAT                                         I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  SEWZ,  14. 12. 91

      SUBROUTINE EINLESEN_ZENTRALFILE 
  
 

      INCLUDE   'common.f'


      INTEGER i
     
 
      
      INQUIRE(FILE=datei_zentral_file,EXIST=EX)      
      if (EX .EQ. .TRUE.) then

  
         OPEN (3, FILE=datei_zentral_file, STATUS='OLD')

         
        z_axe_la=-0.45
        d2_kugel=1.00

         READ       (3, *) numnabe, anzkranz
         READ       (3, *) posnabe,poskranz
         CALL ZEILE (3, 3)

c        print*,anznabe,anzkranz
c        print*,posnabe,poskranz
         do i=1,numnabe

            READ    (3, *) r_nabe(i), z_nabe(i)
c            print*,r_nabe(i),z_nabe(i)
         end do
c         print*,'fertig'
         CALL ZEILE (3, 3)

         do i=1,anzkranz
            READ (3, *) r_kranz(i), z_kranz(i)
c            print*,r_kranz(i),z_kranz(i)
         end do           
             
 	 CLOSE (3)

      ELSE  
         lesenok=0
      ENDIF

      RETURN
      END







C Zeile                   *************************
C                         *                       *
C                         *    Prozedur ZEILE     *
C                         *                       *
C                         *************************
   
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Liest eine Anzahl Zeilen von einem Datenfile. Zweck ist, Textzeilen I
C    I  beim Lesen eines Files zu ueberspringen.                            I
C    I                                                                      I
C    I     - IFILE ................... Kanalnummer,                         I
C    I     - N ....................... Anzahl Zeilen.                       I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  SEWZ,  21. 7. 86
      SUBROUTINE ZEILE (IFILE, N)
  
      CHARACTER*2 TEXT
   10 FORMAT (A2)
   
      DO I=1,N
         READ (IFILE, 10) TEXT
      END DO
  
      RETURN
      END




C                          ******************************
C                          *                            *
C                          *    EINGABE_PROTOTYP.DAT    *
C                          *                            *
C                          ******************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Eingabe der Prototypdaten Q, H, n ....                              I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  3. 6. 96

      SUBROUTINE EINGABE_Q_H_N

      INCLUDE 'common.f'

c      CHARACTER*25 PROJEKT, KOMMENTAR, SCHAUFEL_BEZ
c      COMMON /PROJE/ SCHAUFEL_BEZ, KOMMENTAR, PROJEKT
      COMMON /maxw/  mwi,mwa
      REAL           mwi,mwa
      
      INTEGER  UNIT
   10 FORMAT ( ' prototyp.dat eingelesen ')
   20 FORMAT ( 43X, F8.3 )
   30 FORMAT ( A20  )
   40 FORMAT ( 43X, I8  )
   50 FORMAT (A25)
   60 FORMAT (A,A)
c verzeichnis , 'prototyp.dat'

      INQUIRE(FILE=datei_prototyp,EXIST=EX)
      if (EX .EQ. .TRUE.) then



      	  OPEN  (3, FILE=datei_prototyp, STATUS='OLD')

      	  CALL ZEILE (3, 3)
      	  READ (3, 50) PROJEKT
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) volumen
      	  READ (3, 20) fallhoehe
      	  READ (3, 20) drehzahl
      	  READ (3, 20) d2
      	  READ (3, 20) di_da
      	  CALL ZEILE (3, 1)
      	  READ (3, 40) Nprofile
      	  READ (3, 40) nlschaufel
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) umschlingung
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) max2
      	  READ (3, 20) pe2i
      	  READ (3, 20) pe2a
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) max1
      	  READ (3, 20) pe1i
      	  READ (3, 20) pe1a
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) dicke_i
      	  READ (3, 20) dicke_a
      	  READ (3, 20) d_strich_a
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) versch_i
      	  READ (3, 20) versch_a
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) db2i
      	  READ (3, 20) db2a
      	  READ (3, 20) db1i
      	  READ (3, 20) db1a
      	  CALL ZEILE (3, 1)
      	  READ (3, 20) mwi
      	  READ (3, 20) mwa
      	  CALL ZEILE (3, 1)
      	  READ (3, 40) Nleit
      	  READ (3, 20) d0
      	  READ (3, 20) b0
      	  READ (3, 20) l0
      	  READ (3, 20) leit_dr
      	  CALL ZEILE (3, 2)
      	  READ (3, 30) KOMMENTAR

      	  Q = volumen
      	  H = fallhoehe
      	  N = drehzahl
      	  HM = fallhoehe_m
      	  eta_h  = eta_h/100.
      	  eta_el = eta_el/100.
      	  eta_m  = eta_m/100.

      	  CLOSE (3)
      
      ELSE  
         lesenok=0
      ENDIF


  
      RETURN
      END





C                       *****************************
C                       *                           *
C                       *    EINGABE_BASISPROFIL    *
C                       *                           *
C                       *****************************
    
    
C    +-----------------------------------------------------------------+
C    I                                                                 I
C    I  Liest die Profilkoordinaten fuer das Basisprofil               I
C    I  vom File 'NAME'.DAT .                                          I                                                I
C    I                                                                 I
C    +-----------------------------------------------------------------+
C                                        E. Goede IHS, 21. 9. 96

      SUBROUTINE EINGABE_BASISPROFIL
   
      INCLUDE     'common.f'
  
    5 FORMAT (' ',A9,' eingelesen ')
    

      INQUIRE(FILE=datei_profil,EXIST=EX)
      if (EX .EQ. .TRUE.) then

      
 	  OPEN (10,FILE=datei_profil,STATUS='OLD')
 	  READ(10,*) ZDP
 	  DO 13 I=1,ZDP,1
	    READ(10,*) XL_basprof(I), DL_basprof(I)
            
13	  CONTINUE
 	  CLOSE(10)


 
 
      ELSE  
         lesenok=0
      ENDIF
      

      RETURN
      END







C                       ************************************
C                       *                                  *
C                       *    EINGABE_LEITSCHAUFELPROFIL    *
C                       *                                  *
C                       ************************************
    
    
C    +-----------------------------------------------------------------+
C    I                                                                 I
C    I  Liest die Profilkoordinaten fuer das Basisprofil               I
C    I  vom File 'NAME'.DAT .                                          I                                                I
C    I                                                                 I
C    +-----------------------------------------------------------------+
C                                        E. Goede IHS, 21. 9. 96

      SUBROUTINE EINGABE_LEITPROFIL
   	
      INCLUDE     'common.f'
  
    5 FORMAT (' ',A9,' eingelesen ')
    

      INQUIRE(FILE=datei_leitprofil,EXIST=EX)
      if (EX .EQ. .TRUE.) then

      
 	  OPEN (10,FILE=datei_leitprofil,STATUS='OLD')
 	  READ(10,*) num_leit_pts
 	  DO 13 I=1,num_leit_pts,1
	    READ(10,*) x_leit_prof(I), y_leit_prof(I)
              y_leit_prof(I)=y_leit_prof(I)*0.8
              x_leit_prof(i)=((x_leit_prof(i)/100.0)**1.2)*100.0
13	  CONTINUE
 	  CLOSE(10)

          

 
      ELSE  
         lesenok=0
      ENDIF
      

      RETURN
      END










C                        ***********************
C                        *                     *
C                        *    EINGABE_DATEN    *
C                        *                     *
C                        ***********************
    
    
C    +-----------------------------------------------------------------+
C    I                                                                 I
C    I  Liest die Parameter fuer die Profile vom File DATEN.DAT .      I                                                I
C    I                                                                 I
C    +-----------------------------------------------------------------+
C                                        E. Goede IHS, 21. 9. 96

      SUBROUTINE EINGABE_DATEN
   
      INCLUDE     'common.f'

   10 FORMAT ( ' daten.dat eingelesen ')


      INQUIRE(FILE=datei_daten,EXIST=EX)
      if (EX .EQ. .TRUE.) then



 	  OPEN (3,FILE=datei_daten,STATUS='OLD')
 	  read(3,*) BETA1
 	  read(3,*) BETA2
 	  read(3,*) DS
 	  read(3,*) AL
 	  read(3,*) Z
 	  read(3,*) dzul
 	  read(3,*) sdpktzul
 	  READ(3,'(A10)') NAME
 	  CLOSE (3)
     
      ELSE  
         lesenok=0
      ENDIF

      RETURN
      END



C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Eingabe der relativen Schaufeldicke des Laufrades als Vertei-  I
C    I       von der Nabe zum Kranz (d/l = f(r).                            I
C    I                                                                      I
C    I       -  vom File dicke.dat                                          I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  22. 9. 96

      SUBROUTINE EINGABE_PROFILDICKE

      INCLUDE 'common.f'

   10 FORMAT (' dicke.dat eingelesen '  )


      INQUIRE(FILE=datei_dicke,EXIST=EX)
      if (EX .EQ. .TRUE.) then



 	  OPEN  (3, FILE=datei_dicke, STATUS='OLD')

 	  call ZEILE (3, 2)
 	  do i=1,Nprofile
 	     READ (3, *) dzul_r(i), d_theta2(i), d_theta1(i)
 	  end do
 	  CLOSE (3)
  
      ELSE  
         lesenok=0
      ENDIF

  
      RETURN
      END




C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I       Ausgabe der Schaufelprofile in karthesischen Koordinaten.      I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  24. 9. 96

      SUBROUTINE eingabe_xyz()

      INCLUDE 'common.f'




      INQUIRE(FILE=datei_schaufel_xyz,EXIST=EX)
      if (EX .EQ. .TRUE.) then
      


 	  open (20,file=datei_schaufel_xyz,status='OLD')

 	  READ(20,*) ZDP, Nprofile
 	  do j=1,Nprofile
 	     do i=1,ZDP
 	  	READ(20,*) x_ds(i,j), y_ds(i,j), z_ds(i,j),
     .	  		   x_ss(i,j), y_ss(i,j), z_ss(i,j)
 	     end do
 	  end do
 	  close (20)
	  success=1
          lesenok=1
      else
          success=0

      endif

  
      RETURN
      END
