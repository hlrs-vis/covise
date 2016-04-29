C                          ******************************
C                          *                            *
C                          *    UMRECHNEN_AUF_MODELL    *
C                          *                            *
C                          ******************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Umrechnung der Prototypdaten Q, H, n ... auf Modellmassstab.        I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  3. 6. 96

      SUBROUTINE UMRECHNEN_AUF_MODELL

      INCLUDE 'common2.f'

      PI = 3.14159
      G  = 9.806

      NM = N * d2/D2M * SQRT ( HM/H )

      RETURN
      END
C                          ************************
C                          *                      *
C                          *    RECHNE_KCM_KU     *
C                          *                      *
C                          ************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Umrechnung der Prototypdaten Q, H, n ...  in Kcm, Ku, ....          I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  3. 6. 96

      SUBROUTINE RECHNE_KCM_KU

      INCLUDE 'common2.f'

      PI = 3.14159
      G  = 9.806

      VAUBEZ = SQRT ( 2.*G*H )
      U2a  = d2*N*PI / 60.
      KU2  = U2a / VAUBEZ
      KCM2 = Q / (d2**2 * PI/4.*VAUBEZ)
      Nq   = N*SQRT(Q) / H**0.75
      N1str = N*d2 / SQRT ( H )
      Q1str = Q / (d2**2 * SQRT (H) )

      RETURN
      END

C                          ***************************
C                          *                         *
C                          *   RECHNE_WINKEL_STROE   *
C                          *                         *
C                          ***************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Ein- und Austrittswinkel fuer das Laufrad.           I
C    I  Voraussetzung ist die Kenntnis der Radien der jeweiligen Station.   I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  8. 6. 96

      SUBROUTINE RECHNE_WINKEL_STROE

      INCLUDE 'common2.f'

      INTEGER leit
      REAL r
      PI = 3.14159
      g  = 9.806
      
      leit = 0
      r = 1.
 
C
C    ------------------ Laufradaustritt (Ebene 2) --------------------
C
C      di_da = 0.4375

 
      D2i = d2 * di_da
      D2a = d2
      cm2 = Q / (PI/4.*(d2**2)*( 1.0 - di_da**2 ))
      do i=1,Nprofile
         d_2(i) = D2i + float(i-1)/float(Nprofile-1) * (D2a - D2i)
         u2_i  = d_2(i)*N*PI / 60.
         wu2_i = u2_i
	 
	 if (leit.EQ.1) wu2_i = r*g*H / u2_i
	 
         beta_2(i) = ATAN2 (cm2, wu2_i)
	 
	 
      end do
C
C    ------------------- Laufradeintritt (Ebene 1) -------------------
C
      
      D1i = d2 * di_da
      D1a = d2
      cm1 = Q / (PI/4.*(D1a**2)*( 1.0 - di_da**2 ))
      do i=1,Nprofile
         d_1(i) = D1i + float(i-1)/float(Nprofile-1) * (D1a - D1i)
         u1_i  = d_1(i)*N*PI / 60.
         cu1_i = g*H / u1_i
         wu1_i = u1_i - cu1_i
         beta_1(i) = ATAN2 (cm1, wu1_i)
	 
	 if (leit.EQ.1) beta_1(i) =0.5 * PI
	 
      end do

      RETURN
      END

C                          ******************************
C                          *                            *
C                          *    WINKEL_UEBERTREIBUNG    *
C                          *                            *
C                          ******************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Winkeluebertreibung gegenueber den Eulerschen Stroe- I
C    I  mungswinkeln an Ein- und Austrittswinkel fuer Lauf- und Leitrad.    I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  9. 6. 96

      SUBROUTINE LAUF_WINKEL_UEBERTREIBUNG

      INCLUDE 'common2.f'
      REAL           beta1_sch, beta2_sch,
     .               delta_beta1, delta_beta2     
      common /WINK/  beta1_sch(npla), beta2_sch(npla),
     .               delta_beta1(npla), delta_beta2(npla)
      REAL rtemp, delta_r, faktor_a, faktor_b

      PI = 3.14159
      g  = 9.806
      grad = 180./PI
 
C
C    ------------------ Laufradaustritt (Ebene 2) --------------------
C

C    Winkeluebertreibung mit linearer Verteilung

      delta_b2i = db2i * PI/180.
      delta_b2a = db2a * PI/180.
      do i=1,Nprofile
         faktor = float(i-1)/float(Nprofile-1)
         delta_beta2(i) = delta_b2i - faktor * (delta_b2i-delta_b2a)
         beta2_sch(i) = beta_2(i) - delta_beta2(i)

         delta_beta2(i) = delta_beta2(i) * grad 
         beta2_sch(i) = beta2_sch(i) * grad 


      end do
C
C    ------------------- Laufradeintritt (Ebene 1) -------------------
C
      delta_b1i = db1i * PI/180.
      delta_b1a = db1a * PI/180.
      db_strich_a=0.3
      delta_r=(d_2(Nprofile)-d_2(1))/2
      faktor_b=(db1i-db1a)/delta_r * db_strich_a
      faktor_a=(db1i-faktor_b*delta_r-db1a)/delta_r**2.0

      do i=1,Nprofile



         faktor = float(i-1)/float(Nprofile-1)
         rtemp=delta_r*(Nprofile-i)/(Nprofile-1)               
         delta_beta1(i) = faktor_a*rtemp**2.0+faktor_b*rtemp+db1a        
         
         beta1_sch(i) = beta_1(i)*grad + delta_beta1(i)

    
      end do
C
C    ------------------- Leitradaustritt (Ebene 0) -------------------
C
      RETURN
      END
C                        *****************************
C                        *                           *
C                        *    RECHNE_LEITRAD_EK_AK   *
C                        *                           *
C                        *****************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung von Eintritts- und Austrittskante des Leitapparates.     I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  1. 7. 96

      SUBROUTINE RECHNE_LEITRAD_EK_AK

      INCLUDE 'common.f'

      REAL l_hkante      

      pi = 3.14159
      g  = 9.806

      alpha_0=atan(N*Q/60/g/H/b0)
      l_hkante=l0*(1-leit_dr)
      
      D0i = 2.0*(((d0/2.0-l_hkante*sin(alpha_0))**2.0
     #           +(l_hkante*cos(alpha_0))**2.0)**0.5)



c      D0a = d0 + 2*l0*(leit_dr) * sin(alpha_0)

      RETURN
      END



C                          ***************************
C                          *                         *
C                          *   RECHNE_WINKEL_LEIT    *
C                          *                         *
C                          ***************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Ein- und Austrittswinkel fuer das Leitrad.           I
C    I  Voraussetzung ist die Kenntnis der Radien der jeweiligen Station.   I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  2. 7. 96

      SUBROUTINE RECHNE_WINKEL_LEIT

      INCLUDE 'common.f'

      PI = 3.14159
      g  = 9.806
 
C
C    ------------------ Leitradaustritt (Ebene 0i) --------------------
C
C     print*,b0
      cm0i = Q/(PI*D0i*b0)
      u0i   = PI*D0i*N/60.
      cu0i = g*H / u0i
      c0i  = sqrt(cm0i**2 + cu0i**2)
      alpha_0i = ATAN2 (cm0i, cu0i)      
      wu0i = u0i - cu0i
      w0i  = sqrt(cm0i**2 + wu0i**2)
      beta_0i = ATAN2 (cm0i, wu0i)      
C
C    ------------------- Leitradeintritt (Ebene 0a) -------------------
C
      RETURN
      END




C                          ***************************
C                          *                         *
C                          *   RECHNE_GE_DREIECKE    *
C                          *                         *
C                          ***************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Geschwindigkeitsdreiecke an Ein- und Austritt des    I
C    I  Laufrades.                                                          I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  15. 11. 96

      SUBROUTINE RECHNE_GE_DREIECKE

      INCLUDE 'common2.f'

      PI = 3.14159
      g  = 9.806
 
C
C    ------------------ Laufradaustritt (Ebene 2) --------------------
C
C     D2izuD2a = 0.4375
      D2i = d2 * di_da
      D2a = d2
      cm2 = Q / (PI/4.*(d2**2)*( 1.0 - di_da**2 ))
      do i=1,Nprofile
         d_2(i) = D2i + float(i-1)/float(Nprofile-1) * (D2a - D2i)
         u2(i)  = d_2(i)*N*PI / 60.
         cu2(i) = 0.
         wu2(i) = u2(i)
         w2(i)  = sqrt(cm2**2 + wu2(i)**2)
         c2(i)  = sqrt(cm2**2 + cu2(i)**2)
         alpha2(i) = ATAN2 (cm2, cu2(i))    
         d_2(i) = d_2(i) * 1000.
      end do
C
C    ------------------- Laufradeintritt (Ebene 1) -------------------
C
     
      D1i = d2 * di_da
      D1a = d2
      cm1 = Q / (PI/4.*(D1a**2)*( 1.0 - di_da**2 ))
      do i=1,Nprofile
         d_1(i) = D1i + float(i-1)/float(Nprofile-1) * (D1a - D1i)
         u1(i)  = d_1(i)*N*PI / 60.
         cu1(i) = g*H / u1(i)
         wu1(i) = u1(i) - cu1(i)
         w1(i)  = sqrt(cm1**2 + wu1(i)**2)
         c1(i)  = sqrt(cm1**2 + cu1(i)**2)
         alpha1(i) = ATAN2 (cm1, cu1(i))      
      end do

      RETURN
      END


C                          ***********************************
C                          *                                 *
C                          *   RECHNE_UMLENKUNG_NABE_KRANZ   *
C                          *                                 *
C                          ***********************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Umlenkung am Kranz                                   I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           A. Kaps  IHS,  01. 08. 00

      SUBROUTINE RECHNE_UMLENKUNG_NABE_KRANZ

      INCLUDE 'common.f'

      REAL xmax,ymax,t
      INTEGER i,k

      xmax=uml_r
      ymax=uml_z
      
  150 FORMAT (2X, 2F14.6)
 
      INQUIRE(FILE="umlenkung.dat",EXIST=EX)
      if (EX .EQV. .TRUE.) then
          OPEN (3, FILE="umlenkung.dat", STATUS='OLD')
          CLOSE(3,STATUS='DELETE')
      ENDIF      
      
      OPEN (3, FILE="umlenkung.dat", STATUS='NEW')

 
      do i=1,11
         k=i+poskranz-1
         t=(i-1)/10.0
         
         r_kranz(k) = 0.5+xmax*(t**2.0)-2.0*xmax*t+xmax
         z_kranz(k) = -ymax*(t**2.0)
c         print*,k,t,r_kranz(k),z_kranz(k)
 
c            WRITE (3,*) r_kranz(k), z_kranz(k)    
            WRITE (3,150) 1340.*r_kranz(k), 1340.*z_kranz(k)    
      end do
      
      CLOSE (3)


      RETURN
      END



C                          ***********************************
C                          *                                 *
C                          *   RECHNE_UMLENKUNG_NABE_KRANZ   *
C                          *                                 *
C                          ***********************************
    
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I  Berechnung der Umlenkung am Kranz                                   I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           A. Kaps  IHS,  01. 02. 99

      SUBROUTINE rechne_kugel

      INCLUDE 'common.f'

      INTEGER anzkugel
      INTEGER i

      anzkugel=0
      if (d2_kugel.gt.1.0) anzkugel=10
c      d2_kugel=1.07
      anznabe=numnabe+anzkugel
      
      alpha=acos(1/d2_kugel)
c     print*,d2_kugel

      do i=1,anzkugel
         alpha_temp=alpha-(2.0*alpha)*(i-1)/(anzkugel-1)   
c        print*,alpha_temp
         r_nabe(numnabe+i)=d2_kugel/2*cos(alpha_temp)
         z_nabe(numnabe+i)=d2_kugel/2*sin(alpha_temp)
     #                       +z_axe_la 
      end do 



      RETURN
      END

