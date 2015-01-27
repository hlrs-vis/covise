
C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I    Berechnung der abgewickelten Profile mit den Ein- und Austritts-  I
C    I    winkel des Laufrades.                                             I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  24. 9. 96

      SUBROUTINE RECHNE_XYZ_KOORDINATEN

      INCLUDE 'common.f'
      REAL ix, yps, zet, uh, er, sp

      theta_0 = 360./real(nlschaufel)
      z_axe_la=-0.45

      do j=1,Nprofile
         
         er = d_2(j)/2.
        
         
         do i=1,ZDP
            uh  = - y_ss_ab(i,j)
            zet = - x_ss_ab(i,j)
            theta = uh/er
            x_ss(i,j) = er * sin(theta)
            y_ss(i,j) = er * cos(theta)
            z_ss(i,j) = zet + z_axe_la*di_da*d_2(Nprofile)
            uh  = - y_ds_ab(i,j)
            zet = - x_ds_ab(i,j)
            theta = uh/er
            x_ds(i,j) = er * sin(theta)
            y_ds(i,j) = er * cos(theta)
            z_ds(i,j) = zet + z_axe_la*di_da*d_2(Nprofile)  
            uh  = - y_sl_ab(i,j)
            zet = - x_sl_ab(i,j)
            theta = uh/er
            x_sl(i,j) = er * sin(theta)
            y_sl(i,j) = er * cos(theta)
            z_sl(i,j) = zet + z_axe_la*di_da*d_2(Nprofile)           

         end do
      end do
    
      RETURN
      END
 



C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I    Berechnung der abgewickelten Profile mit den Ein- und Austritts-  I
C    I    winkel des Laufrades.                                             I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           A. Kaps  IHS,   9. 98

      SUBROUTINE RECHNE_SCHAUFELKONTUR

      INCLUDE 'common.f'

      REAL xtemp

      PI = 3.14159

      theta_0 = 360./real(nlschaufel)
      thetab_0 = 2*PI/ real(nlschaufel)
      ord3=0.4
      
      IF (max2.ne.0) then 
               
	
	     do j=1,Nprofile
            IF ((d_2(j)-d_2(1)).lt.(max2*(d_2(Nprofile)-d_2(1)))) then
               xtemp=(max2-((j-1.0)/(Nprofile-1)))/(max2)
               d_theta2(j)=pe2i*theta_0*umschlingung
     .      			       *(ord3*xtemp**3.0+(1-ord3)*xtemp**2.0)       
            ELSE
               xtemp=(((j-1.0)/(Nprofile-1))-max2)/(1.0-max2)
               d_theta2(j)=pe2a*theta_0*umschlingung
     .      			       *(ord3*xtemp**3.0+(1-ord3)*xtemp**2.0)
            END IF	    
	     end do
	    
      ELSE
	
	 do j=1,Nprofile             
	        xtemp=thetab_0*umschlingung*d_2(1)*pe2i
     .          +(thetab_0*umschlingung*d_2(Nprofile)*pe2a
     .            -thetab_0*umschlingung*d_2(1)*pe2i)
     .          *(j-1)/(Nprofile-1)
            d_theta2(j)=xtemp/d_2(j)  /PI*180

	                           
  	 end do

      END IF

      
      
      IF ((max1.ne.0).and.(max1.ne.1)) then 

         do j=1,Nprofile
            IF ((d_2(j)-d_2(1)).lt.(max1*(d_2(Nprofile)-d_2(1)))) then
               xtemp=(max1-((j-1.0)/(Nprofile-1)))/(max1)
               d_theta1(j)=pe1i*theta_0*umschlingung
     .      			       *(ord3*xtemp**3.0+(1-ord3)*xtemp**2.0)
            ELSE
               xtemp=(((j-1.0)/(Nprofile-1))-max1)/(1.0-max1)
               d_theta1(j)=pe1a*theta_0*umschlingung
     .      			       *(ord3*xtemp**3.0+(1-ord3)*xtemp**2.0)
            END IF
	 end do
	 
      ELSE
	
	 do j=1,Nprofile
	        xtemp=thetab_0*umschlingung*d_2(1)*pe1i
     .          +(thetab_0*umschlingung*d_2(Nprofile)*pe1a
     .            -thetab_0*umschlingung*d_2(1)*pe1i)
     .          *(j-1)/(Nprofile-1)
            d_theta1(j)=xtemp/d_2(j)  /PI*180
  	 end do

      END IF


      RETURN
      END





C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I    Berechnung der abgewickelten Profile mit den Ein- und Austritts-  I
C    I    winkel des Laufrades.                                             I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           A. Kaps  IHS,   9. 98

      SUBROUTINE RECHNE_SCHAUFELDICKEN

      INCLUDE 'common.f'

      REAL rtemp, delta_r, faktor_a, faktor_b

      PI = 3.14159

      theta_0 = 360./real(nlschaufel)
      thetab_0 = 2.*PI/ real(nlschaufel)
      ord3=0.2

      delta_r=(d_2(Nprofile)-d_2(1))/2.
      faktor_b=(dicke_i-dicke_a)/delta_r * d_strich_a
      faktor_a=(dicke_i-faktor_b*delta_r-dicke_a)/delta_r**2.0


      	    
	 do j=1,Nprofile            
               rtemp=delta_r*(Nprofile-j)/(Nprofile-1)               
               dzul_r(j)=faktor_a*rtemp**2.0+faktor_b*rtemp+dicke_a        
	 end do
	    



      RETURN
      END



C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I    Berechnung der abgewickelten Profile mit den Ein- und Austritts-  I
C    I    winkel des Laufrades.                                             I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           A. Kaps  IHS,   9. 98

      SUBROUTINE RECHNE_SCHAUFELPROFILVERSCHIEBUNG

      INCLUDE 'common.f'

      REAL versch

      PI = 3.14159

    	    
	 do j=1,Nprofile       


               versch=versch_i+(versch_a-versch_i)
     #                *(dicke_i-dzul_r(j))/(dicke_i-dicke_a)
               print*,versch
     
               	 do i=1,ZDP       
       
	            DL_prof(j,i)=DL_basprof(i)
                    XL_prof(j,i)=((XL_basprof(i)/100.0)**versch)*100.0
 	
	         end do
	 end do
	    



      RETURN
      END




C    +----------------------------------------------------------------------+
C    I                                                                      I
C    I    Berechnung der abgewickelten Profile mit den Ein- und Austritts-  I
C    I    winkel des Laufrades.                                             I
C    I                                                                      I
C    +----------------------------------------------------------------------+
C                                           E. Goede  IHS,  21. 9. 96

      SUBROUTINE RECHNE_ABWICKLUNG

      INCLUDE 'common.f'
      
      COMMON /maxw/  mwi,mwa
      REAL           mwi,mwa

  
      PI = 3.14159

      
C      faktor  = 15./18.
      faktor  = 1.
      theta_0 = 2./real(nlschaufel)*PI
      r_0    = d_2(1)/2.
      u_nabe = r_0 * theta_0/0.5
      do j=1,Nprofile
         beta1 = 180. - beta1_sch(j)  
         beta2 = 180. - beta2_sch(j)
         theta_sch(j) = theta_0 * umschlingung
     .                  - (d_theta2(j)+d_theta1(j))*PI/180.
         faktor_l =  theta_sch(j)/(theta_0*umschlingung)*d_2(j)/d_2(1)

         theta_s=theta_sch(j)
         r_j = d_2(j)/2.

            faktor_d = dzul_r(j) 
     .                  *d_2(1)*theta_0*umschlingung/2/10000.


         do i=1,ZDP   
            XL(I) = XL_prof(j,i)
            DL(I) = DL_prof(j,i) * faktor_d 
         end do
 
C        --- Definition der maximalen Kruemmung (linear) ---
         XWRL=mwi+(mwa-mwi)*(j-1)/(Nprofile-1)



C        !!! Achtung: ab hier gilt: x- und y-Achse vertauscht !!!

C        --- Aufruf der Unterroutine ---


         call PROFILE (faktor_l)

         do i=1,ZDP
            x_sl_ab(i,j) = SLX(I)
            y_sl_ab(i,j) = SLY(I)
            x_ds_ab(i,j) = DSX(I)
            y_ds_ab(i,j) = DSY(I)
            x_ss_ab(i,j) = SSX(I)
            y_ss_ab(i,j) = SSY(I)
         end do
         x_sl_dpkt(j) = SLX_dpkt
         y_sl_dpkt(j) = SLY_dpkt
      end do





c
C     --- Profile werden in die Drehachse gelegt ---
C
      do j=1,Nprofile

         r_j = d_2(j)/2.
         u_j = r_j * theta_0 * umschlingung
         diffy_j = abs (y_sl_ab(1,j) - y_sl_ab(ZDP,j))         
         faktor_l = theta_sch(j)/theta_0 * d_2(j)/d_2(1)
        
         diffx = x_sl_dpkt(j)
         diffy = d_2(j)*d_theta1(j)*PI/180/2
                           
         do i=1,ZDP
            x_sl_ab(i,j) = x_sl_ab(i,j) - diffx
            x_ds_ab(i,j) = x_ds_ab(i,j) - diffx
            x_ss_ab(i,j) = x_ss_ab(i,j) - diffx
         end do
         x_sl_dpkt(j) = x_sl_dpkt(j) - diffx
   
      end do



C
C     ------------- Skalierung auf Maschinengroesse -------------
C
      do j=1,Nprofile

         r_j = d_2(j)/2.
         u_j = r_j * theta_0 * umschlingung
         diffy_j = abs (y_sl_ab(1,j) - y_sl_ab(ZDP,j))  
         faktor_lj = r_j / diffy_j * theta_sch(j)
         diffy = d_2(j)*d_theta1(j)*PI/180/2
         
         do i=1,ZDP
            
            x_sl_ab(i,j) = x_sl_ab(i,j) * faktor_lj
            y_sl_ab(i,j) = y_sl_ab(i,j) * faktor_lj
            x_ds_ab(i,j) = x_ds_ab(i,j) * faktor_lj
            y_ds_ab(i,j) = y_ds_ab(i,j) * faktor_lj
            x_ss_ab(i,j) = x_ss_ab(i,j) * faktor_lj
            y_ss_ab(i,j) = y_ss_ab(i,j) * faktor_lj
            
            y_sl_ab(i,j) = y_sl_ab(i,j) - diffy
            y_ds_ab(i,j) = y_ds_ab(i,j) - diffy
            y_ss_ab(i,j) = y_ss_ab(i,j) - diffy

         end do
         x_sl_dpkt(j) = x_sl_dpkt(j) * faktor_lj
         y_sl_dpkt(j) = y_sl_dpkt(j) * faktor_lj 
         y_sl_dpkt(j) = y_sl_dpkt(j) - diffy    
      end do

      RETURN
      END





C
C***************************************************************
C                                                              *
C  Die Subroutine Profil berechnet die Profilform in einem     *
C  Schaufelschnitt bei gegebenen Ein- und Austrittswinkeln.    *
C  auf Basis Programm Batrekhy                                 *
C                                                              *
C***************************************************************
C                                          EGoe   21. 9. 1996
      SUBROUTINE profile (faktor_l)

      INCLUDE 'common.f'
      DATA PHITEMP/3.141593/,EPS/0.001/
C*****Bestimmung des staffelungswinkel ******

      PI = 3.14159

    
      BETA2N=BETA2
50    CONTINUE


C     --- Loesen der impliziten Gleichung fur GAMMA ---
      GAMMAA=0.0
      DO 10 I=1,100,1
	ARG1=(90.0-BETA1+GAMMAA)*PHITEMP/180.0
	ARG2=ABS((4.0*XWRL-1.0)/(3.0-4.0*XWRL)*TAN(ARG1))
	GAMMAN=BETA2N-90.0-ATAN(ARG2)*180.0/PHITEMP
	DDELTA=ABS((GAMMAN-GAMMAA)/GAMMAN)
	IF(DDELTA.LT.EPS) GOTO 100
	GAMMAA=GAMMAN
10    CONTINUE



C     --- Startwerte fuer Fehleriteration ---
      fehler=1
C      WRITE(*,*) 'Keine Konvergenz (Schleife 10)'
C      WRITE(*,*) 'Geben Sie einen Startwert fœr gamma ein'
C      READ(*,*)  GAMMAN
      GAMMAN=1


C     --- Loesen der impliziten Gleichung fuer GAMMA bei Fehler ---
      DO 11 I=1,200,1
	ARG1=(90.0-BETA1+GAMMAN)*PHITEMP/180.0
        ARG2=ABS((4.0*XWRL-1.0)/(3.0-4.0*XWRL)*TAN(ARG1))
        BETA2I=GAMMAN+90.0-ATAN(ARG2)*180.0/PHITEMP
        DELTBE=BETA2N-BETA2I
        IF(DELTBE.LT.-0.1) GAMMA=GAMMAN-0.1
	IF(DELTBE.GT.0.1) GAMMA=GAMMAN+0.1
        IF (ABS(DELTBE).LT.0.1) GOTO 100
	GAMMAN=GAMMA
11    CONTINUE




100   CONTINUE

C     --- Bestimmung von PHI ---
      DELTA1=90.0-BETA1+GAMMAN
      DELTA2=BETA2N-GAMMAN-90.0
      DELT1B=DELTA1*PHITEMP/180.0
      DELT2B=DELTA2*PHITEMP/180.0
      CC=(TAN(DELT1B)-TAN(DELT2B))/2.0/TAN(DELT1B)/TAN(DELT2B)
      GROPHI=ATAN(CC)
      GROPHI1=GROPHI*180.0/PHITEMP
      A1=TAN(DELT1B-GROPHI)
      A2=-1.0*(A1+TAN(GROPHI))/COS(GROPHI)
      cosphi = COS(GROPHI)
      sinphi = SIN(GROPHI)


C     --- Bestimmung der Bogenlaengen ---
      ETA_ALT=0
      XI_ALT=0
      GESLAENGE=0
      P_XI(1)=0
      P_ETA(1)=0
      BGLAENGE(1)=0
      do i=2,ZDP
         x_soll = XL(i)/100.
         x = x_soll
         P_XI(i) = x*cosphi
         P_ETA(i)=A1*P_XI(i)+A2*P_XI(i)**2.0
         BGSEHNE(i-1)=SQRT((P_XI(i)-P_XI(i-1))**2. 
     .                 + (P_ETA(i)-P_ETA(i-1))**2.)
         GESLAENGE=GESLAENGE+BGSEHNE(i-1)
         BGLAENGE(i)=GESLAENGE         
      end do

      do i=1,ZDP         
         BGLAENGE(i)=BGLAENGE(i)/GESLAENGE         
      end do




C --- Feststellen, wie stark das Profil spaeter ---
C     skaliert wird und Profildicke mit der 
C     invertierten skalieren 

	ARG3=GAMMAN/180.0*PHITEMP-GROPHI
	SLY(1)=P_ETA(1)*COS(ARG3)-P_XI(1)*SIN(ARG3)
	SLY(ZDP)=P_ETA(ZDP)*COS(ARG3)-P_XI(ZDP)*SIN(ARG3)
        sly_0 = sly(1)
        sly_1 = sly(ZDP)
        slaenge_1 = abs(sly_1-sly_0) 
        faktor_lj = r_j * theta_s
    

      do I=1,ZDP         
         DL(I)=DL(I)*slaenge_1/faktor_lj*10   
      end do
      











C     --- Bestimmung des x-Werts ---

      DO 20 I=1,ZDP,1

        j=1
  70    continue     

C       --- Iteration bis Bogenlaenge ---
        x_soll = XL(I)/100.
        if (x_soll.gt.BGLAENGE(j)) then
           j = j+1
           goto 70
        end if
        
        
        UEBER= (BGLAENGE(j)-x_soll) / (BGLAENGE(j)-BGLAENGE(j-1))      

        XI  = P_XI(j)  - (P_XI(j)-P_XI(j-1))   * UEBER 
        ETA = P_ETA(j) - (P_ETA(j)-P_ETA(j-1)) * UEBER 

        

	ALPHA=ATAN(A1+2.0*A2*XI)
	ARG3=GAMMAN/180.0*PHITEMP-GROPHI

C       --- Skelletlinie in x,y-Koordinaten ---
	SLY(I)=ETA*COS(ARG3)-XI*SIN(ARG3)
	SLX(I)=ETA*SIN(ARG3)+XI*COS(ARG3)


C       --- Schaufelsaugseite in eta,xi-Koordinaten ---
	ASSX=XI-SIN(ALPHA)*DL(I)/2.0
	ASSY=ETA+COS(ALPHA)*DL(I)/2.0

C       --- Schaufeldruckseite in eta,xi-Koordinaten ---
	ADSX=XI+SIN(ALPHA)*DL(I)/2.0
	ADSY=ETA-COS(ALPHA)*DL(I)/2.0

C       --- Schaufelsaugseite in x,y-Koordinaten ---
	SSY(I)=ASSY*COS(ARG3)-ASSX*SIN(ARG3)
	SSX(I)=ASSY*SIN(ARG3)+ASSX*COS(ARG3)

C       --- Schaufeldruckseite in x,y-Koordinaten ---
	DSY(I)=ADSY*COS(ARG3)-ADSX*SIN(ARG3)
	DSX(I)=ADSY*SIN(ARG3)+ADSX*COS(ARG3)

20    CONTINUE
      


C
C     --- Bestimmung des Drehpunkts auf Skelettlinie bei sdpkt/sl * 100% *******
C     Na-ja!!!

        x_soll = sdpktzul
        x = x_soll
        j = 1
  66    continue
        xi = x*cosphi
        ETA=A1*XI+A2*XI**2.0
        x_par = xi*cosphi - eta*sinphi
        diffx = x_soll - x_par
        if (abs(diffx).gt.0.001 .and. j.lt.1000) then
           x = x + diffx
           j = j + 1
           goto 66
        end if
      SLY_dpkt=ETA*COS(ARG3)-XI*SIN(ARG3)
      SLX_dpkt=ETA*SIN(ARG3)+XI*COS(ARG3)


C
C     *****  Normierung auf Profillaenge = 100%*(faktor_l) *******
C
      slx_0 = 0.
      sly_0 = 0.
      slx_1 = slx(ZDP)
      sly_1 = sly(ZDP)
C      slaenge = abs(sly_1-sly_0)/faktor_l
      slaenge = abs(sly_1-sly_0)

      do i=1,ZDP
   	 SLY(I)=SLY(I)/slaenge*100.
	 SLX(I)=SLX(I)/slaenge*100.
	 SSY(I)=SSY(I)/slaenge*100.
	 SSX(I)=SSX(I)/slaenge*100.
	 DSY(I)=DSY(I)/slaenge*100.
	 DSX(I)=DSX(I)/slaenge*100.
      end do
      SLY_dpkt=SLY_dpkt/slaenge*100.
      SLX_dpkt=SLX_dpkt/slaenge*100.
C
C*****Bestimmung des tatsÃchlichen Austrittswinkels ***
C
      T=DS*PHITEMP/Z*1000.0
      AMIN=T
      DO 40 L=1,ZDP,1
	A=SQRT((SLX(ZDP)-SSX(L))**2.0+(SLY(ZDP)+T-SSY(L))**2.0)
	IF (A.LT.AMIN) AMIN=A
40    CONTINUE
      BET2SR=ASIN(AMIN/T)*180.0/PHITEMP
      BET2R=180.0-BET2SR

53    CONTINUE

      ANTW = 2
      IF (ANTW.EQ.1) GOTO 51
      IF (ANTW.EQ.2) GOTO 52
      GOTO 53
51    CONTINUE
      WRITE(*,*) 'Geben Sie das neue beta2 ein!'
      READ(*,*) BETA2N
      GOTO 50
52    CONTINUE




  

      RETURN
      END
