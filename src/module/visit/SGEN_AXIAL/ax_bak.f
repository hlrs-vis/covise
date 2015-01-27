c ----------------------------------------------------------------------
c AXNET VERSION 1.0
c GENERIERUNG VON BERECHNUNGSGITTERN FUER AXIALE PROPELLERTURBINEN
c INSTITUT FUER HYDRAULISCHE STROEMUNGSMECHANIK
c M.BURGER - 29.10.98
c ----------------------------------------------------------------------
c
c Eingabestruktur: - Geometriepunkte der Schaufelgeometrie als
c                    DAT-File nach Institutsnorm
c                  Anmerkung:
c                  Die Vernetzungsroutine erwartet lediglich k Geometrie
c                  punkte, die hintereinander versplinebar sind, doppel
c                  te Punkte werden automatisch eliminiert. 
c                  Aufgrund der Berechnung des Ein- bzw. Austrittswinkel
c                  und der neutralen Faser sollten die Punkte symmetri
c                  sch (n Punkte auf beiden Seiten) angeordnet sein.
c
c Ausgabestruktur: - GEO-File, Institutsnorm
c                  - RB-File, Institutsnorm
c                  - Zylinderschnitte als 2D-Geofile (zum Einstellen)
c     
c Parameter 
c     
c a_gebi..... Anzahl der Gebiete
c a_deck..... Anzeahl Dreiecksgebiete
c n.......... Anzahl Geometriepunkte pro Seite eines Naca-Profiles
c b.......... Anzahl der geg. Radialschnitte
c k.......... Anzahl Geometriepunkte pro Naca-Profil
c DIM_F...... Grosse des Datenfeldes f (real)
c DIM_E...... Grosse des Datenfeldes e (integer)
c DIM_BI..... Dimensionierung der Datenfelder xi,yi,zi    
c a_flug..... Anzahl der Fluegel 
c stf_fi..... Steuerfile
c ----------------------------------------------------------------------
c PROGRAMM AXNET
c ----------------------------------------------------------------------
      SUBROUTINE axnet()
c ----------------------------------------------------------------------
c INTEGER VARIABLEN DEKLARIEREN
c ----------------------------------------------------------------------
      implicit none
c
      integer n,b,k,m,DIM_F,DIM_E,DIM_BI,a_flug,bi,PG(5),a_gebi,a_deck,
     #        AUSG,dat1,dat2,dat3,dat4,dat5,dat6,dat7,dat8,dat9,dat10,
     #        ase1,ase2,b_anf,b_end,
     #        ixp,iyp,ixn,iyn,ixr,iyr,ixk,iyk,izk,
     #        ixb1,iyb1,ixb2,iyb2,ixb3,iyb3,ixb4,iyb4,ixpel2,iypel2,
     #        ixper2,iyper2,ixpel1,iypel1,ixper1,iyper1,ixpel3,iypel3,
     #        ixper3,iyper3,
     #        ikrbx,ikrby,ikrbz,ikrbk,ikrbe,
     #        ixkn,iykn,izkn,
     #        izli,iela,ielb,ielc,ield,iel1,iel2,iel3,iel4,iel5,iel6,
     #        iel7,iel8,
     #        igrkli,ireli,irkli,inakli,irokli,irukli,irlkli,irrkli,
     #        ikrbn,iwrba,iwrbb,iwrbe,ikbia,ikbib,ikbie,ikbiw,ikman,
     #        ikmaw,ikrb1,iwrb1,iwrb2,iwrb3,iwrb4,iwrb5,ikbi1,ikbi2,
     #        ikbi3,ikbi4,ikbi5,ikbi6,ikma1,ikma2,ende,
     #        schnitt,seed,bord,bord2,d_seed,d_bord,d_bord2,
     #        za,pob,por,an_kno,a_2Del,a_3Del,an_grk,anz_re,
     #        anz_rk,an_nak,an_rok,an_ruk,an_rlk,an_rrk,an_krb,an_wrb,
     #        an_kbi,an_kma,an3_kr,an3_wr,an3_kb,an3_km,an3_kw,
     #        wen_kb,weo_kb,weu_kb,wel_kb,wer_kb,wel_km,wer_km,weo_km,
     #        DIM_PVR,apvr

      integer lese_xyz_dat,netz_speichern     
      common /akno/ an_kno, bi, ixkn, iykn, izkn
      common /ainf/ lese_xyz_dat,netz_speichern
      common /aplo/ ixr,iyr,seed
c ----------------------------------------------------------------------
c GRUNDEINSTELLUNG FUER VERNETZUNGSROUTINE, NICHT AENDERN ! 
c ----------------------------------------------------------------------
      parameter (a_gebi=14,a_deck=0)
c ----------------------------------------------------------------------
c DIMENSION DES DATENFELDES, BEI FEHLERMELDUNG VERGROESSERN 
c ----------------------------------------------------------------------
      parameter (DIM_F=1500000,DIM_E=4000000,DIM_BI=100,DIM_PVR=100)
c ----------------------------------------------------------------------
c DATENFORMAT DES DAT_FILE 
c ----------------------------------------------------------------------
      parameter (n=19,b=11,k=2*n)
c ----------------------------------------------------------------------
c INTEGER VARIABLEN DEKLARIEREN
c ----------------------------------------------------------------------
      integer ase(a_gebi,2),e(DIM_E)
c ----------------------------------------------------------------------
c REAL VARIABLEN DEKLARIEREN
c ----------------------------------------------------------------------
      real f(DIM_F),PI,radius,xu,yu,xo,yo,delta,bgs,alpha,beta,val,valN,
     #     valK,vbe,ael,aer,v_ein,l_mod,norm_fr,norm_fp,norm_fz,KEA,
     #     hilf1,hilf2,hilf3,hilf4,
     #     xspu,yspu,xmp1,ymp1,xmp2,ymp2,xmp3,ymp3,vsdeck,vsfdec,vs1dec,
     #     vs2dec,gevs1N,gevs2N,gevs1K,gevs2K,
     #     x(k,b),y(k,b),z(k,b),zu_nac,zo_nac,r_nab,r_kra,SPL12,SPL12N,
     #     SPL12K,xi(k,DIM_BI),yi(k,DIM_BI),zi(k,DIM_BI),
     #     lvs1,lvs2,lvs3,lvs4,lvs5,lvs6,lvs7,bvs1,bvs2,bvs3,bvs4,bvs5,
     #     rvs,alp_hi,
     #     rpvrN(DIM_PVR),rpvrK(DIM_PVR),zpvr(DIM_PVR),
     #     h_feldx(50),h_feldy(50),
     #     xdr1,ydr1,xdr2,ydr2,xdr3,ydr3,xdr4,ydr4,xdr5,ydr5,xdr6,ydr6,
     #     xdr7,ydr7,xl25(50),yl25(50),xl51(50),yl51(50),
     #     xl26(50),yl26(50),xl63(50),yl63(50)


      common /anet/ f(DIM_F),e(DIM_E)
      common /awrb/ an3_wr,iwrb1,iwrb2,iwrb3,iwrb4
      common /akbi/ an3_kb,ikbi1,ikbi2,ikbi3,ikbi4
      common /axel/ a_3Del,iel1,iel2,iel3,iel4,iel5,iel6,iel7,iel8

c
      parameter(PI=3.141592654)
c ----------------------------------------------------------------------
c LOGICAL VARIABLEN DEKLARIEREN
c ----------------------------------------------------------------------
      logical CRE_RB,MOVR_TF
c ----------------------------------------------------------------------
c CHARACTER VARIABLEN DEKLARIEREN
c ----------------------------------------------------------------------
      character*200 in_fi,stf_fi,mov_fi

c ----------------------------------------------------------------------
c PROGRAMM START
c ----------------------------------------------------------------------
      print*, 'AXNET V1.0 - IHS - BURGER - 29.10.1998'
c ----------------------------------------------------------------------
c EINLESEN DES STEUERFILES
c ----------------------------------------------------------------------
      print*, 'Lese Steuerfile ein:'
      call INP_STF(in_fi,MOVR_TF,mov_fi,bi,AUSG,a_flug,yu,yo,bgs,
     # valN,valK,vbe,ael,aer,SPL12N,SPL12K,m,PG,dat1,dat2,dat3,dat4,
     # dat5,dat6,dat7,dat8,dat9,dat10,lvs1,lvs2,lvs3,lvs4,lvs5,lvs6,
     # lvs7,bvs1,bvs2,bvs3,bvs4,bvs5,rvs,vsfdec,gevs1N,gevs2N,
     # gevs1K,gevs2K,CRE_RB,v_ein,l_mod,wen_kb,weo_kb,weu_kb,
     # wel_kb,wer_kb,wel_km,wer_km,weo_km)
c ----------------------------------------------------------------------
c UEBERPRUEFE EINGABEN DURCH DAS STEUERFILE
c ----------------------------------------------------------------------
      if (AUSG.gt.bi) then
       print*, 'STOP: Ausgabe Rad.schnitte groesser Anzahl Rad.schnitte'
       stop
      elseif ((m/2.).eq.aint(m/2.)) then
       print*, 'STOP: Anzahl Stuetzpnukte sind nicht ungerade'
       stop
      elseif (PG(1).ne.0) then
       print*, 'STOP: Stuetzpunkt 1 ist ungleich 0'
       stop
      elseif (PG(5).ne.(m-1)) then
       print*, 'STOP: Stuetzpunkt 5 ist ungleich ',(m-1)
       stop
      elseif (PG(2).lt.PG(1)) then
       print*, 'STOP: Stuetzpunkt 2 ist kleiner Stuetzpunkt 1'
       stop
      elseif (PG(2).gt.PG(3)) then
       print*, 'STOP: Stuetzpunkt 2 ist groesser Stuetzpunkt 3'
       stop 
      elseif (PG(3).lt.PG(2)) then
       print*, 'STOP: Stuetzpunkt 3 ist kleiner Stuetzpunkt 2'
       stop
      elseif (PG(3).gt.PG(4)) then
       print*, 'STOP: Stuetzpunkt 3 ist groesser Stuetzpunkt 4'
       stop 
      elseif (PG(4).lt.PG(3)) then
       print*, 'STOP: Stuetzpunkt 4 ist kleiner Stuetzpunkt 3'
       stop
      elseif (PG(4).gt.PG(5)) then
       print*, 'STOP: Stuetzpunkt 4 ist groesser Stuetzpunkt 5'
       stop 
      endif
c ----------------------------------------------------------------------
c UEBERPRUEFE ob DIM_BI gross genug    
c ----------------------------------------------------------------------
      if (bi.gt.DIM_BI) then
       print*, 'STOP: DIM_BI kleiner ',bi
       stop
      endif
c ----------------------------------------------------------------------
c ERZEUGE BEDINGUNG: MESHSEEDS DER DREIECKE MUSS UNGERADE SEIN    
c ----------------------------------------------------------------------
      if ((dat5/2.).eq.(aint(dat5/2.))) then
         dat5=dat5+1
         print*, 'Anzahl Knoten Gebiet 5 auf ',dat5,' gesetzt'
      endif

      dat10=int((dat9+1)/2)
  
c ----------------------------------------------------------------------
c ANZAHL DER MESHSEEDS FESTSETZEN
c ----------------------------------------------------------------------
      ase(1,1)=dat1
      ase(1,2)=dat2
c
      ase(2,1)=dat3
      ase(2,2)=dat2
c      
      ase(3,1)=dat4
      ase(3,2)=dat2
c      
      ase(4,1)=dat1+(dat9-1)
      ase(4,2)=dat2
c      
      ase(5,1)=dat4
      ase(5,2)=dat6
c      
      ase(6,1)=dat3
      ase(6,2)=dat6
c      
      ase(7,1)=dat7
      ase(7,2)=dat6
c      
      ase(8,1)=dat1
      ase(8,2)=dat7
c      
      ase(9,1)=dat9
      ase(9,2)=dat7
c      
      ase(10,1)=dat9
      ase(10,2)=dat2
c      
      ase(11,1)=dat9
      ase(11,2)=dat2
c      
      ase(12,1)=dat10
      ase(12,2)=dat10
c
      ase(13,1)=dat9-dat10+1
      ase(13,2)=dat9-dat10+1
c
      ase(14,1)=dat10
      ase(14,2)=dat10
c

c ----------------------------------------------------------------------
c DIMENSIONIERUNG DER DATENFELDER
c ----------------------------------------------------------------------
      d_seed=0
      d_bord=0
      d_bord2=0
      do za=1,a_gebi,1
       d_seed=ase(za,1)*ase(za,2)+d_seed
       d_bord=ase(za,1)+d_bord
c Dreiecksgebiete werden mit NETZX vernetzt       
c       if (za.gt.(a_gebi-(3*a_deck))) d_bord2=ase(za,2)+d_bord2 
      end do
c Sonstige Gebiete die mit NETZX vernetzt werden beruecksichtigen
c      d_bord2=d_bord2+ase(6,2)+ase(9,2)+ase(11,2)+ase(12,2)+
c     # ase(13,2)+ase(14,2) 
c ----------------------------------------------------------------------
c DATENFELD f (real)
c ----------------------------------------------------------------------
c Stuetzpunkte NACA  
      ixp=1
      iyp=ixp+m
c Stuetzpunkte neutrale Faser 
      ixn=iyp+m
      iyn=ixn+int((m+1)/2)
c Punkte zylindrisch 
      ixr=iyn+int((m+1)/2)
      iyr=ixr+d_seed
c Punkte kartesisch
      ixk=iyr+d_seed
      iyk=ixk+d_seed
      izk=iyk+d_seed
c Raender (Hilfspunkte) 
      ixb1=izk+d_seed
      iyb1=ixb1+d_bord
      ixb2=iyb1+d_bord
      iyb2=ixb2+d_bord
      ixb3=iyb2+d_bord
      iyb3=ixb3+d_bord2
      ixb4=iyb3+d_bord2
      iyb4=ixb4+d_bord2
c Periodische Raender unten (Hilfspunkte)
      ixpel1=iyb4+d_bord2
      iypel1=ixpel1+ase(11,1)
      ixper1=iypel1+ase(11,1)
      iyper1=ixper1+ase(11,1)
c Periodische Raender (Hilfspunkte)
      ixpel2=iyper1+ase(11,1)
      iypel2=ixpel2+ase(3,2)
      ixper2=iypel2+ase(3,2)
      iyper2=ixper2+ase(3,2)
c Periodische Raender (Hilfspunkte)
      ixpel3=iyper2+ase(3,2)
      iypel3=ixpel3+ase(5,2)
      ixper3=iypel3+ase(5,2)
      iyper3=ixper3+ase(5,2)
c Knoten fuer Berechnungsgitter
      ixkn=iyper3+ase(5,2)
      iykn=ixkn+d_seed*bi
      izkn=iykn+d_seed*bi
c Knotenrandbedingungen 3D (Knotenwerte in 3D-Koordinaten)
      ikrbx=izkn+d_seed*bi
      ikrby=ikrbx+d_seed*bi
      ikrbz=ikrby+d_seed*bi
      ikrbk=ikrbz+d_seed*bi
      ikrbe=ikrbk+d_seed*bi
c      
      ende=ikrbe+d_seed*bi
c ----------------------------------------------------------------------
c Pruefe ob Dimensionierung ausreichend
c ----------------------------------------------------------------------
      print*, 'BENOETIGTE FELDGROESSE F: ',ende
      print*, 'DIMENSIONIERTE FELDGROESSE F: ',DIM_F
      if (ende.gt.DIM_F) then
         print*, 'STOP: DATENFELD F ZU KLEIN DIMENSIONIERT'
         stop
      endif
c ----------------------------------------------------------------------
c DATENFELD e (integer)
c ----------------------------------------------------------------------
c Zuordnugsliste Punkte - Knoten
      izli=1
c Elemente 2D      
      iela=izli+d_seed
      ielb=iela+d_seed
      ielc=ielb+d_seed
      ield=ielc+d_seed
c Elemente 3D
      iel1=ield+d_seed*(bi-1)
      iel2=iel1+d_seed*(bi-1)
      iel3=iel2+d_seed*(bi-1)
      iel4=iel3+d_seed*(bi-1)
      iel5=iel4+d_seed*(bi-1)
      iel6=iel5+d_seed*(bi-1)
      iel7=iel6+d_seed*(bi-1)
      iel8=iel7+d_seed*(bi-1)
c Gebietsrandknotenliste 2D
      igrkli=iel8+d_seed*(bi-1)
c Randelemente 2D    
c alle
      ireli=igrkli+d_seed
c Randknoten 2D
c alle
      irkli=ireli+d_seed
c Naca-Profil
      inakli=irkli+d_seed
c Rand oben 
      irokli=inakli+d_seed
c Rand unten
      irukli=irokli+d_seed
c Rand links 
      irlkli=irukli+d_seed
c Rand rechts 
      irrkli=irlkli+d_seed
c KNOTENRANDBEDINGUNGEN 2D
c Knotennummer
      ikrbn=irrkli+d_seed
c WANDRANDBEDINGUNGEN 2D
c Knotennummer
      iwrba=ikrbn+d_seed
      iwrbb=iwrba+d_seed
c Elementnummer   
      iwrbe=iwrbb+d_seed
c KRAFT- u. BILANZFLAECHEN 2D
c Knotennummer
      ikbia=iwrbe+d_seed
      ikbib=ikbia+d_seed
c Elementnummer
      ikbie=ikbib+d_seed
c Wert (Gruppennummer)
      ikbiw=ikbie+d_seed
c KNOTENMARKIERUNGEN 2D (HILFSKNOTEN)
c Knotennummer
      ikman=ikbiw+d_seed
c Wert
      ikmaw=ikman+d_seed
c KNOTENRANDBEDINGUNGEN 3D
c Knotennummer
      ikrb1=ikmaw+d_seed
c WANDRANDBEDINGUNGEN 3D
c Knotennummer
      iwrb1=ikrb1+d_seed*bi
      iwrb2=iwrb1+d_seed*(bi-1)
      iwrb3=iwrb2+d_seed*(bi-1)
      iwrb4=iwrb3+d_seed*(bi-1)
c Elementnummer
      iwrb5=iwrb4+d_seed*(bi-1)
c KRAFT- u. BILANZFLAECHEN 3D
c Knotennummer
      ikbi1=iwrb5+d_seed*(bi-1)
      ikbi2=ikbi1+d_seed*(bi-1)
      ikbi3=ikbi2+d_seed*(bi-1)
      ikbi4=ikbi3+d_seed*(bi-1)
c Elementnummer
      ikbi5=ikbi4+d_seed*(bi-1)
c Wert (Gruppennummer)
      ikbi6=ikbi5+d_seed*(bi-1)
c KNOTENMARKIERUNGEN 3D (HILFSKNOTEN)
c Knotennummer
      ikma1=ikbi6+d_seed*(bi-1)
c Wert
      ikma2=ikma1+d_seed*bi
c      
      ende=ikma2+d_seed*bi
c ----------------------------------------------------------------------
c Pruefe ob Dimensionierung ausreichend
c ----------------------------------------------------------------------
      print*, 'BENOETIGTE FELDGROESSE E: ',ende
      print*, 'DIMENSIONIERTE FELDGROESSE E: ',DIM_E
      if (ende.gt.DIM_E) then
         print*, 'STOP: DATENFELD E ZU KLEIN DIMENSIONIERT'
         stop
      endif
c ----------------------------------------------------------------------
c PROGRAMM ANFANG
c ----------------------------------------------------------------------
c ----------------------------------------------------------------------
c Geometriedaten 
c ----------------------------------------------------------------------
c Einlesen
      print*, 'Einlesen von ',in_fi
      if (lese_xyz_dat.eq.1) call INP_DAT(in_fi,n,b,k,x,y,z)
      if (lese_xyz_dat.eq.0) call INP_ONLINE(x,y,z,a_flug)







c Radius an der Nabe und am Kranz ermitteln
      r_nab=(x(1,1)**2+y(1,1)**2)**0.5
      r_kra=(x(1,b)**2+y(1,b)**2)**0.5
      print*, 'Radius an der Nabe: ',r_nab
      print*, 'Radius am Kranz: ',r_kra
c Oberen und unteren Punkt des Schaufelprofils ermitteln
      call ZUZO_NACA(k,b,z,zu_nac,zo_nac)
      print*, 'Unterste z-Koordinate der Geometriedaten: ',zu_nac
      print*, 'Oberste z-Koordinate der Geometriedaten: ',zo_nac
c Pruefe obere und untere Grenze
      if (yu.gt.zu_nac) then
       print*, 'STOP: Untere Grenze schneidet mit Schaufelprofil: ',yu 
       stop
      elseif (yo.lt.zo_nac) then
       print*, 'STOP: Obere Grenze schneidet mit Schaufelprofil: ',yo
       stop
      endif
c Interpolieren     
      call IPOL_RS(k,b,x,y,z,bi,2,rvs,xi,yi,zi)
c ----------------------------------------------------------------------
c Daten fuer die Radialverschiebung einlesen
c ----------------------------------------------------------------------
      if (MOVR_TF) call INP_MOVR(mov_fi,DIM_PVR,rpvrN,rpvrK,zpvr,apvr)
c ----------------------------------------------------------------------
c ANFANG DER 2D-KNOTENBERECHNUNG FUER EIN RADIALSCHNITT (NACA-PROFIL)
c ----------------------------------------------------------------------
c Variablen Werte zuweisen
      an_kno=0
      e(izli)=12345
      an_krb=0
      an_wrb=0
      an_kbi=0
      an_kma=0
      an3_kw=0
c Abfrage ob Ausgabe 3D oder 2D     
      if (AUSG.eq.0) then 
        b_anf=1
        b_end=bi
      else
        b_anf=1
        b_end=AUSG
      endif
c      
      do schnitt=b_anf,b_end,1 
       print*,'PROFIL ',schnitt
c ----------------------------------------------------------------------
c Stuetzpunkte generieren
c ----------------------------------------------------------------------
c Faktor L12 zur Generierung der Stuetzpunkte berechnen
       SPL12=SPL12N+(SPL12K-SPL12N)/(bi-1)*(schnitt-1)
c Generierung der Stuetzpunkte des NACA-Profils 
       call CRE_NACA(k,bi,xi,yi,zi,schnitt,m,SPL12,radius,f(ixp),
     #  f(iyp),norm_fr,norm_fp,norm_fz)
       print*, 'Normierungsfaktoren (r,phi,z):',norm_fr,norm_fp,norm_fz
c ----------------------------------------------------------------------
c Berechnung von Verschiebungswinkel delta
c ----------------------------------------------------------------------
       delta=(2*PI/a_flug)*norm_fp
       print*, 'Verschiebung der periodischen Raender: delta = ',delta
c ----------------------------------------------------------------------
c Berechnung der neutralen Faser
c ----------------------------------------------------------------------
       call N_FASER(m,f(ixp),f(iyp),int((m+1)/2),f(ixn),f(iyn))
c ----------------------------------------------------------------------
c Berechne den Mittelpunkt und Winkel der oberen und unteren Grenze
c ----------------------------------------------------------------------
       call RAND_SPOTS(f(ixn),f(iyn),int((m+1)/2),yu,yo,valN,valK,vbe,
     #  alp_hi,schnitt,bi,xu,xo,alpha,val,beta,radius,r_nab,r_kra)

c ----------------------------------------------------------------------
c Setze Anfangspunkt und Endpunkt auf neutrale Faser
c ----------------------------------------------------------------------
       f(ixp)=f(ixn)
       f(iyp)=f(iyn)
       f(ixp+m-1)=f(ixn)
       f(iyp+m-1)=f(iyn)
c ----------------------------------------------------------------------
c Berechnung der Verschiebungen im Dreieck
c ----------------------------------------------------------------------
       vsdeck=(f(iyn)-yu)*vsfdec
       vs1dec=vsdeck*(gevs1N+(gevs1K-gevs1N)
     #              *(radius-r_nab)/(r_kra-r_nab))
       vs2dec=vsdeck*(gevs2N+(gevs2K-gevs2N)
     #              *(radius-r_nab)/(r_kra-r_nab))
c ----------------------------------------------------------------------
c Berechnung der Hilfspunkte fuer die periodischen Raender
c ----------------------------------------------------------------------

c Periodische Raender 1
       ase1=ase(11,1)
c       call GERADE(f(ixn),f(iyn),xu,yu,ase1,1,lvs5,
c     #  f(ixpel1),f(iypel1))

       do za=0,ase1-1,1
        f(ixper1+za)=f(ixpel1+za)+delta
        f(iyper1+za)=f(iypel1+za) 
       end do

c Periodische Raender 2       
       ase1=ase(3,2)
c       call MESHSEED(f(ixn),f(iyn),0,0.,0.,0,int((m+1)/2)-1,0.,0.,
c     #  0.,1,lvs6,int((m+1)/2)-1,ase1,f(ixpel2),f(iypel2))
c       call MESHSEED(f(ixn),f(iyn),0,0.,0.,0,int((m+1)/2)-1,delta,0.,
c     #  0.,1,lvs6,int((m+1)/2)-1,ase1,f(ixper2),f(iyper2))


c Periodische Raender 3
       ase1=ase(5,2)
c       call GERADE(f(ixn+int((m+1)/2)-1),f(iyn+int((m+1)/2)-1),xo,yo,
c     #  ase1,1,lvs7,f(ixpel3),f(iypel3))
       do za=0,ase1-1,1
        f(ixper3+za)=f(ixpel3+za)+delta
        f(iyper3+za)=f(iypel3+za) 
       end do


c ----------------------------------------------------------------------
c START: VERNETZUNGSROUTINE (BERECHNUNG DER PUNKTE)  
c ----------------------------------------------------------------------
       seed=0
       bord=0
       bord2=0
c ----------------------------------------------------------------------
c Grenzschicht 
c ----------------------------------------------------------------------
c ----------------------------------------------------------------------
c Gebiet 1
c ----------------------------------------------------------------------
       ase1=ase(1,1)
       ase2=ase(1,2)
c Rand 1   
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(1),PG(2),delta,0.,
     #  0.,1,1./lvs1,m-1,ase1,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(1),PG(2),delta,0.,
     #  aint(bgs),1,1./lvs1,m-1,ase1,f(ixb2+bord),f(iyb2+bord))
c Vernetzen      
       do za=bord,(ase1-1+bord),1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 

       bord=ase1+bord



 
c ----------------------------------------------------------------------
c Gebiet 2, Achtung: Rapu werden berechnet - G1 muss gleiche Rapu haben 
c ----------------------------------------------------------------------
       ase1=ase(2,1)
       ase2=ase(2,2)
c Rand 1       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(2),PG(3),delta,0.,
     #  0.,1,1./lvs2,m-1,ase1,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(2),PG(3),delta,0.,
     #  aint(bgs),1,1./lvs2,m-1,ase1,f(ixb2+bord),f(iyb2+bord))
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord

c ----------------------------------------------------------------------
c Gebiet 3, Achtung: Rapu werden berechnet - G2 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(3,1)
       ase2=ase(3,2)
c Rand 1       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(3),PG(4),0.,0.,0.,
     #  1,lvs3,m-1,ase1,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(3),PG(4),0,0.,aint(bgs),
     #  1,lvs3,m-1,ase1,f(ixb2+bord),f(iyb2+bord))
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord

c ----------------------------------------------------------------------
c Gebiet 4, Achtung: Rapu werden berechnet - G3 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(4,1)
       ase2=ase(4,2)
c Rand 1       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(4),PG(5),0.,0.,0.,
     #  1,lvs4,m-1,ase1,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       call MESHSEED(f(ixp),f(iyp),0,0.,0.,PG(4),PG(5),0.,0.,aint(bgs),
     #  1,lvs4,m-1,ase1,f(ixb2+bord),f(iyb2+bord))
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord

c ----------------------------------------------------------------------
c Huelle  
c ----------------------------------------------------------------------

c ----------------------------------------------------------------------
c Gebiet 5, Achtung: Rapu werden berechnet - G7 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(5,1)
       ase2=ase(5,2)
c Rand 2       
       pob=ase(1,1)+ase(2,1)
       hilf1=(yo-f(iyb2+pob))/tan(alpha)+f(ixb2+pob)
       pob=ase(1,1)+ase(2,1)+ase(3,1)-1
       hilf2=(yo-f(iyb2+pob))/tan(alpha-aer)+f(ixb2+pob)
       call GERADE(hilf1,yo,hilf2,yo,ase1,0,0.,f(ixb2+bord),
     #  f(iyb2+bord))
c Rand 1       
       pob=ase(1,1)+ase(2,1)
       do za=0,ase1-1,1
        f(ixb1+bord+za)=f(ixb2+pob+za)
        f(iyb1+bord+za)=f(iyb2+pob+za)
       end do   
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb1+za),f(iyb1+za),f(ixb2+za),f(iyb2+za),
     #   ase2,1,lvs7,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do
        bord=ase1+bord

c ----------------------------------------------------------------------
c Gebiet 6, Achtung: Rapu werden berechnet - G6 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(6,1)
       ase2=ase(6,2)
c Rand 2       
       pob=ase(1,1)
       hilf1=(yo-f(iyb2+pob))/tan(alpha+ael)+f(ixb2+pob)
       pob=ase(1,1)+ase(2,1)-1
       hilf2=(yo-f(iyb2+pob))/tan(alpha)+f(ixb2+pob)
       call GERADE(hilf1,yo,hilf2,yo,ase1,0,0.,f(ixb2+bord),
     #  f(iyb2+bord))
c Rand 1      
       pob=ase(1,1)
       do za=0,ase1-1,1
        f(ixb1+bord+za)=f(ixb2+pob+za)
        f(iyb1+bord+za)=f(iyb2+pob+za)
       end do   
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb1+za),f(iyb1+za),f(ixb2+za),f(iyb2+za),
     #   ase2,1,lvs7,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do
        bord=ase1+bord

c ----------------------------------------------------------------------
c Gebiet 7, Achtung: Rapu werden berechnet - G5 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(7,1)
       ase2=ase(7,2)
c Rand 1       
       pob=ase(1,1)
       hilf1=(yo-f(iyb2+pob))/tan(alpha+ael)+f(ixb2+pob)
       pob=ase(1,1)+ase(2,1)+ase(3,1)
       hilf2=(yo-f(iyb2+pob))/tan(alpha-aer)+f(ixb2+pob)



      
       call GERADE(hilf1,yo,hilf2,yo,ase1,2,bvs2,f(ixb1+bord),
     #  f(iyb1+bord))
c Rand 2       
       pob=ase(1,1)
       hilf1=f(ixb2+pob)
       hilf2=f(iyb2+pob)
       pob=ase(1,1)+ase(2,1)+ase(3,1)
       hilf3=f(ixb2+pob)
       hilf4=f(iyb2+pob)

       call GERADE(hilf1,hilf2,hilf3,hilf4,ase1,
     #  2,lvs7,f(ixb2+bord),f(iyb2+bord))
c Rand 3
       pob=bord
       do za=bord,ase1-1+bord,1
       call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #  ase2,1,lvs7,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do
       bord=ase1+bord

c Rand 4
c       do za=0,ase2-1,1
c        f(ixb4+bord2+za)=f(ixpel3+za)
c        f(iyb4+bord2+za)=f(iypel3+za)
c       end do
c Vernetzten      
c       call NETZ(f(ixb1+bord),f(iyb1+bord),f(ixb2+bord),f(iyb2+bord),
c     #  f(ixb3+bord2),f(iyb3+bord2),f(ixb4+bord2),f(iyb4+bord2),ase1,
c     #  ase2,f(ixr+seed),f(iyr+seed))   
c

 
c ----------------------------------------------------------------------
c Gebiet 8
c ----------------------------------------------------------------------
       ase1=ase(8,1)
       ase2=ase(8,2)
c Rand 2       
       pob=ase(1,1)+ase(2,1)+ase(3,1)
       do za=0,ase1-1,1
        f(ixb2+bord+za)=f(ixb2+pob+za)      
        f(iyb2+bord+za)=f(iyb2+pob+za)
       end do
c Rand 1       


       pob=ase(1,1)-1
       do za=0,ase1-1,1
        f(ixb1+bord+za)=f(ixb2+pob-za)
        f(iyb1+bord+za)=f(iyb2+pob-za)
       end do 


       
  
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb1+za),f(iyb1+za),f(ixb2+za),f(iyb2+za),
     #   ase2,2,lvs7,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord 
  
c ----------------------------------------------------------------------
c Gebiet 9, Achtung: Rapu werden berechnet - G8 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(9,1)
       ase2=ase(9,2)

c Rand 1       

       pob=0
       hilf1=((yu-f(iyp))/tan(beta)+f(ixp)+delta)-aint(bgs)/sin(beta)
  
       hilf2=f(ixb2)
       hilf3=f(iyb2)

       call GERADE(hilf2,hilf3,hilf1,yu,ase1,
     #  1,lvs5,f(ixb1+bord),f(iyb1+bord))

c Rand 2       
       pob=ase(1,1)+ase(2,1)+ase(3,1)+ase(8,1)-1
       do za=0,ase1-1,1
        f(ixb2+bord+za)=f(ixb2+pob+za)      
        f(iyb2+bord+za)=f(iyb2+pob+za)
       end do
c Rand 1       
   


       
  
c Vernetzen      
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,2,lvs7,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       do za=1,ase(14,2),1
         xl26(za)=f(ixr+seed-ase2+za-1)
         yl26(za)=f(iyr+seed-ase2+za-1)
       end do 
       do za=1,ase(12,1),1
         xl63(za)=f(ixr+seed-ase2+za+ase(14,2)-2)
         yl63(za)=f(iyr+seed-ase2+za+ase(14,2)-2)
       end do 





       xdr6=f(ixr+seed-ase2+dat10-1)
       ydr6=f(iyr+seed-ase2+dat10-1)
       bord=ase1+bord 

c ----------------------------------------------------------------------
c Gebiet 10, Achtung: Rapu werden berechnet - G1 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(10,1)
       ase2=ase(10,2)
c Rand 1       
       hilf1=(yu-f(iyp))/tan(beta)+f(ixp)+delta
 

       call GERADE(f(ixp)+delta,f(iyp),hilf1,yu,ase1,
     #  1,lvs5,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       pob=0
       

       hilf1=((yu-f(iyp))/tan(beta)+f(ixp)+delta)-aint(bgs)/sin(beta)
       xdr3=hilf1
       ydr3=yu
 
       call GERADE(f(ixb2),f(iyb2),hilf1,yu,ase1,
     #  1,lvs5,f(ixb2+bord),f(iyb2+bord))
 
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do   
 
c
       bord=ase1+bord
       bord2=ase2+bord2



c ----------------------------------------------------------------------
c Nachlauf
c ----------------------------------------------------------------------
c
c ----------------------------------------------------------------------
c Gebiet 11
c ----------------------------------------------------------------------
       ase1=ase(11,1)
       ase2=ase(11,2)
 
c Rand 1       
       hilf1=(yu-f(iyp))/tan(beta)+f(ixp)
 
       call GERADE(f(ixp),f(iyp),hilf1,yu,ase1,
     #  1,lvs5,f(ixb1+bord),f(iyb1+bord))
c Rand 2       
       
       
       pob=ase(1,1)+ase(2,1)+ase(3,1)+ase(4,1)-1
       hilf1=((yu-f(iyp))/tan(beta)+f(ixp))+aint(bgs)/sin(beta)

       xdr1=hilf1
       ydr1=yu
       xdr2=f(ixb2+pob)
       ydr2=f(iyb2+pob)

       call GERADE(f(ixb2+pob),f(iyb2+pob),hilf1,yu,ase1,
     #  1,lvs5,f(ixb2+bord),f(iyb2+bord))

       do za=1,ase(14,1),1
         xl25(za)=f(ixb2+bord+za-1)
         yl25(za)=f(iyb2+bord+za-1)
       end do 
       do za=1,ase(13,1),1
         xl51(za)=f(ixb2+bord+za+ase(14,1)-2)
         yl51(za)=f(iyb2+bord+za+ase(14,1)-2)
       end do 
       
       xdr5=f(ixb2+bord+dat10-1)
       ydr5=f(iyb2+bord+dat10-1)
 
       do za=bord,ase1-1+bord,1
        call GERADE(f(ixb2+za),f(iyb2+za),f(ixb1+za),f(iyb1+za),
     #   ase2,1,bvs1,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do   
 
c
       bord=ase1+bord
       bord2=ase2+bord2
 
c





c ----------------------------------------------------------------------
c Gebiet 12, Achtung: Rapu werden berechnet - G9 muss gleiche Rapu haben
c ----------------------------------------------------------------------


       xdr4=(xdr1+xdr3)/2.
       ydr4=yu
      
       xdr7=(xdr1+xdr3)/4.+xdr2/2.
       ydr7=(ydr1+ydr2+ydr3)/3.



       ase1=ase(12,1)
       ase2=ase(12,2)


       call GERADE(xdr7,ydr7,xdr4,ydr4,ase1,
     #  0,0.,f(ixb2+bord),f(iyb2+bord))
c       
       do za=bord,ase1-1+bord,1
        call GERADE(xl63(za-bord+1),yl63(za-bord+1),
     #   f(ixb2+za),f(iyb2+za),ase2,0,0.,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord 
 

c ----------------------------------------------------------------------
c Gebiet 13, Achtung: Rapu werden berechnet - G4 muss gleiche Rapu haben
c ----------------------------------------------------------------------
       ase1=ase(13,1)
       ase2=ase(13,2)

       call GERADE(xdr7,ydr7,xdr4,ydr4,ase1,
     #  0,0.,f(ixb2+bord),f(iyb2+bord))
c       
       do za=bord,ase1-1+bord,1
        call GERADE(xl51(za-bord+1),yl51(za-bord+1),
     #   f(ixb2+za),f(iyb2+za),ase2,0,0.,f(ixr+seed),f(iyr+seed))
        seed=seed+ase2
       end do 
       bord=ase1+bord
c       bord2=ase2+bord2
 
 
c ----------------------------------------------------------------------
c Gebiet 14
c ----------------------------------------------------------------------
       ase1=ase(14,1)
       ase2=ase(14,2)


       call GERADE(xdr6,ydr6,xdr7,ydr7,ase1,
     #  0,0.,f(ixb1+bord),f(iyb1+bord))
       call GERADE(xdr5,ydr5,xdr7,ydr7,ase1,
     #  0,0.,f(ixb2+bord),f(iyb2+bord))

c       
       call NETZ(xl25,yl25,f(ixb1+bord),f(iyb1+bord),
     #           xl26,yl26,f(ixb2+bord),f(iyb2+bord),ase1,ase2,
     #           f(ixr+seed),f(iyr+seed))
        seed=seed+ase1*ase2
       
c
       bord=ase1+bord 
c       bord2=ase2+bord2


c ----------------------------------------------------------------------
c ENDE: VERNETZUNGSROUTINE (BERECHNUNG DER PUNKTE)
c ----------------------------------------------------------------------
c
c ----------------------------------------------------------------------
c UEBERPRUEFUNG DER DIMENSIONIERUNG
c ----------------------------------------------------------------------
c       if ((bord.ne.d_bord).OR.(seed.ne.d_seed).OR.(bord2.ne.
c     #  d_bord2)) then
       if ((bord.ne.d_bord).OR.(seed.ne.d_seed)) then
         print*, 'STOP: PROGRAMM HAT MESHSEEDS FALSCH VORAUSBERECHNET !'
     #   ,bord,d_bord,bord2,d_bord2,seed,d_seed
         stop
       endif

c ----------------------------------------------------------------------
c RUECKTRANSFORMATION DER KNOTEN INS KARTESISCHE KOORDINATENSYSTEM
c ----------------------------------------------------------------------
       call PU_RUECK(seed,radius,f(ixr),f(iyr),norm_fr,norm_fp,
     # norm_fz,f(ixk),f(iyk),f(izk))
c ----------------------------------------------------------------------
c GITTERPUNKTE IN RADIALER RICHTUNG VERSCHIEBEN
c ----------------------------------------------------------------------
       if (MOVR_TF) call MOVR_KNOT(seed,f(ixk),f(iyk),f(izk),radius,
     #  r_nab,r_kra,apvr,rpvrN,rpvrK,zpvr)
c ----------------------------------------------------------------------
c ERZEUGE NETZ
c ----------------------------------------------------------------------
       call CREATE_NET(f(ixr),f(iyr),f(ixk),f(iyk),f(izk),seed,ase,
     #  a_gebi,e(izli),f(ixkn+an_kno*(schnitt-1)),f(iykn+an_kno*
     #  (schnitt-1)),f(izkn+an_kno*(schnitt-1)),an_kno,e(iela),
     #  e(ielb),e(ielc),e(ield),a_2Del,an_grk,e(igrkli),KEA)
c ----------------------------------------------------------------------
c ERZEUGE RAND (KNOTEN,ELEMENTE)
c ----------------------------------------------------------------------
       if ((AUSG.eq.0).and.(CRE_RB)) then 
        if (schnitt.eq.b_anf) then
         print*, 'CRE_RAND'
         call CRE_RAND(seed,a_2Del,e(iela),e(ielb),e(ielc),e(ield),
     #    an_grk,e(igrkli),anz_re,e(ireli),anz_rk,e(irkli))
         print*, anz_re,' Randelemente'
c
         call ZUORD_RAND(anz_rk,e(irkli),yu,yo,ase(11,1),f(ixpel1),
     #    f(iypel1),f(ixper1),f(iyper1),ase(10,1),f(ixpel2),f(iypel2),
     #    f(ixper2),f(iyper2),ase(6,2),f(ixpel3),f(iypel3),f(ixper3),
     #    f(iyper3),a_gebi,ase,seed,f(ixr),f(iyr),e(izli),KEA,
     #    an_nak,e(inakli),an_rok,e(irokli),an_ruk,e(irukli),an_rlk,
     #    e(irlkli),an_rrk,e(irrkli))
         print*, an_nak,' Knoten am Profil'
         print*, an_rok,' Knoten am Eintritt'
         print*, an_ruk,' Knoten am Austritt'
         print*, an_rlk,' Knoten am linken Rand'
         print*, an_rrk,' Knoten am rechten Rand'
c ----------------------------------------------------------------------
c ERZEUGE RANDBEDINGUNGEN 2D
c ----------------------------------------------------------------------
         call RB2D_WAND(anz_re,e(ireli),a_2Del,e(iela),e(ielb),
     #    e(ielc),e(ield),an_nak,e(inakli),seed,e(iwrba),e(iwrbb),
     #    e(iwrbe),an_wrb)
         if (wen_kb.ne.0) call RB2D_KBIL(anz_re,e(ireli),a_2Del,e(iela),
     #    e(ielb),e(ielc),e(ield),an_nak,e(inakli),wen_kb,seed,e(ikbia),
     #    e(ikbib),e(ikbie),e(ikbiw),an_kbi)
         if (weo_kb.ne.0) call RB2D_KBIL(anz_re,e(ireli),a_2Del,e(iela),
     #    e(ielb),e(ielc),e(ield),an_rok,e(irokli),weo_kb,seed,e(ikbia),
     #    e(ikbib),e(ikbie),e(ikbiw),an_kbi)
         if (weu_kb.ne.0) call RB2D_KBIL(anz_re,e(ireli),a_2Del,e(iela),
     #    e(ielb),e(ielc),e(ield),an_ruk,e(irukli),weu_kb,seed,e(ikbia),
     #    e(ikbib),e(ikbie),e(ikbiw),an_kbi)
         if (wel_kb.ne.0) call RB2D_KBIL(anz_re,e(ireli),a_2Del,e(iela),
     #    e(ielb),e(ielc),e(ield),an_rlk,e(irlkli),wel_kb,seed,e(ikbia),
     #    e(ikbib),e(ikbie),e(ikbiw),an_kbi)
         if (wer_kb.ne.0) call RB2D_KBIL(anz_re,e(ireli),a_2Del,e(iela),
     #    e(ielb),e(ielc),e(ield),an_rrk,e(irrkli),wer_kb,seed,e(ikbia),
     #    e(ikbib),e(ikbie),e(ikbiw),an_kbi)
         call RB2D_MARK(an_rlk,e(irlkli),wel_km,seed,e(ikman),e(ikmaw),
     #    an_kma)
         call RB2D_MARK(an_rrk,e(irrkli),wer_km,seed,e(ikman),e(ikmaw),
     #    an_kma)
         if (weo_km.ne.0) then
          call RB2D_MARK(an_rok,e(irokli),weo_km,seed,e(ikman),e(ikmaw),
     #     an_kma)
         else
          call RB2D_KRB(an_rok,e(irokli),seed,e(ikrbn),an_krb)
         endif
        endif
        if (weo_km.eq.0) call WERT_KRB(an_krb,e(ikrbn),v_ein,r_nab,
     #   r_kra,(alpha-val),seed,e(izli),radius,f(ixr),f(iyr),norm_fr,
     #   norm_fp,norm_fz,l_mod,f(ikrbx+an_krb*(schnitt-1)),
     #   f(ikrby+an_krb*(schnitt-1)),f(ikrbz+an_krb*(schnitt-1)),
     #   f(ikrbk+an_krb*(schnitt-1)),f(ikrbe+an_krb*(schnitt-1)),an3_kw)
       endif
c ----------------------------------------------------------------------
c ENDE DER KNOTENBERECHNUNG
c ----------------------------------------------------------------------
      end do             
c ----------------------------------------------------------------------
c ERZEUGE 3D_ELEMENTE
c ----------------------------------------------------------------------
      if (AUSG.eq.0) call CRE_3DELM(e(iela),e(ielb),e(ielc),e(ield),
     # a_2Del,an_kno,bi,e(iel1),e(iel2),e(iel3),e(iel4),e(iel5),
     # e(iel6),e(iel7),e(iel8),a_3Del)
c ----------------------------------------------------------------------
c ERZEUGE 3D_RANDBEDINGUNGEN
c ----------------------------------------------------------------------
      if ((AUSG.eq.0).and.(CRE_RB)) then
       call RB3D_KRB(an_krb,e(ikrbn),an_kno,bi,e(ikrb1),an3_kr)
       call RB3D_MARK(an_kma,e(ikman),e(ikmaw),an_kno,bi,e(ikma1),
     #  e(ikma2),an3_km)
       call RB3D_WAND(an_wrb,e(iwrba),e(iwrbb),e(iwrbe),a_2Del,
     #  a_3Del,e(iel1),e(iel2),e(iel3),e(iel4),e(iel5),e(iel6),
     #  e(iel7),e(iel8),an_kno,bi,e(iwrb1),e(iwrb2),e(iwrb3),e(iwrb4),
     #  e(iwrb5),an3_wr)
       call RB3D_KBIL(an_kbi,e(ikbia),e(ikbib),e(ikbie),e(ikbiw),
     #  a_2Del,an_kno,bi,e(ikbi1),e(ikbi2),e(ikbi3),e(ikbi4),
     #  e(ikbi5),e(ikbi6),an3_kb)
      endif
c ----------------------------------------------------------------------
c ERSTELLE GEO_FILES
c ----------------------------------------------------------------------
      if (AUSG.ne.0) then
       call CRE_2DGEO(radius,f(ixr),f(iyr),seed,AUSG,e(iela),e(ielb),
     #  e(ielc),e(ield),a_2Del,e(izli),an_kno)
      else
        if (netz_speichern.eq.1) call CRE_3DGEO(f(ixkn),f(iykn),
     #  f(izkn),an_kno,bi,e(iel1),
     #  e(iel2),e(iel3),e(iel4),e(iel5),e(iel6),e(iel7),e(iel8),
     #  a_3Del)
c ----------------------------------------------------------------------
c ERSTELLE RB_FILES
c ----------------------------------------------------------------------
      if ((CRE_RB).and.(netz_speichern.eq.1)) call CRE_3DRB(an3_kr,
     #  e(ikrb1),an3_kw,f(ikrbx),
     #  f(ikrby),f(ikrbz),f(ikrbk),f(ikrbe),an3_wr,e(iwrb1),e(iwrb2),
     #  e(iwrb3),e(iwrb4),e(iwrb5),an3_kb,e(ikbi1),e(ikbi2),e(ikbi3),
     #  e(ikbi4),e(ikbi5),e(ikbi6),an3_km,e(ikma1),e(ikma2))
      endif
c ----------------------------------------------------------------------
c PROGRAMM ENDE
c ----------------------------------------------------------------------


      end
c ----------------------------------------------------------------------
c UNTERPROGRAMME
c ----------------------------------------------------------------------
c
c ----------------------------------------------------------------------
c ZUZO_NACA 
c ----------------------------------------------------------------------
      subroutine ZUZO_NACA(k,b,z,zu,zo)
c
c Sucht den obersten und untersten Wert der z-Koordinate des Feldes
c
c EINGABE
c k......Geometriepunkte pro Radialschnitt
c b......Radialschnitte 
c z......Koordinate der Geometriepunkte 
c
c AUSGABE
c zu.....Unterster Wert der z-Koordinate
c zo.....Oberster Wert der z-Koordinate
c      
      implicit none
c
      integer k,b,i,j
c
      real z(k,b),zu,zo
c
      zu=z(1,1)
      zo=z(1,1)
c      
      do i=1,b,1
       do j=1,k,1
        if (z(j,i).lt.zu) zu=z(j,i)
        if (z(j,i).gt.zo) zo=z(j,i)
       end do
      end do
c
      end    
c ----------------------------------------------------------------------
c GERADE 
c ----------------------------------------------------------------------
      subroutine GERADE(x1,y1,x2,y2,seed,m,l12,x,y)
c
c Berechnet die Punkte einer Gerade
c
c EINGABE
c x1,y1...Koordinaten Startpunkt
c x2,y2...Koordinaten Endpunkt
c seed....Anzahl der Punkte
c m.......Modus: m=0 Aequidistante Einteilung (L12 ignoriert)
c                m=1 Einteilung im Verhaeltnis L1/L2 (eine Seite)
c                m=2 Einteilung im Verhaeltnis L1/L2 (beide Seiten)
c l12.....Verhaeltnis L1/L2
c
c AUSGABE
c x,y Koordinaten Geradenpunkte 
c
      implicit none
c
      integer seed,i,m,DIM
c
      parameter (DIM=1000)
c      
      real x1,y1,x2,y2,x(seed),y(seed),l12 
c
      real LANG,PHI,TAB(DIM)
c
      if (seed.gt.DIM) then
        print*, 'GERADE: DIM zu klein'
        return
      endif
c
      LANG=(DBLE(x1-x2)**2+DBLE(y1-y2)**2)**0.5
      PHI=DATAN(DABS(DBLE(y1-y2))/DABS(DBLE(x1-x2))) 
      call EINT_L12(seed,0.,LANG,m,l12,TAB)
c
      do i=1,seed,1 
       x(i)=x1+real(TAB(i))*sign(1.,(x2-x1))*real(cos(PHI))
       y(i)=y1+real(TAB(i))*sign(1.,(y2-y1))*real(sin(PHI))
      end do
c
      end
c ----------------------------------------------------------------------
c RAND_SPOTS
c ----------------------------------------------------------------------
      subroutine RAND_SPOTS(x,y,s,ymin,ymax,valN,valK,vbe,weinhi,
     # schnitt,bi,pxu,pxo,wi_ein,val,wi_aus,radius,r_nab,r_kra)
c
c Berechnet den An- und Abstroemwinkel, die Gebietswinkel, die 
c Verschiebewinkel sowie den Schnittpunkt mit der oberen bzw. unteren 
c Grenze
c
c EINGABE
c x,y........ Koordinaten der Splinepunkte der Sehne  
c s.......... Anzahl der Splinepunkte
c ymin....... untere Grenze
c ymax....... obere Grenze  
c valN....... Verschiebung von alpha an der Nabe
c valK....... Verschiebung von alpha am Kranz
c vbe........ Verschiebung von beta
c weinhi..... Winkel wi_ein vom vorherigen Radialschnitt
c schnitt.... aktueller Radialschnitt
c bi......... Anzahl der Radialschnitte
c radius..... Radius des aktuellen Zylinderschnitts
c r_nab...... Radius an der Nabe
c r_kra...... Radius am Kranz
c
c AUSGABE
c pxu........ Koordinaten des unteren Randpunktes
c pxo........ Koordinaten des oberen Randpunktes
c wi_ein..... Gebietswinkel Vorlauf
c val........ Verschiebung von alpha
c wi_aus..... Gebietswinkel Nachlauf
c
      implicit none
      integer s,dif,schnitt,bi
      real x(s),y(s),pxu,pyu,pxo,pyo,ymin,ymax,PI,
     # alpha,beta,valN,valK,val,vbe,wi_aus,wi_ein,weinhi,
     # radius,r_nab,r_kra,eins_durch_tan_wi_ein

c
      parameter(PI=3.141592654)
c      
      dif=int(s*0.05+0.5)
      if (dif.lt.1) dif=1
      alpha=atan2((y(s)-y(s-dif)),(x(s)-x(s-dif)))
      beta=atan2((y(2)-y(1)),(x(2)-x(1)))
c
      if (valK.le.PI) then
       val=valN+(valK-valN)*(radius-r_nab)/(r_kra-r_nab)
       print*,'val: ',val,radius,r_kra,r_nab
      else
       if (schnitt.eq.1) then
	val=valN
       else 
	val=weinhi-alpha
       endif
      endif
c
      wi_ein=PI/2.0+val
      if (wi_ein.eq.PI/2.0) then
        eins_durch_tan_wi_ein=0
      else
        eins_durch_tan_wi_ein=1./tan(wi_ein)
      endif

      wi_aus=beta+vbe
      weinhi=wi_ein
c      
      pxu=(ymin-y(1))/tan(wi_aus)+x(1)
      pyu=ymin
      pxo=(ymax-y(s))*eins_durch_tan_wi_ein+x(s)
      pyo=ymax 
c
      print*, 'Winkel Alpha/Beta= ',alpha,'   ',beta
      print*, 'Verschiebung Alpha/Beta= ',val,'   ',vbe
c      
      end
c ----------------------------------------------------------------------
c N_FASER
c ----------------------------------------------------------------------
      subroutine N_FASER(m,xp,yp,n,x,y)
c
c Berechnet die neutrale Faser des Profils
c
c EINGABE
c m........ Anzahl Punkte des Profilschnittes 
c xp,yp.... Punkte des Profilschnittes
c n........ Anzahl Punkte der Sehne
c
c AUSGABE
c x,y...... Punkte der Sehne
c
      integer m,n,i,a 
      real xp(m),yp(m),x(n),y(n)
c
      do i=1,n 
       a=(m+1)-i
       x(i)=(xp(i)+xp(a))/2.
       y(i)=(yp(i)+yp(a))/2.
      end do
c
      end
c ----------------------------------------------------------------------
c MESHSEED
c ----------------------------------------------------------------------
      subroutine MESHSEED(XK,YK,MO,AA,AE,PA,PE,VX,VY,VT,M,L12,N,NL
     # ,XS,YS)
c
c Berechnet die Punkte eines kubischen Splines
c
c EINGABE
c
c XK...Stuetzpunkt x-Koordinate
c YK...Stuetzpunkt y-Koordinate
c MO...Modus: 0..Anfangs/Endsteigung wird ueber nachforlgende/
c                vorhergehenden Knoten berechnet 
c             1..Angabe der Steigungen ueber AA,AE
c AA...1. Ableitung am 1. Stuetzpunkt (wird bei MO=0 ignoriert) 
c AE...1. Ableitung am letzten Stuetzpunkt (wird bei MO=0 ignoriert)
c PA...Nummer des Stuetzpunktes ab dem Punkte gesetzt werden sollen
c PE...Nummer des Stuetzpnuktes bis zu dem Punkte gesetzt werden 
c      sollen
c VX...Betrag der Verschiebung in x-Richtung
c VY...Betrag der Verschiebung in y-Richtung
c VT...Betrag der Verschiebung in Richtung des Normalenvektors
c M....Modus: M=0 Aequidistante Einteilung (L12 ignoriert)
c             M=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
c             M=2 Einteilung im Verhaeltnis L1/L2 (beide Richtungen)
c L12..Verhaeltnis L1/L2
c N....Nummer des letzten Stuetzpunktes
c NL...Anzahl der gewuenschten Kurvenpunkte
c
c AUSGABE
c XS...Punkte x-Koordinate
c YS...Punkte y-Koordinate
c
      implicit none
c
      integer N,IB,MT,IERR,NL,PA,PE,MO,i,M,DIM1,DIM2
      parameter (MT=2,IB=3,DIM1=500,DIM2=1000)
      real XK(0:N),YK(0:N),XS(0:NL-1),YS(0:NL-1),VX,VY,VT,AA,AE,
     # L12 
      DOUBLE PRECISION X(0:DIM1),Y(0:DIM1),XTAB(0:DIM2-1),
     # YTAB(0:DIM2-1),BX(0:DIM1),CX(0:DIM1),DX(0:DIM1),BY(0:DIM1),
     # CY(0:DIM1),DY(0:DIM1),DUMMY(1:5*DIM1+1),T(0:DIM1),ALPHA(2),
     # BETA(2),XVO,YVO,XNA,YNA,INFT,e1,e2,betr,XX(0:DIM1),YY(0:DIM1)
c
      if (N.gt.DIM1) then
        print*, 'MESHSEED: DIM1 zu klein'
        return
      elseif (NL.gt.DIM2) then
        print*, 'MESHSEED: DIM2 zu klein'
        return
      endif
c
      ALPHA(2)=0
      BETA(2)=0
      if (MO.eq.0) then
       ALPHA(1)=DBLE(YK(1)-YK(0))/DBLE(XK(1)-XK(0))
       BETA(1)=DBLE(YK(N)-YK(N-1))/DBLE(XK(N)-XK(N-1))
      else
       ALPHA(1)=DBLE(AA)
       BETA(1)=DBLE(AE)
      endif
c
      do i=0,N
        X(i)=DBLE(XK(i))
        Y(i)=DBLE(YK(i))
      end do
c
      call ISPLPA (N,X,Y,T,MT,IB,ALPHA,BETA,BX,CX,DX,BY,CY,DY,
     # DUMMY,IERR)
       if (IERR.ne.0) then
          PRINT*, 'ISPLAPA: FEHLER = ',IERR
          return
       endif
c
      call PSPOINTS (N,NL,T(PA),T(PE),T,X,BX,CX,DX,Y,BY,CY,DY,
     # M,L12,XTAB,YTAB)
c
      if (VT.ne.0.) then
c
       INFT=(T(N)-T(0))/1000
       do i=0,N
        call PSPVAL(T(i)+INFT,N,T,X,BX,CX,DX,Y,BY,CY,DY,XVO,YVO) 
        call PSPVAL(T(i)-INFT,N,T,X,BX,CX,DX,Y,BY,CY,DY,XNA,YNA) 
        e1=XVO-XNA
        e2=YVO-YNA
        betr=(e1**2+e2**2)**0.5
        XX(i)=X(i)-e2/betr*DBLE(VT)
        YY(i)=Y(i)+e1/betr*DBLE(VT)
       end do
       if (MO.eq.0) then
        ALPHA(1)=(YY(1)-YY(0))/(XX(1)-XX(0))
        BETA(1)=(YY(N)-YY(N-1))/(XX(N)-XX(N-1))
       endif
c
       call ISPLPA (N,XX,YY,T,MT,IB,ALPHA,BETA,BX,CX,DX,BY,CY,DY,
     #  DUMMY,IERR)
       if (IERR.ne.0) then
          PRINT*, 'ISPLAPA: FEHLER = ',IERR
          return
       endif
c
       call PSPOINTS (N,NL,T(PA),T(PE),T,XX,BX,CX,DX,YY,BY,CY,DY,
     #  M,L12,XTAB,YTAB)
c
      endif 
c
      do i=0,NL-1
       XS(i)=real(XTAB(i))+VX
       YS(i)=real(YTAB(i))+VY
      end do
c
      end
c ----------------------------------------------------------------------
c PSPOINTS
c ----------------------------------------------------------------------
      subroutine PSPOINTS(N,NL,TBEG,TEND,T,AX,BX,CX,DX,AY,BY,CY,DY,
     # M,L12,XTAB,YTAB)
c
c Gibt die Punkte eines berechneten kubischen Splines aus
c
c EINGABE
c
c N.......Nummer des letzten Stuetzpunktes
c NL......Anzahl der gewuenschten Kurvenpunkte
c TBEG....Anfang der Ausgabe von Kurvenpunkten 
c TEND....Ende der Ausgabe von Kurvenpunkten
c T.......Laengenparameter
c AX..DY..Splinekoeffizienten
c M.......Modus: M=0 Aequidistante Einteilung (L12 ignoriert)
c                M=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
c                M=2 Einteilung im Verhaeltnis L1/L2 (beide Richt.)
c L12.....Verhaeltnis L1/L2
c
c AUSGABE
c
c XTAB...Punkte x-Koordinate
c YTAB...Punkte y-Koordinate
c
      implicit none
c
      integer N,NL,i,M
c
      DOUBLE PRECISION TBEG,TEND,T(0:N),AX(0:N),BX(0:N),CX(0:N),
     # DX(0:N),AY(0:N),BY(0:N),CY(0:N),DY(0:N),XTAB(0:NL-1),
     # YTAB(0:NL-1),TV,sw,swm,schr1,schr2
      real L12
c
      swm=(TEND-TBEG)/(NL-1) 
      if (m.eq.1) then
        schr1=2*swm/(1+1/DBLE(L12))
        schr2=2*swm/(1+DBLE(L12))
      elseif ((m.eq.2).and.((NL/2.).ne.int(NL/2.))) then
        schr1=2*swm/(1+1/DBLE(L12))
        schr2=2*swm/(1+DBLE(L12))
      elseif ((m.eq.2).and.((NL/2.).eq.int(NL/2.))) then
        schr1=2*(TEND-TBEG)/((1/DBLE(L12))*(DBLE(L12)*NL+NL-2))
        schr2=2*(TEND-TBEG)/(DBLE(L12)*NL+NL-2)
      endif
c     
      TV=TBEG 
c
      do i=0,NL-1,1
       if (i.eq.0) then
            sw=0
       elseif (m.eq.1) then 
            sw=schr1+(schr2-schr1)*(i-1)/(NL-2)
       elseif ((m.eq.2).and.((NL/2.).ne.int(NL/2.))) then   
         if (i.le.((NL-1)/2)) then
            sw=schr1+(schr2-schr1)*(i-1)/((NL-3)/2.)
         else
            sw=schr2+(schr1-schr2)*(i-1-((NL-1)/2))/((NL-3)/2.)
         endif   
       elseif ((m.eq.2).and.((NL/2.).eq.int(NL/2.))) then
         if (i.le.(NL/2)) then 
            sw=schr1+(schr2-schr1)*(i-1)/((NL-2)/2.)
         else
            sw=schr2+(schr1-schr2)*(i-(NL/2))/((NL-2)/2.) 
         endif
       else
         sw=swm 
       endif
c
       TV=TV+sw      
c
       call PSPVAL (TV,N,T,AX,BX,CX,DX,AY,BY,CY,DY,XTAB(i),YTAB(i))
      end do   
c
      end
c ----------------------------------------------------------------------
c CRE_NACA
c ----------------------------------------------------------------------
      subroutine CRE_NACA(k,b,x,y,z,schnitt,m,L12,radius,ip,iz,fr,fp,
     # fz)
c
c Transformiert die Geometriepunkte aus dem kartesischen in ein 
c zylindrisches Koordinatensystem und normiert die r-Koord. so, dass 
c die Punkte quasi auf der Ebene abgerollt werden. Doppelte Punkte 
c werden geloescht, um das Versplinen zu ermoeglichen. Die Stuetzpunkte
c werden mit einem Spline durch die Geometriepunkte erzeugt. 
c
c EINGABE
c k..........Geometriepunkte pro Radialschnitt
c b..........Radialschnitte 
c x,y,z......Koordinaten der Geometriepunkte 
c schnitt....Schnitt fuer den Stuetzpunkte ausgegeben werden sollen
c m..........Anzahl Stuetzpunkte 
c L12....... Verhaeltnis, mit dem Stuetzpunkte erzeugt werden
c
c AUSGABE
c radius,ip,iz...Ausgabe der Punkte in Zylinderkoordinaten 
c                (radius ist konst.)
c fr,fp,fz.......Normierungsfaktoren
c
      implicit none
c
      integer k,b,schnitt,m,DIM1,DIM2
c
      parameter (DIM1=38,DIM2=100)      
c       
      real x(k,b),y(k,b),z(k,b),
     #     r(DIM1,DIM2),p(DIM1,DIM2),fr,fp,fz,L12,radius,ip(m),iz(m)
c
      if (k.gt.DIM1) then
        print*, 'CRE_NACA: DIM1 zu klein'
        return
      elseif (b.gt.DIM2) then
        print*, 'CRE_NACA: DIM2 zu klein'
        return
      endif
c        
      call GPU_TRAFO(k,b,x,y,r,p)
      call NORM_BOG(k,b,r,p,schnitt,fr,fp,fz)
      call CRE_SP(k,b,r,p,z,schnitt,m,L12,radius,ip,iz) 
c
      end
c ----------------------------------------------------------------------
c INP_DAT
c ----------------------------------------------------------------------
      subroutine INP_DAT(file_name,n,b,k,x,y,z)
c
c Liest die Daten aus dem Geometriefile ein. Die Ausgabe der Punkte er
c folgt in einer versplinebaren Reihenfolge
c
c EINGABE
c n...... vorg Punkte pro Seite eines Naca-Profiles
c b ..... vorg. Schnitte durch Schaufelprofil
c k ..... vorg. Punkte auf NACA-Profil
c
c AUSGABE
c x,y,z . kartesische Koordinaten  
c
      implicit none 
      integer n,b,k,m,l,z1,z2
      real x(k,b),y(k,b),z(k,b)
      character*16 file_name
c
      open(13,file=file_name)
      read(13,*) z1,z2
      do l=1,b
       do m=1,n
        read(13,*) x(n-m+1,l),y(n-m+1,l),z(n-m+1,l),
     #  x(m+n,l),y(m+n,l),z(m+n,l)        
       end do 
      end do 
      close(13)
c
      if ((z1.ne.n).or.(z2.ne.b)) then
        print*, 'INP_DAT: Falsches Datenformat'
        return
      endif
c
      end




c ----------------------------------------------------------------------
c INP_ONLINE
c ----------------------------------------------------------------------
      subroutine INP_ONLINE(x,y,z,a_flug)
c
c Uebernimmt die Daten online aus dem C++-Programm. Die Ausgabe der
c Punkte erfolgt in einer versplinebaren Reihenfolge
c

c
c AUSGABE
c x,y,z . kartesische Koordinaten  
c a_flug  Anzahl der Fluegel
c

      parameter (IMAX=19,npla=11,k=2*IMAX)
      integer m,l,z1,z2
      integer nlschaufel,a_flug
      real x(k,npla),y(k,npla),z(k,npla)
      real  x_ds(IMAX,npla), y_ds(IMAX,npla), z_ds(IMAX,npla),
     .      x_ss(IMAX,npla), y_ss(IMAX,npla), z_ss(IMAX,npla)  
      real  d2, di_da, n_z2, B0, D1, D2Z, z_axe_la 
      common /xyz/  x_ds(IMAX,npla), y_ds(IMAX,npla), z_ds(IMAX,npla),
     .              x_ss(IMAX,npla), y_ss(IMAX,npla), z_ss(IMAX,npla)
      common /zwis/  d2, di_da, nlschaufel, n_z2, B0, D1, D2Z,
     .               z_axe_la
c      

      do l=1,npla
       do m=1,IMAX
        x(IMAX-m+1,l) = x_ds(m,l)
        y(IMAX-m+1,l) = y_ds(m,l)
        z(IMAX-m+1,l) = z_ds(m,l)
        x(m+IMAX,l)   = x_ss(m,l)
        y(m+IMAX,l)   = y_ss(m,l)
        z(m+IMAX,l)   = z_ss(m,l)  
       end do 
      end do 
      a_flug=nlschaufel
c
      end










c ----------------------------------------------------------------------
c GPU_TRAFO
c ----------------------------------------------------------------------
      subroutine GPU_TRAFO(k,b,x,y,r,p)
c
c Transformiert die Koordinaten aus dem kartesischen in das r-phi System
c
c EINGABE
c k ........ Anzahl vorg. Geometriepunkte auf NACA-Profil
c b ........ Anzahl vorg. Schnitte durch Schaufelprofil
c r,p ...... Zylinderkoordinaten
c
c AUSGABE
c r,p ...... Zylinderkoordinaten
c
       implicit none
       integer b,n,m,k
       real x(k,b),y(k,b),r(k,b),p(k,b)
c
       do n=1,b
       do m=1,k
        r(m,n)=sqrt(x(m,n)**2+y(m,n)**2)
        p(m,n)=atan2(y(m,n),x(m,n))
       end do 
      end do 
c
      end
c ----------------------------------------------------------------------
c NORM_100
c ----------------------------------------------------------------------
      subroutine NORM_100(k,b,r,p,z,fr,fp,fz)
c
c Normiert die Koordinaten r und p so, dass die globale Differenz 
c zwischen max. und min. Wert 100 ist. 
c
c EINGABE
c k ........ Anzahl vorg. Geometriepunkte auf NACA-Profil
c b ........ Anzahl vorg. Schnitte durch Schaufelprofil
c r,p,z .... Zylinderkoordinaten
c
c AUSGABE
c r,p,z .... Zylinderkoordinaten
c fr,fp,fz . Normierungsfaktoren
c
      implicit none
      integer b,m,n,k
      real r(k,b),p(k,b),z(k,b),fr,fp,fz,min,max
c
      min=r(1,1)
      max=r(1,b)
      fr=1.
c  
      min=z(1,1)
      max=z(1,1)
c  
      do n=1,b
       do m=1,k
        if (z(m,n).LT.min) min=z(m,n)
        if (z(m,n).GT.max) max=z(m,n)
       end do
      end do
c
      fz=1/(max-min)*100
c
      min=p(1,1)
      max=p(1,1)
c  
      do n=1,b
       do m=1,k
        if (p(m,n).LT.min) min=p(m,n)
        if (p(m,n).GT.max) max=p(m,n)
       end do
      end do
c
      fp=1/(max-min)*100
c
c
      do n=1,b
       do m=1,k
        r(m,n)=r(m,n)*fr
        p(m,n)=p(m,n)*fp
        z(m,n)=z(m,n)*fz
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c NORM_BOG
c ----------------------------------------------------------------------
      subroutine NORM_BOG(k,b,r,p,s,fr,fp,fz)
c
c Normiert die Koordinate p so, dass die Radialschnitte quasi auf einer 
c Ebene abgerollt werden.
c    
c
c EINGABE
c k ........ Anzahl vorg. Geometriepunkte auf NACA-Profil
c b ........ Anzahl vorg. Schnitte durch Schaufelprofil
c r,p,z .... Zylinderkoordinaten
c s......... Schnitt, fuer den normiert werden soll
c
c AUSGABE
c r,p,z .... Zylinderkoordinaten
c fr,fp,fz . Normierungsfaktoren
c
      implicit none
      integer b,m,s,k
      real r(k,b),p(k,b),fr,fp,fz,min,max
c
      fr=1.
      fz=1.
c
      min=p(1,s)
      max=p(1,s)
c  
       do m=1,k
        if (p(m,s).LT.min) min=p(m,s)
        if (p(m,s).GT.max) max=p(m,s)
       end do
c
      fp=r(1,s)
c
c
       do m=1,k
        p(m,s)=p(m,s)*fp
       end do
c
      end
c ----------------------------------------------------------------------
c CRE_SP
c ----------------------------------------------------------------------
      subroutine CRE_SP(k,b,r,p,z,sc,m,L12,ir,ip,iz) 
c
c Versplint Geometriepunkte (aufeinanderfolgende dopp. Punkte werden 
c geloescht) und berechnet die Stuetzpunkte
c
c EINGABE
c k ........ Anzahl vorg. Geometriepunkte auf NACA-Profil
c b ........ Anzahl vorg. Schnitte durch Schaufelprofil
c r,p,z .... Geometriepunkte in Zylinderkoordinaten
c sc........ Schnitt, fuer den Stuetzpunkte ausgegeben werden sollen
c m......... Anzahl der zu versplinenden Stuetzpunkte
c L12....... Verhaeltnis, mit dem Stuetzpunkte erzeugt werden
c
c AUSGABE
c ir,ip,iz.. Koordinaten der auszugebenden Stuetzpunkte (ir ist konst.)
c
      implicit none
      integer i,sc,k,b,m,DIM,dop
      parameter (DIM=38)
      real r(k,b),p(k,b),z(k,b),ir,sp(DIM),sz(DIM),ip(m),iz(m),L12,TOL
      logical IERR 
      parameter (TOL=0.001)
c
      if (k.gt.DIM) then
        print*, 'CRE_SP: DIM zu klein'
        return
      endif  
c Eliminiere aufeinanderfolgende dopp. Punkte
      sp(1)=p(1,sc)
      sz(1)=z(1,sc)
      dop=0
      do i=2,k
       if (((p(i,sc)-p(i-1,sc))**2+(z(i,sc)-z(i-1,sc))**2)**0.5.le.TOL)
     #  then     
         dop=dop+1
       else
         sp(i-dop)=p(i,sc)
         sz(i-dop)=z(i,sc)
       endif
      end do
      print*, 'Doppelte Geometriepunkte: ',dop
c Pruefe, ob Punkte auf einem Radialschnitt liegen     
      IERR=.FALSE.
      ir=r(1,sc)
      do i=2,k,1
       if (abs(r(i,sc)-r(1,sc)).gt.TOL) then
         print*, 'CRE_SP: ES LIEGEN KEINE RADIALSCHNITTE VOR - ',
     #    r(i,sc),r(1,sc)
	 IERR=.TRUE.
       endif
      end do
c      if (IERR) return
c
      if (L12.gt.0.) then
        call MESHSEED(sp(1),sz(1),0,0.,0.,0,nint((k-1)/2.)-1,0.,0.,0.,1,
     #   L12,(k-dop)-1,int((m+1)/2),ip(1),iz(1))
        call MESHSEED(sp(1),sz(1),0,0.,0.,nint((k-1)/2.)-1,(k-dop)-1,0.,
     #   0.,0.,1,(1/L12),(k-dop)-1,int((m+1)/2),ip(int((m+1)/2)),
     #   iz(int((m+1)/2)))
      else
        print*, 'CRE_SP: Fehler beim Erzeugen der Stuetzpunkte'
        return
      endif
c
      end
c ----------------------------------------------------------------------
c PU_RUECK
c ----------------------------------------------------------------------
      subroutine PU_RUECK(seed,radius,ixr,iyr,fr,fp,fz,ixk,iyk,izk)
c
c Transformiert und normiert die Koordinaten in das urspruengliche 
c Koordinatensystem zurueck.
c
c EINGABE
c seed...........Anzahl der Knoten
c radius,ixr,iyr.normierte Zylinderkoordinaten (radius ist konst.)
c fr,fp,fz.......Normierungsfaktoren
c
c AUSGABE
c ixk,iyk,izk....Knoten in kart. Kooord.system
c
      implicit none
c
      integer seed,i
c
      real radius,ixr(seed),iyr(seed),r,p,fr,fp,fz,
     # ixk(seed),iyk(seed),izk(seed)
c
c
      do i=1,seed
       r=radius/fr
       p=ixr(i)/fp
       izk(i)=iyr(i)/fz
c
       ixk(i)=r*cos(p)
       iyk(i)=r*sin(p)
      end do
c
      end
c ----------------------------------------------------------------------
c CRE_2DGEO
c ----------------------------------------------------------------------
      subroutine CRE_2DGEO(radius,pu_x,pu_y,ap,AUSG,elm_a,elm_b,
     # elm_c,elm_d,anz_elm,zuli,anz_kno)
c
c Gibt ein 2D-Geofile aus.
c
c EINGABE
c radius.........Radius der Ebene, in der sich der Radialschnitt befin 
c pu_x,pu_y......Koordinaten der Punkte
c ap.............Anzahl der Punkte
c AUSG...........Nummer des Radialschnitts (fuer bez. des DAT-Files)
c elm_a..elm_d...Knotennummern der 2D-Elemente
c anz_elm........Anzahl der Elemente
c zuli...........Zuordnungsliste Knoten-Punkte
c anz_kno........Anzahl der Knoten
c
      implicit none
c
      integer DIM
      parameter (DIM=65535)
c
      integer ap,elm_a(ap),elm_b(ap),elm_c(ap),elm_d(ap),anz_elm,
     # zuli(ap),anz_kno,AUSG,i
c
      real pu_x(ap),pu_y(ap),knolipz_x(DIM),knolipz_y(DIM),radius
c
      character*13, file
      character*4 nummer
c ----------------------------------------------------------------------
c Ueberpruefe Dimension
c ----------------------------------------------------------------------
      if (anz_kno.gt.DIM) then 
        print*, 'CRE_2DGEO: DIM zu klein'
        return
      endif 
c ----------------------------------------------------------------------
c Kopiere Knoten
c ----------------------------------------------------------------------
       do i=1,ap
        knolipz_x(zuli(i))=pu_x(i)
        knolipz_y(zuli(i))=pu_y(i)
        if (zuli(i).gt.anz_kno) then
          print*, 'CRE_2DGEO: Fehler beim Kopieren der Knoten'
          return
        endif
       end do 
c ----------------------------------------------------------------------
c Ausgabe GEO-File zylindrisch
c ----------------------------------------------------------------------
c Format entspricht Institutsnorm
c
      print*, 'Schreibe GEO-File zylindrisch' 
c      
      write(nummer,'(i4.4)') AUSG
      file(1:5)='PZ2D_'
      file(6:9)=nummer
      file(10:13)='.GEO'
      open(15,file=file)
c      
      do i=1,10
       write(15,*) 'C'
      end do
      write(15,*) anz_kno,anz_elm,' 0 0 0 0 ',anz_kno,anz_elm
c
c Ausgabe der Knoten      
      do i=1,anz_kno
       write(15,'(I7,3(F12.3))') i,knolipz_x(i),knolipz_y(i),radius
      end do
c
c Ausgabe der Elemente
      do i=1,anz_elm
       write(15,'(4(I7))') elm_a(i),elm_b(i),elm_c(i),elm_d(i)
      end do
c
      close(15)
c
      end
c ----------------------------------------------------------------------
c CREATE_NET
c ----------------------------------------------------------------------
      subroutine CREATE_NET(x,y,xk,yk,zk,ap,ase,ag,zuli,klikax,
     # klikay,klikaz,anz_kno,elm_a,elm_b,elm_c,elm_d,anz_elm,
     # an_grk,grkli,TOL)
c
c Erzeugt aus den eingegebenen Punkten ein Netz mit Knoten und Elemente
c
c EINGABE
c x,y............Zylinderkoordinaten (ohne radius da konst.) der Punkte
c xk,yk,zk.......Kartesische Koordinaten der Punkte
c ap.............Anzahl der Punkte
c ase............Anzahl der Meshseeds (Randpunkte jedes Gebietes) 
c ag.............Anzahl der Gebiete
c
c AUSGABE
c zuli...........Zuordnungsliste Punkte-Knoten
c klikax..z......Koordinaten der Knoten
c anz_kno........Anzahl der Knoten
c elm_a..elm_d...Knotennummern je 2D-Element
c anz_elm........Anzahl der Elemente
c an_grk.........Anzahl der Gebietsrandknoten
c grkli..........Knotennummern der Gebietsrandknoten
c TOL............Kleinster Elementabstand
c
      implicit none
c
      integer DIM
      parameter (DIM=65535)
c
      integer ag,ap,ase(ag,2),verli(DIM),grenz,grenzli(DIM),zuli(ap),i,
     # j,k,dop,dopli(DIM,2),pos,anz_kno,anz_elm,anz_dop,elm(DIM,4),
     # elm_a(ap),elm_b(ap),elm_c(ap),elm_d(ap),an_grk,grkli(DIM)

c
      real x(ap),y(ap),xk(ap),yk(ap),zk(ap),klikax(ap),
     # klikay(ap),klikaz(ap),TOL,ABST
c
      if (ap.gt.DIM) then 
        print*, 'CREATE_NET: DIM zu klein'
        return
      endif 
c ----------------------------------------------------------------------
c Abfrage ob Knoten und Elemente schon erzeugt wurden
c ----------------------------------------------------------------------
      if (zuli(1).ne.12345) then 
c ----------------------------------------------------------------------
c Kopiere Knoten
c ----------------------------------------------------------------------
       do i=1,ap
        klikax(zuli(i))=xk(i)
        klikay(zuli(i))=yk(i)
        klikaz(zuli(i))=zk(i)
        if (zuli(i).gt.anz_kno) then
          print*, 'CREATE_NET: Fehler beim Kopieren der Knoten'
          return
        endif
       end do 
c ----------------------------------------------------------------------
c Ordne Punkte Elemente zu   
c ----------------------------------------------------------------------
      else
c
       anz_elm=0
       pos=0
       do i=1,ap
        verli(i)=0
       end do
       do i=1,ag
        do j=1,ase(i,1)
         do k=1,ase(i,2)
          pos=pos+1
          if ((k.lt.ase(i,2)).and.(j.lt.ase(i,1))) then
            anz_elm=anz_elm+1 
            elm(anz_elm,1)=pos
            elm(anz_elm,2)=pos+1
            elm(anz_elm,3)=pos+ase(i,2)+1
            elm(anz_elm,4)=pos+ase(i,2)
            verli(pos)=verli(pos)+1
            verli(pos+1)=verli(pos+1)+1
            verli(pos+ase(i,2)+1)=verli(pos+ase(i,2)+1)+1
            verli(pos+ase(i,2))=verli(pos+ase(i,2))+1
          endif 
         end do
        end do
       end do
c Kontrolle 
       if (ap.ne.pos) then 
        print*, 'CRE_2DGEO: Fehler beim Erstellen der Elemente'
        return
       endif
c ----------------------------------------------------------------------
c Berechne kleinste Toleranz 
c ----------------------------------------------------------------------
       TOL=1000.
       do i=1,anz_elm,1
c Kanten       
        do j=1,3,2
         do k=2,4,2
          ABST=((x(elm(i,j))-x(elm(i,k)))**2+(y(elm(i,j))-y(elm(i,k)))**
     #     2)**0.5
          if (ABST.lt.TOL) TOL=ABST
         end do
        end do
c Ûberkreuzt
        do j=1,2,1
         ABST=((x(elm(i,j))-x(elm(i,j+2)))**2+(y(elm(i,j))-y(elm(i,j+2))
     #    )**2)**0.5
         if (ABST.lt.TOL) TOL=ABST
        end do
       end do 
c ----------------------------------------------------------------------
c Suche nach Punkte, die weniger als 4 Elementen zugeordnet sind - 
c Gebietsrandknoten
c ----------------------------------------------------------------------
       grenz=0
       do i=1,ap
        if (verli(i).lt.4) then
         grenz=grenz+1
         grenzli(grenz)=i
        endif
       end do
c ----------------------------------------------------------------------
c Suche gemeinsame Koordinaten und erstelle Liste mit doppelten Punkte 
c ----------------------------------------------------------------------
       anz_dop=0
       do i=1,grenz
        do j=1,i-1
         ABST=((x(grenzli(i))-x(grenzli(j)))**2+(y(grenzli(i))-y(grenzli
     #    (j)))**2)**0.5
         if (ABST.lt.(TOL*0.9)) then 
           anz_dop=anz_dop+1
           dopli(anz_dop,1)=grenzli(i)
           dopli(anz_dop,2)=grenzli(j)
           goto 10 
         endif
        end do
10      continue      
       end do
c ----------------------------------------------------------------------
c Erstelle Zuordnungsliste Punkte - Knoten
c ----------------------------------------------------------------------
       dop=0
       anz_kno=0
       do i=1,ap
         if ((i.eq.dopli(dop+1,1)).and.(dop.lt.anz_dop)) then 
           zuli(i)=zuli(dopli(dop+1,2))
           dop=dop+1
         else 
           anz_kno=anz_kno+1 
           zuli(i)=i-dop
           klikax(anz_kno)=xk(i)
           klikay(anz_kno)=yk(i)
           klikaz(anz_kno)=zk(i)
         endif
       end do
       if (dop.ne.anz_dop) then
         print*, 'CRE_2DGEO: Fehler beim Erstellen der Zuordn.liste'
         return  
       endif
c ----------------------------------------------------------------------
c Ersetzte bei Elemente Punkte durch Knoten
c ----------------------------------------------------------------------
       do i=1,anz_elm
        do j=1,4
         elm(i,j)=zuli(elm(i,j)) 
        end do
       end do
c ----------------------------------------------------------------------
c Bildschirmausgabe
c ----------------------------------------------------------------------
       print*, 'Kleinster Elementabstand',TOL
       print*, 'Anzahl der doppelten Punkte:',anz_dop
       print*, 'Anzahl der Knoten:',anz_kno
       print*, 'Anzahl der Elemente: ',anz_elm
c ----------------------------------------------------------------------
c Schreibe Elementeknoten in 1-DIM Feld
c ----------------------------------------------------------------------
       do i=1,anz_elm
        elm_a(i)=elm(i,1)
        elm_b(i)=elm(i,2)
        elm_c(i)=elm(i,3)
        elm_d(i)=elm(i,4)
       end do
c ----------------------------------------------------------------------
c Erzeuge Gebietsrandknotenliste
c ----------------------------------------------------------------------
       an_grk=0
       do i=1,grenz
        do j=1,anz_dop
         if (grenzli(i).eq.dopli(j,1)) goto 20
        end do
        an_grk=an_grk+1
        grkli(an_grk)=zuli(grenzli(i))
20     continue
       end do
       if (an_grk.ne.(grenz-anz_dop)) then
        print*, 'Fehler beim Erzeugen der Gebietsrandknotenliste'
       endif
c
      endif
c
      end
c ----------------------------------------------------------------------
c INP_STF
c ----------------------------------------------------------------------
      subroutine INP_STF(dat_file,MOVR_TF,mov_fi,bi,schnitt,af,
     # yu,yo,bgs,valN,valK,vbe,ael,aer,L12N,L12K,m,PG,dat1,dat2,dat3,
     # dat4,dat5,dat6,dat7,dat8,dat9,dat10,lvs1,lvs2,lvs3,lvs4,lvs5,
     # lvs6,lvs7,bvs1,bvs2,bvs3,bvs4,bvs5,rvs,vsfdec,gevs1N,gevs2N,
     # gevs1K,gevs2K,CRE_RB,v_ein,l_mod,wen_kb,weo_kb,weu_kb,
     # wel_kb,wer_kb,wel_km,wer_km,weo_km)
c
c
c AUSGABE
c dat_file..Name des einzulesenden Datenfiles
c MOVR_TF...adialkontur beruecksichtigen (TRUE/FALSE)
c mov_fi....Radialkontur_Datenfile
c bi........Anzahl Radialschnitte
c schnitt...Ausgabe Radialschnitt
c af........Anzahl Fluegel
c yu........Untere Grenze [mm]
c yo........Obere Grenze [mm]
c bgs.......Breite Grenzschicht [mm]
c valN......Verschiebung Alpha Nabe [bog]
c valK......Verschiebung Alpha Kranz [bog]
c vbe.......Verschiebung Beta [bog]
c ael.......Aufweitung Eintritt links [bog]
c aer.......Aufweitung Eintritt rechts [bog]
c L12N......L12 der Stuetzpunkte an der Nabe
c L12K......L12 der Stuetzpunkte am Kranz
c m.........Anzahl Stuetzpunkte (ungerade)
c PG........Stuetzpunkt 1
c PG........Stuetzpunkt 2
c PG........Stuetzpunkt 3
c PG........Stuetzpunkt 4
c PG........Stuetzpunkt 5
c dat1......Anzahl Knoten Gebiet 1 laengs
c dat2......Anzahl Knoten Grenzschicht quer
c dat3......Anzahl Knoten Gebiet 2 laengs
c dat4......Anzahl Knoten Gebiet 3 laengs
c dat5......Anzahl Knoten Aussenseite quer
c dat6......Anzahl Knoten Vorlauf laengs
c dat7......Anzahl Knoten Nachlauf laengs
c lvs1......Verdichtung Gebiet 1 laengs, beidseitig
c lvs2......Verdichtung Gebiet 2 laengs, Richt. Staupunkt
c lvs3......Verdichtung Gebiet 3 laengs, Richt. Hinterkante
c lvs4......Verdichtung Gebiet 4 laengs, Richt. Hinterkante
c lvs5......Verdichtung Nachlauf laengs, Richt. Austritt
c lvs6......Verdichtung Mitte laengs, Richt. Vorlauf
c lvs7......Verdichtung Vorlauf laengs, Richt. Eintritt
c bvs1......Verdichtung Grenzschicht quer, Richt. Profil
c bvs2......Verdichtung li. Seite quer Mitt-Ob., Richt. Per. Rand
c bvs3......Verdichtung re. Seite quer Mitt-Ob., Richt. Per. Rand
c bvs4......Verdichtung re. Seite quer Nachlauf, Richt. Per. Rand
c bvs5......Verdichtung re. Seite quer Nachlauf, Richt. Per. Rand
c rvs.......Verdichtung in radialer Richtung, beidseitig
c vsfdec....VerschFak fuer Unters. Dreieck
c gevs1N....VerschFak fuer Schwpu Deck links an der Nabe
c gevs2N....VerschFak fuer Schwpu Deck rechts an der Nabe
c gevs1K....VerschFak fuer Schwpu Deck links am Kranz
c gevs2K....VerschFak fuer Schwpu Deck rechts am Kranz
c CRE_RB....Randbedingungsfile erstellen (.T/.F)
c v_ein.....Betrag der Eintrittsgeschw.
c l_mod.....Modellaenge (k-eps-Model)
c wen_kb....Bilanzflaechennr. Profil
c weo_kb....Bilanzflaechennr. Eintr.
c weu_kb....Bilanzflaechennr. Austr.
c wel_kb....Bilanzflaechennr. Pe.Ra.li.
c wer_kb....Bilanzflaechennr. Pe.Ra.re.
c weo_km....Knotenmarkierung Eintr.
c wel_km....Knotenmarkierung Pe.Ra.li.
c wer_km....Knotenmarkierung Pe.Ra.re.
c
      implicit none
c

      integer bi,schnitt,af,PG(5),m,dat1,dat2,dat3,dat4,dat5,dat6,dat7,
     # dat8,dat9,dat10,wen_kb,weo_kb,weu_kb,wel_kb,wer_kb,
     # wel_km,wer_km,weo_km
c
      real yu,yo,bgs,valN,valK,vbe,ael,aer,L12N,L12K,vsfdec,gevs1N,
     #     gevs2N,gevs1K,gevs2K,v_ein,l_mod,lvs1,lvs2,lvs3,lvs4,lvs5,
     #     lvs6,lvs7,bvs1,bvs2,bvs3,bvs4,bvs5,rvs
c
      logical CRE_RB,MOVR_TF

c
      character*200 dat_file,mov_fi 
      character*200 datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb

      common /ver2/ datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
c
      open(14,file=datei_ax_stf)
      read(14,*)
      read(14,*)
      read(14,*)
      read(14,*)
      read(14,*)
      read(14,'(T18,A20)',ERR=1) dat_file
      read(14,'(T44,L2)',ERR=1) MOVR_TF
      read(14,'(T24,A20)',ERR=1) mov_fi
      read(14,'(T23,I2)',ERR=1) bi
      read(14,'(T23,I2)',ERR=1) schnitt
      read(14,'(T16,I2)',ERR=1) af
      read(14,'(T20,F5.0)',ERR=1) yu
      read(14,'(T19,F5.0)',ERR=1) yo
      read(14,'(T26,F5.3)',ERR=1) bgs
      read(14,'(T31,F5.3)',ERR=1) valN
      read(14,'(T32,F5.3)',ERR=1) valK
      read(14,'(T25,F5.3)',ERR=1) vbe
      read(14,'(T33,F5.3)',ERR=1) ael
      read(14,'(T34,F5.3)',ERR=1) aer
      read(14,'(T34,F5.3)',ERR=1) L12N
      read(14,'(T31,F5.3)',ERR=1) L12K
      read(14,'(T32,I3)',ERR=1) m
      read(14,'(T15,I3)',ERR=1) PG(1)
      read(14,'(T15,I3)',ERR=1) PG(2)
      read(14,'(T15,I3)',ERR=1) PG(3)
      read(14,'(T15,I3)',ERR=1) PG(4)
      read(14,'(T15,I3)',ERR=1) PG(5)
      read(14,'(T51,I3)',ERR=1) dat1
      read(14,'(T51,I3)',ERR=1) dat2
      read(14,'(T51,I3)',ERR=1) dat3
      read(14,'(T51,I3)',ERR=1) dat4
      read(14,'(T51,I3)',ERR=1) dat6
      read(14,'(T51,I3)',ERR=1) dat7
      read(14,'(T51,I3)',ERR=1) dat8
      read(14,'(T51,I3)',ERR=1) dat9
      read(14,'(T48,F5.3)',ERR=1) lvs1 
      read(14,'(T48,F5.3)',ERR=1) lvs2 
      read(14,'(T48,F5.3)',ERR=1) lvs3 
      read(14,'(T48,F5.3)',ERR=1) lvs4 
      read(14,'(T48,F5.3)',ERR=1) lvs5 
      read(14,'(T42,F5.3)',ERR=1) lvs6 
      read(14,'(T45,F5.3)',ERR=1) lvs7 
      read(14,'(T46,F5.3)',ERR=1) bvs1 
      read(14,'(T55,F5.3)',ERR=1) bvs2 
      read(14,'(T55,F5.3)',ERR=1) bvs3 
      read(14,'(T55,F5.3)',ERR=1) bvs4 
      read(14,'(T55,F5.3)',ERR=1) bvs5 
      read(14,'(T46,F5.3)',ERR=1) rvs 
      read(14,'(T32,F5.3)',ERR=1) vsfdec 
      read(14,'(T46,F5.3)',ERR=1) gevs1N
      read(14,'(T47,F5.3)',ERR=1) gevs2N
      read(14,'(T43,F5.3)',ERR=1) gevs1K
      read(14,'(T44,F5.3)',ERR=1) gevs2K
      read(14,'(T43,L2)',ERR=1) CRE_RB
      read(14,'(T29,F5.3)',ERR=1) v_ein
      read(14,'(T27,F5.3)',ERR=1) l_mod
      read(14,'(T25,I3)',ERR=1) wen_kb
      read(14,'(T28,I3)',ERR=1) weo_kb
      read(14,'(T28,I3)',ERR=1) weu_kb
      read(14,'(T35,I3)',ERR=1) wel_kb
      read(14,'(T36,I3)',ERR=1) wer_kb
      read(14,'(T27,I3)',ERR=1) weo_km
      read(14,'(T34,I3)',ERR=1) wel_km
      read(14,'(T35,I3)',ERR=1) wer_km
      goto 2
c
1     continue
      print*, ('INP_STF: Fehler beim Einlesen des Steuerfiles')
      return
2     continue
c    
      close(14)
      print*, dat_file,MOVR_TF,mov_fi,bi,schnitt,af,yu,yo,bgs,valN,valK,
     # vbe,ael,aer,
     # L12N,L12K,m,PG,dat1,dat2,dat3,dat4,dat5,dat6,dat7,
     # lvs1,lvs2,lvs3,lvs4,lvs5,lvs6,lvs7,bvs1,bvs2,bvs3,bvs4,bvs5,rvs,
     # vsfdec,gevs1N,gevs2N,gevs1K,gevs2K,CRE_RB,v_ein,l_mod,wen_kb,
     # weo_kb,weu_kb,wel_kb,wer_kb,weo_km,wel_km,wer_km
      end
c ----------------------------------------------------------------------
c SCHNI_PU
c ----------------------------------------------------------------------
      subroutine SCHNI_PU(xa,ya,xb,yb,xc,yc,xd,yd,xe,ye)
c
c Berechnet den Schnittpunkt zweier Geraden
c
c EINGABE
c xa,ya.....Punkt a, Gerade 1
c xb,yb.....Punkt b, Gerade 1
c xc,yc.....Punkt c, Gerade 2
c xd,yd.....Punkt d, Gerade 2
c
c AUSGABE
c xe,ye.....Schnittpunkt
c
      implicit none
c
c     
      real xa,ya,xb,yb,xc,yc,xd,yd,xe,ye,m1,m2,c1,c2
c
      m1=(yb-ya)/(xb-xa)
      m2=(yd-yc)/(xd-xc)
      c1=ya-xa*m1
      c2=yc-xc*m2
      xe=(c1-c2)/(m2-m1)
      ye=m1*xe+c1
c
      end
c ----------------------------------------------------------------------
c NETZ
c ----------------------------------------------------------------------
      subroutine NETZ(xa,ya,xb,yb,xc,yc,xd,yd,seed1,seed2,xe,ye)
c
c Vernetzt ein viereckiges Gebiet
c Die Vernetzung laeuft von a nach b und von c nach d, so dass auf c
c die ersten und auf d die letzten Punkte liegen. Alle Raender
c muessen Geraden sein.
c
c EINGABE
c xa,ya.....Punkt a, Rand 1, muss eine Gerade sein
c xb,yb.....Punkt b, Rand 2, muss eine Gerade sein
c xc,yc.....Punkt c, Rand 3, muss eine Gerade sein
c xd,yd.....Punkt d, Rand 4, muss eine Gerade sein
c seed1.....Anzahl Punkte Rand 1,2
c seed2.....Anzahl Punkte Rand 3,4
c
c AUSGABE
c xe,ye.....Knotenpunkte 
c
      implicit none
c
      integer seed1,seed2,a,c
c
      real xa(seed1),ya(seed1),xb(seed1),yb(seed1),xc(seed2),yc(seed2),
     #  xd(seed2),yd(seed2),xe(seed1*seed2),ye(seed1*seed2)
c
      do a=1,seed1
       do c=1,seed2
        call SCHNI_PU(xa(a),ya(a),xb(a),yb(a),xc(c),yc(c),xd(c),yd(c),
     #   xe(c+(a-1)*seed2),ye(c+(a-1)*seed2))
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c NETZ2
c ----------------------------------------------------------------------
      subroutine NETZ2(xa,ya,xb,yb,xc,yc,xd,yd,seed1,seed2,xe,ye)
c
c Vernetzt ein viereckiges Gebiet 
c Die Vernetzung laeuft von a nach b und von c nach d, so dass auf c
c die ersten und auf d die letzten Punkte liegen. Wenn c keine 
c Gerade ist wird dies per Verschiebungsvektor interpoliert. Die 
c anderen Raender muessen Geraden sein.
c Achtung:  Die Reihenfolge der Punkte ist entscheidend:
c a(1) und b(1) liegen auf Gerade c
c a(seed1) und b(seed1) liegen auf Gerade d  
c
c EINGABE
c xa,ya.....Punkt a, Rand 1, muss eine Gerade sein
c xb,yb.....Punkt b, Rand 2, muss eine Gerade sein
c xc,yc.....Punkt c, Rand 3, kann eine Kurve sein
c xd,yd.....Punkt d, Rand 4, muss eine Gerade sein
c seed1.....Anzahl Punkte Rand 1,2
c seed2.....Anzahl Punkte Rand 3,4
c
c AUSGABE
c xe,ye.....Knotenpunkte 
c
      implicit none
c
      integer seed1,seed2,a,c,DIM
c
      parameter(DIM=1000)
c
      real xa(seed1),ya(seed1),xb(seed1),yb(seed1),xc(seed2),yc(seed2),
     #  xd(seed2),yd(seed2),xe(seed1*seed2),ye(seed1*seed2),
     #  dx(DIM),dy(DIM)
c
      if (seed2.gt.DIM) then
        print*, 'NETZ2: DIM zu klein'
        return
      endif  
c
c  Berechnung des Vektors der 1. Schnittpunkte zu den Punkten auf Rand c
      do c=1,seed2 
       call SCHNI_PU(xa(1),ya(1),xb(1),yb(1),xc(c),yc(c),xd(c),yd(c),
     #  xe(c),ye(c))
       dx(c)=xc(c)-xe(c)
       dy(c)=yc(c)-ye(c)
      end do
c
      do a=1,seed1
       do c=1,seed2
        call SCHNI_PU(xa(a),ya(a),xb(a),yb(a),xc(c),yc(c),xd(c),yd(c),
     #   xe(c+(a-1)*seed2),ye(c+(a-1)*seed2))
c
c Verschiebung der Punkte mit abnehmender Gewichtung       
        xe(c+(a-1)*seed2)=xe(c+(a-1)*seed2)+dx(c)*(seed1-a)/(seed1-1)
        ye(c+(a-1)*seed2)=ye(c+(a-1)*seed2)+dy(c)*(seed1-a)/(seed1-1)
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c NETZ3
c ----------------------------------------------------------------------
      subroutine NETZ3(xa,ya,xb,yb,xc,yc,xd,yd,seed1,seed2,xe,ye)
c
c Vernetzt ein viereckiges Gebiet 
c Die Vernetzung laeuft von a nach b und von c nach d, so dass auf 
c c(1) der erste und auf d(seed2) der letzte Punkte liegt. Wenn a 
c oder c keine Gerade ist wird dies per Verschiebungsvektor 
c interpoliert. Die anderen Raender muessen Geraden sein.
c Achtung:  Die Reihenfolge der Punkte ist entscheidend:
c a(1) und b(1) liegen auf Gerade c
c a(seed1) und b(seed1) liegen auf Gerade d  
c c(1) und d(1) liegen auf Gerade a
c c(seed2) und d(seed2) liegen auf Gerade b  
c
c EINGABE
c xa,ya.....Punkt a, Rand 1, muss eine Gerade sein
c xb,yb.....Punkt b, Rand 2, muss eine Gerade sein
c xc,yc.....Punkt c, Rand 3, kann eine Kurve sein
c xd,yd.....Punkt d, Rand 4, muss eine Gerade sein
c seed1.....Anzahl Punkte Rand 1,2
c seed2.....Anzahl Punkte Rand 3,4
c
c AUSGABE
c xe,ye.....Knotenpunkte 
c
      implicit none
c
      integer seed1,seed2,a,c,DIM
c
      parameter(DIM=1000)
c
      real xa(seed1),ya(seed1),xb(seed1),yb(seed1),xc(seed2),yc(seed2),
     #  xd(seed2),yd(seed2),xe(seed1*seed2),ye(seed1*seed2),
     #  dx1(DIM),dy1(DIM),dx2(DIM),dy2(DIM)
c
      if ((seed2.gt.DIM).or.(seed1.gt.DIM)) then
        print*, 'NETZ2: DIM zu klein'
        return
      endif  
c
c  Berechnung des Vektors der 1. Schnittpunkte zu den Punkten auf Rand c
      do c=1,seed2 
       call SCHNI_PU(xa(1),ya(1),xb(1),yb(1),xc(c),yc(c),xd(c),yd(c),
     #  xe(c),ye(c))
       dx2(c)=xc(c)-xe(c)
       dy2(c)=yc(c)-ye(c)
      end do
c  Berechnung des Vektors der 1. Schnittpunkte zu den Punkten auf Rand a
      do a=1,seed1 
       call SCHNI_PU(xa(a),ya(a),xb(a),yb(a),xc(1),yc(1),xd(1),yd(1),
     #  xe(a),ye(a))
       dx1(a)=xa(a)-xe(a)
       dy1(a)=ya(a)-ye(a)
      end do
c
      do a=1,seed1
       do c=1,seed2
        call SCHNI_PU(xa(a),ya(a),xb(a),yb(a),xc(c),yc(c),xd(c),yd(c),
     #   xe(c+(a-1)*seed2),ye(c+(a-1)*seed2))
c
c Verschiebung der Punkte mit abnehmender Gewichtung       
        xe(c+(a-1)*seed2)=xe(c+(a-1)*seed2)+dx2(c)*(seed1-a)/(seed1-1)+
     #   dx1(a)*(seed2-c)/(seed2-1)
        ye(c+(a-1)*seed2)=ye(c+(a-1)*seed2)+dy2(c)*(seed1-a)/(seed1-1)+
     #   dy1(a)*(seed2-c)/(seed2-1)
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c CRE_3DELM
c ----------------------------------------------------------------------
      subroutine CRE_3DELM(ela,elb,elc,eld,anz_2Delm,
     #  anz_kno,t,el1,el2,el3,el4,el5,el6,el7,el8,anz_3Delm)
c
c Erzeugt aus den 2D-Elementen zweier Radialschnitte 3D-Elemente
c
c EINGABE
c ela..eld...Knotennummern je 2D-Element
c anz_2Delm..Anzahl der 2D-Elemente
c anz_kno....Anzahl der Knoten
c t..........Anzahl der Radialschnitte
c
c AUSGABE
c el1..el8...Knotennummern je 3D-Element
c anz_3Delm..Anzahl der 3D-Elemente
c
      implicit none
c
      integer t,anz_kno,anz_2Delm,ela(anz_2Delm),elb(anz_2Delm),
     # elc(anz_2Delm),eld(anz_2Delm),el1(anz_2Delm*(t-1)),el2(anz_2Delm*
     # (t-1)),el3(anz_2Delm*(t-1)),el4(anz_2Delm*(t-1)),el5(anz_2Delm*
     # (t-1)),el6(anz_2Delm*(t-1)),el7(anz_2Delm*(t-1)),el8(anz_2Delm*
     # (t-1)),anz_3Delm,i,j
c
      anz_3Delm=0
c
      do i=1,t-1,1
       do j=1,anz_2Delm,1
        anz_3Delm=anz_3Delm+1
        el1(anz_3Delm)=ela(j)+anz_kno*(i-1)
        el2(anz_3Delm)=elb(j)+anz_kno*(i-1)
        el3(anz_3Delm)=elc(j)+anz_kno*(i-1)
        el4(anz_3Delm)=eld(j)+anz_kno*(i-1)
        el5(anz_3Delm)=ela(j)+anz_kno*(i)
        el6(anz_3Delm)=elb(j)+anz_kno*(i)
        el7(anz_3Delm)=elc(j)+anz_kno*(i)
        el8(anz_3Delm)=eld(j)+anz_kno*(i)
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c CRE_3DGEO
c ----------------------------------------------------------------------
      subroutine CRE_3DGEO(x,y,z,anz_kno,t,el1,el2,el3,el4,el5,el6,
     # el7,el8,anz_elm)
c
c Gibt ein 3D-Geofile aus
c
c EINGABE
c x,y,z.....Koordinaten der Knoten
c anz_kno...Anzahl der Knoten
c t.........Anzahl der Radialschnitte
c el1..el8..Knotennummern je 3D-Element
c anz_elm...Anzahl der Elemente
c
      implicit none
c
      integer anz_kno,t,anz_elm,el1(anz_elm),el2(anz_elm),el3(anz_elm),
     # el4(anz_elm),el5(anz_elm),el6(anz_elm),el7(anz_elm),el8(anz_elm),
     # i
c
      real x(anz_kno*t),y(anz_kno*t),z(anz_kno*t)


c
      
      character*200 datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
      common /ver2/  datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
c ----------------------------------------------------------------------
c Ausgabe GEO-File kartesisch
c ----------------------------------------------------------------------
c Format entspricht Institutsnorm
c
      print*, 'CRE_3DGEO' 
      print*, t*anz_kno,' Knoten '
      print*, anz_elm,' Elemente '
c                  
      open(16,file=datei_kart3d_geo)
c      
      do i=1,10,1
       write(16,*) 'C'
      end do
      write(16,*) anz_kno*t,anz_elm,' 0 0 0 0 ',anz_kno*t,anz_elm
c
c Ausgabe der Knoten      
      do i=1,anz_kno*t,1
       write(16,'(I7,3(F12.3))') i,x(i),y(i),z(i)
      end do
c
c Ausgabe der Elemente
      do i=1,anz_elm
       write(16,'(8(I7))') el1(i),el2(i),el3(i),el4(i),el5(i),el6(i),
     #  el7(i),el8(i)
      end do
c
      close(16)
c
      end
c ----------------------------------------------------------------------
c IPOL_RS
c ----------------------------------------------------------------------
      subroutine IPOL_RS(k,b,xo,yo,zo,bi,mo,l12,xa,ya,za)
c
c Interpoliert die vorgegebenen Radialschnitte um eine beliebige Anzahl
c in radialer Richtung
c
c EINGABE
c b ......... vorg. Radialschnitte durch Schaufelprofil
c k ......... vorg. Punkte auf NACA-Profil
c xo,yo,zo .. Koordinaten der vorg. Radialschnitte 
c bi......... Anzahl der zu interpolierenden Radialschnitte
c mo......... mo=0 Aequidistante Einteilung (L12 ignoriert)
c             mo=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
c             mo=2 Einteilung im Verhaeltnis L1/L2 (beide Richtungen)
c l12........ Verhaeltnis L1/L2
c
c AUSGABE
c xa,ya,za .. Koordinaten der interpolierten Radialschnitte 
c
      implicit none
c
      integer b,k,bi,mo,DIM1,DIM2,DIM3
c
      parameter (DIM1=38,DIM2=11,DIM3=100)
c
      real xo(k,b),yo(k,b),zo(k,b),l12,ro(DIM1,DIM2),po(DIM1,DIM2),
     # ra(DIM1,DIM3),pa(DIM1,DIM3),za(k,bi),xa(k,bi),ya(k,bi)
c
      if (k.gt.DIM1) then
       print*, 'IPOL_RS: DIM1 zu klein'
       return
      elseif (b.gt.DIM2) then
       print*, 'IPOL_RS: DIM2 zu klein'
       return
      elseif (bi.gt.DIM3) then
       print*, 'IPOL_RS: DIM3 zu klein'
       return
      endif 
c
      call GPU_TRAFO(k,b,xo,yo,ro,po)
      call INTERPOL(k,b,ro,po,zo,bi,mo,l12,ra,pa,za)
      call RTRAFO(k,bi,ra,pa,xa,ya) 
c
      end
c ----------------------------------------------------------------------
c INTERPOL
c ----------------------------------------------------------------------
      subroutine INTERPOL(k,b,xo,yo,zo,ia,mo,l12,xa,ya,za)
c
c Versplint die Geometriepunkte in radialer Richtung und interpoliert 
c diese
c
c EINGABE
c b.......... vorg. Radialschnitte durch Schaufelprofil
c k.......... vorg. Punkte auf NACA-Profil
c xo,yo,zo... Koordinaten der vorg. Radialschnitte 
c ia......... Anzahl der zu interpolierenden Radialschnitte
c mo......... mo=0 Aequidistante Einteilung (L12 ignoriert)
c             mo=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
c             mo=2 Einteilung im Verhaeltnis L1/L2 (beide Richtungen)
c l12........ Verhaeltnis L1/L2
c
c AUSGABE
c xa,ya,za... Koordinaten der interpolierten Radialschnitte 
c
      implicit none
c
      integer k,b,ia,mo,DIM1,DIM2,i,j,IERR,IB
      parameter (DIM1=11,DIM2=100)
      real xo(k,b),yo(k,b),zo(k,b),l12,xa(k,ia),ya(k,ia),za(k,ia)
      DOUBLE PRECISION ALPHA,BETA,XS(DIM1),YS(DIM1),ZS(DIM1),BY(DIM1),
     # BZ(DIM1),CY(DIM1),CZ(DIM1),DY(DIM1),DZ(DIM1),
     # DUMMY(5*(DIM1-1)+1),SPVAL,XI(DIM2),YI(DIM2),ZI(DIM2)
      real hilfXI(DIM2)
c Parameter ISPLNP
      parameter(IB=5,ALPHA=0,BETA=0)    
c Ueberpruefe Dimensionierung
      if (b.gt.DIM1) then
        print*, 'INTERPOL: DIM1 zu klein'
        return 
      elseif (ia.gt.DIM2) then
        print*, 'INTERPOL: DIM2 zu klein'
        return 
      endif
c Stuetzpunkte
      do i=1,k
       do j=1,b
        XS(j)=xo(i,j)
        YS(j)=yo(i,j)
        ZS(j)=zo(i,j)
       end do 
c Stuetzpunkte berechnen
       call ISPLNP ((b-1),XS,YS,IB,ALPHA,BETA,BY,CY,DY,DUMMY,IERR)
       call ISPLNP ((b-1),XS,ZS,IB,ALPHA,BETA,BZ,CZ,DZ,DUMMY,IERR)
c Splinepunkte berechnen
       call EINT_L12(ia,real(XS(1)),real(XS(b)),mo,l12,hilfXI)
c
       do j=1,ia
        XI(j)=hilfXI(j)
       end do 
       do j=1,ia,1
        YI(j)=SPVAL (XI(j),(b-1),XS,YS,BY,CY,DY)
        ZI(j)=SPVAL (XI(j),(b-1),XS,ZS,BZ,CZ,DZ)
       end do
c Rueckgabe der berechneten Punkte
       do j=1,ia,1
        xa(i,j)=XI(j)
        ya(i,j)=YI(j)
        za(i,j)=ZI(j)
       end do
c
      end do 
c
      end
c ----------------------------------------------------------------------
c EINT_L12
c ----------------------------------------------------------------------
      subroutine EINT_L12(NL,TBEG,TEND,M,L12,TAB)
c
c Teilt ein Intervall in einer beliebigen Anzahl und Einteilung auf
c
c
c EINGABE
c
c NL......Anzahl der gewuenschten Punkte
c TBEG....Anfang der Ausgabe von Punkten 
c TEND....Ende der Ausgabe von Punkten
c M.......Modus: M=0 Aequidistante Einteilung (L12 ignoriert)
c                M=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
c                M=2 Einteilung im Verhaeltnis L1/L2 (beide Richt.)
c L12.....Verhaeltnis L1/L2
c       
c
c AUSGABE
c
c TAB.....Einteilung (TBEG..TEND)  
c
      implicit none
c
      integer NL,i,M,DIM
c
      parameter (DIM=1000)
c
      real TBEG,TEND,TV,sw,swm,schr1,schr2,TAB(0:DIM)
      real L12
c
      swm=(TEND-TBEG)/(NL-1) 
      if (m.eq.1) then
        schr1=2*swm/(1+1/L12)
        schr2=2*swm/(1+L12)
      elseif ((m.eq.2).and.((NL/2.).ne.int(NL/2.))) then
        schr1=2*swm/(1+1/L12)
        schr2=2*swm/(1+L12)
      elseif ((m.eq.2).and.((NL/2.).eq.int(NL/2.))) then
        schr1=2*(TEND-TBEG)/((1/L12)*(L12*NL+NL-2))
        schr2=2*(TEND-TBEG)/(L12*NL+NL-2)
      endif
c     
      TV=TBEG 
c
      do i=0,NL-1,1
       if (i.eq.0) then
            sw=0
       elseif (m.eq.1) then 
            sw=schr1+(schr2-schr1)*(i-1)/(NL-2)
       elseif ((m.eq.2).and.((NL/2.).ne.int(NL/2.))) then   
         if (i.le.((NL-1)/2)) then
            sw=schr1+(schr2-schr1)*(i-1)/((NL-3)/2.)
         else
            sw=schr2+(schr1-schr2)*(i-1-((NL-1)/2))/((NL-3)/2.)
         endif   
       elseif ((m.eq.2).and.((NL/2.).eq.int(NL/2.))) then
         if (i.le.(NL/2)) then 
            sw=schr1+(schr2-schr1)*(i-1)/((NL-2)/2.)
         else
            sw=schr2+(schr1-schr2)*(i-(NL/2))/((NL-2)/2.) 
         endif
       else
         sw=swm 
       endif
c
       TV=TV+sw      
c
       TAB(i)=TV
c
      end do   
c
      end
c ----------------------------------------------------------------------
c RTRAFO
c ----------------------------------------------------------------------
      subroutine RTRAFO(k,b,r,p,x,y)
c
c Transformiert die Koordianten aus dem r-phi in das kartesische 
c Koordinatensystem
c
c EINGABE
c k ........ Anzahl vorg. Geometriepunkte auf Radialschnitt
c b ........ Anzahl vorg. Radialschnitte durch Schaufelprofil
c r,p ...... Zylinderkoordinaten
c
c AUSGABE
c x,y ...... kartesische Koordinaten
c
      implicit none
c
      integer k,b,i,j
c
      real r(k,b),p(k,b),x(k,b),y(k,b)
c
      do j=1,b
       do i=1,k
        x(i,j)=r(i,j)*cos(p(i,j))
        y(i,j)=r(i,j)*sin(p(i,j))
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c CRE_RAND
c ----------------------------------------------------------------------
      subroutine CRE_RAND(ap,anz_el,ela,elb,elc,eld,an_grk,grkli,
     # anz_re,reli,anz_rk,rkli)
c
c Sucht alle Randelemente und -knoten in einem Radialschnitt
c
c EINGABE
c ap..........Anzahl der Punkte (zur Dimensionierung)
c anz_el......Anzahl der Elemente
c ela..eld....Elementknoten a,b,c,d
c
c AUSGABE 
c an_grk......Anzahl der Gebietsrandknoten
c grkli.......Liste mit Gebietsrandknoten
c anz_re......Anzahl der Randelemente
c reli........Liste mit Randelelementen
c anz_rk......Anzahl der Randknoten
c rkli........Liste mit Randknoten
c   
      implicit none
c 
      integer ap,anz_el,ela(anz_el),elb(anz_el),elc(anz_el),eld(anz_el),
     # an_grk,grkli(an_grk),anz_re,reli(anz_el),anz_rk,rkli(ap),
     # el1(4),el2(4),tref,nach,a,b,c,d,eck(4)
      logical pointer(4)
c 
      anz_re=0
      anz_rk=0
c Elemente durchlaufen
      do a=1,anz_el
       nach=0
       do c=1,4
        eck(c)=0
       end do 
c
       el1(1)=ela(a)
       el1(2)=elb(a)
       el1(3)=elc(a)
       el1(4)=eld(a)
c Pruefe, ob mind. 2 Elementknoten den Gebietsrandknoten entsprechen
       tref=0
       do b=1,an_grk,1
        do c=1,4,1
         if (grkli(b).eq.el1(c)) tref=tref+1
        end do
       end do 
       if (tref.lt.2) goto 2 
c Elemente des Gebietes vergleichen
       do b=1,anz_el
        tref=0
        do c=1,4
         pointer(c)=.false.
        end do
c
        el2(1)=ela(b)
        el2(2)=elb(b)
        el2(3)=elc(b)
        el2(4)=eld(b)
c Vergleich die Eckpunkte beider Elemente
        do c=1,4
         do d=1,4
          if (el1(c).eq.el2(d)) then
           tref=tref+1              
           pointer(c)=.true.  
           goto 1   
          endif
         end do
1        continue   
        end do
c Ueberpruefe ob Element ein Nachbar ist
        if (tref.eq.2) then
         nach=nach+1
         do c=1,4
          if (pointer(c)) eck(c)=eck(c)+1
         end do
        endif
       end do
c Ueberpruefe ob Element weniger als 4 Nachbarn hat
       if (nach.lt.4) then
        anz_re=anz_re+1
        reli(anz_re)=a 
        do c=1,4
         if (eck(c).le.1) then 
          do d=1,anz_rk,1
	   if ((anz_rk.ne.0).and.(rkli(d).eq.el1(c))) goto 3 
          end do
	  anz_rk=anz_rk+1 
          rkli(anz_rk)=el1(c)      
3         continue          
	 endif
        end do  
       endif
2      continue
      end do
c
      end
c ----------------------------------------------------------------------
c ZUORD_RAND
c ----------------------------------------------------------------------
      subroutine ZUORD_RAND(anz_rk,rkli,yu,yo,ape1,xpel1,ypel1,xper1,
     # yper1,ape2,xpel2,ypel2,xper2,yper2,ape3,xpel3,ypel3,xper3,yper3,
     # ag,ase,ap,xr,yr,zuli,KEA,
     # an_nak,nakli,an_rok,rokli,an_ruk,rukli,an_rlk,rlkli,an_rrk,rrkli)
c
c Ordnet alle Randknoten den einzelnen Raender zu
c
c EINGABE
c anz_rk.....Anzahl der Randknoten
c rkli.......Liste mit Randknoten
c yu.........Untere Grenze
c yo.........Obere Grenze
c ape1.......Anzahl Knoten des 1. per Randes
c x,ypel1....Koordinaten der Knoten des 1. per. Randes
c ape2.......Anzahl Knoten des 2. per Randes
c x,ypel2....Koordinaten der Knoten des 2. per. Randes
c ape3.......Anzahl Knoten des 3. per Randes
c x,ypel3....Koordinaten der Knoten des 3. per. Randes
c ag.........Anzahl der Gebiete
c ase........Anzahl Meshseeds der einzelnen Gebiete
c ap.........Anzahl der Punkte 
c xr,yr......Koordinaten der Punkte
c zuli.......Zuordnungsliste Punkte-Knoten
c KEA........Kleinste Ekementgroesse
c 
c AUSGABE
c an_nak.....Anzahl der Knoten des Schaufelprofils
c nakli......Liste mit den Knoten des Schaufelprofils
c an_rok.....Anzahl der Knoten des oberen Randes (Eintritt)
c rokli......Liste mit den Knoten des oberen Randes
c an_ruk.....Anzahl der Knoten des unteren Randes (Austritt)
c rukli......Liste mit den Knoten des unteren Randes
c an_rlk.....Anzahl der Knoten des linken Randes
c rlkli......Liste mit den Knoten des linken Randes
c an_rrk.....Anzahl der Knoten des rechten Randes
c rrkli......Liste mit den Knoten des rechten Randes
c
      implicit none
c
      integer anz_rk,rkli(anz_rk),ape1,ape2,ape3,
     # ag,ase(ag,2),ap,zuli(ap),an_nak,
     # nakli(ap),an_rok,rokli(ap),an_ruk,rukli(ap),an_rlk,rlkli(ap),
     # an_rrk,rrkli(ap),DIM,i,j,por,mark,pu
c
      parameter (DIM=65535)
c
      real yu,yo,xr(ap),yr(ap),xpel1(ape1),ypel1(ape1),xper1(ape1),
     # yper1(ape1),xpel2(ape2),ypel2(ape2),xper2(ape2),yper2(ape2),
     # xpel3(ape3),ypel3(ape3),xper3(ape3),yper3(ape3),xrk(DIM),
     # yrk(DIM),KEA,ABST
c
      if (DIM.lt.ap) then
       print*, 'ZUORD_RAND: DIM zu klein'
       return
      endif
c NACA
      por=0
      do i=1,4
       por=por+ase(i,1)*ase(i,2)
      end do
      mark=por-ase(2,2)-ase(3,2)-ase(4,2)
c
      an_nak=0
      do i=1,anz_rk
       if (rkli(i).le.mark) then
        an_nak=an_nak+1
        nakli(an_nak)=rkli(i)
       endif
      end do
c Weise Randknotennummern 2D-Koordinaten zu
      do i=1,anz_rk
       do pu=1,ap
        if (rkli(i).eq.zuli(pu)) goto 10
       end do
       print*, 'ZUORD_RAND: Randknoten in Zuordungsliste nicht gefunden'
10     continue
       xrk(i)=xr(pu)
       yrk(i)=yr(pu)
      end do
c
c Rand unten
      an_ruk=0
      do i=1,anz_rk
       ABST=abs(yrk(i)-yu)
       if (ABST.lt.(KEA*0.9)) then 
        an_ruk=an_ruk+1
        rukli(an_ruk)=rkli(i)
       endif
      end do
c Rand oben
      an_rok=0
      do i=1,anz_rk
       ABST=abs(yrk(i)-yo)
       if (ABST.lt.(KEA*0.9)) then 
        an_rok=an_rok+1
        rokli(an_rok)=rkli(i)
       endif
      end do
c Rand links,rechts
      an_rlk=0
      an_rrk=0
      do i=1,anz_rk
       do j=2,ape1,1
        ABST=((xrk(i)-xpel1(j))**2+(yrk(i)-ypel1(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rlk=an_rlk+1 
         rlkli(an_rlk)=rkli(i)
        endif
        ABST=((xrk(i)-xper1(j))**2+(yrk(i)-yper1(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rrk=an_rrk+1 
         rrkli(an_rrk)=rkli(i)
        endif
       end do
c
       do j=1,ape2,1
        ABST=((xrk(i)-xpel2(j))**2+(yrk(i)-ypel2(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rlk=an_rlk+1 
         rlkli(an_rlk)=rkli(i)
        endif
        ABST=((xrk(i)-xper2(j))**2+(yrk(i)-yper2(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rrk=an_rrk+1 
         rrkli(an_rrk)=rkli(i)
        endif
       end do
c
       do j=2,ape3,1
        ABST=((xrk(i)-xpel3(j))**2+(yrk(i)-ypel3(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rlk=an_rlk+1 
         rlkli(an_rlk)=rkli(i)
        endif
        ABST=((xrk(i)-xper3(j))**2+(yrk(i)-yper3(j))**2)**0.5
        if (ABST.lt.(KEA*0.9)) then 
         an_rrk=an_rrk+1 
         rrkli(an_rrk)=rkli(i)
        endif
       end do
c
      end do
c         
      end
c ----------------------------------------------------------------------
c RB2D_WAND
c ----------------------------------------------------------------------
      subroutine RB2D_WAND(anz_re,reli,anz_el,ela,elb,elc,eld,an_nak,
     # nakli,seed,wrba,wrbb,wrbe,an_wrb)
c
c Erzeugt 2D-Wandrandbedingungen in einem Radialschnitt
c
c EINGABE
c anz_re.....Anzahl der Randelemente
c reli.......Liste mit den Randelementen
c ela..eld...Knoten der einzelnen Elemente
c an_nak.....Anzahl der Knoten des Schaufelprofils
c nakli......Liste mit den Knoten des Schaufelprofils
c seed.......max. Anzahl der Knoten
c
c AUSGABE
c wrba..wrbb.Knoten der einzelnen Wandrandbedingungen
c wrbe.......Liste mit den Elementen der Wandrandbedingungen
c an_wrb.....Anzahl der Wandrandbedingungen
c
      implicit none
c
      integer anz_re,reli(anz_re),anz_el,ela(anz_el),elb(anz_el),
     # elc(anz_el),eld(anz_el),an_nak,nakli(an_nak),seed,wrba(seed),
     # wrbb(seed),wrbe(seed),an_wrb,el1(4),tref,trefli(4),a,b,c
c
      do a=1,anz_re,1
       el1(1)=ela(reli(a))
       el1(2)=elb(reli(a))
       el1(3)=elc(reli(a))
       el1(4)=eld(reli(a))
       tref=0
       do c=1,4,1
        do b=1,an_nak,1 
         if (el1(c).eq.nakli(b)) then 
          tref=tref+1
          trefli(tref)=nakli(b)
          goto 2
         endif
        end do
2       continue
        if (tref.eq.2) then
         an_wrb=an_wrb+1
         wrba(an_wrb)=trefli(1)
         wrbb(an_wrb)=trefli(2)
         wrbe(an_wrb)=reli(a) 
         goto 1
        elseif (tref.gt.2) then
         print*, 'RB2D_WAND: Fehler beim Erzeugen der Randbedingungen'
        endif
       end do
1      continue  
      end do
c
      end 
c ----------------------------------------------------------------------
c RB2D_KBIL
c ----------------------------------------------------------------------
      subroutine RB2D_KBIL(anz_re,reli,anz_el,ela,elb,elc,eld,an_nak,
     # nakli,wert,seed,kbia,kbib,kbie,kbiw,an_kbi)
c
c Erzeugt 2D-Kraft- und Bilanzflaechen in einem Radialschnitt
c
c EINGABE
c anz_re.....Anzahl der Randelemente
c reli.......Liste mit den Randelementen
c ela..eld...Knoten der einzelnen Elemente
c an_nak.....Anzahl der Knoten des Randes
c nakli......Liste mit den Knoten des Randes
c wert.......Nummer fast Elementflaechen zu einer Gruppe zusammen
c seed.......max. Anzahl der Knoten
c
c AUSGABE
c kbia..kbib.Knoten der einzelnen Kraft- und Bilanzflaechen
c kbie.......Liste mit den Elementen der Kraft- und Bilanzflaechen
c an_kbi.....Anzahl der Kraft- und Bilanzflaechen
c
c
      implicit none
c
      integer anz_re,reli(anz_re),anz_el,ela(anz_el),elb(anz_el),
     # elc(anz_el),eld(anz_el),an_nak,nakli(an_nak),wert,seed,
     # kbia(seed),kbib(seed),kbie(seed),kbiw(seed),an_kbi,el1(4),tref,
     # trefli(4),a,b,c
c
      do a=1,anz_re,1
       el1(1)=ela(reli(a))
       el1(2)=elb(reli(a))
       el1(3)=elc(reli(a))
       el1(4)=eld(reli(a))
       tref=0
       do c=1,4,1
        do b=1,an_nak,1 
         if (el1(c).eq.nakli(b)) then 
          tref=tref+1
          trefli(tref)=nakli(b)
          goto 2
         endif
        end do
2       continue
        if (tref.eq.2) then
         an_kbi=an_kbi+1
         kbia(an_kbi)=trefli(1)
         kbib(an_kbi)=trefli(2)
         kbie(an_kbi)=reli(a) 
         kbiw(an_kbi)=wert 
         goto 1
        elseif (tref.gt.2) then
         print*, 'RB2D_WAND: Fehler beim Erzeugen der Randbedingungen'
        endif
       end do
1      continue  
      end do
c
      end 
c ----------------------------------------------------------------------
c RB2D_MARK
c ----------------------------------------------------------------------
      subroutine RB2D_MARK(an_rk,rkli,wert,seed,kman,kmaw,an_kma)
c
c Erzeugt Knotenmarkierungen fuer Hilfsprogramme im Radialaschnitt
c
c EINGABE
c an_rk.....Anzahl der Knoten des Randes
c rkli......Liste mit den Knoten des Randes
c wert......Nummer fast Knoten zu einer Gruppe zusammen
c seed......max. Anzahl der Knoten
c
c AUSGABE
c kman.....Liste mit den Knoten die markiert werden
c kmaw.....Liste mit den Werten der Knoten
c an_kma...Anzahl Knoten die markiert werden
c
      implicit none
c
      integer an_rk,rkli(an_rk),wert,seed,kman(seed),kmaw(seed),an_kma,
     # a
c
      do a=1,an_rk,1
       an_kma=an_kma+1
       kman(an_kma)=rkli(a)
       kmaw(an_kma)=wert
      end do 
c
      end
c ----------------------------------------------------------------------
c RB2D_KRB
c ----------------------------------------------------------------------
      subroutine RB2D_KRB(an_rk,rkli,seed,krbn,an_krb)
c
c Erzeugt Knotenrandbedingungen im Radialschnitt
c
c EINGABE
c an_rk.....Anzahl der Knoten des Randes
c rkli......Liste mit den Knoten des Randes
c seed......max. Anzahl der Knoten
c
c AUSGABE
c krbn.....Liste mit den Knoten die eine Randbedingung erhalten
c an_krb...Anzahl der Knoten die eine Randbedingung erhalten
c
      implicit none
c
      integer an_rk,rkli(an_rk),seed,krbn(seed),an_krb,a 
c
      do a=1,an_rk,1
       an_krb=an_krb+1
       krbn(an_krb)=rkli(a)
      end do
c
      end
c ----------------------------------------------------------------------
c RB3D_KRB
c ----------------------------------------------------------------------
      subroutine RB3D_KRB(an_krb,krbn,an_kno,bi,krb1,an3_kr)
c
c Erzeugt 3D-Knotenrandbedingungen aus den 2D-Knotenrandbed. 
c
c EINGABE
c an_krb...Anzahl der Knoten die eine 2D-Knotenrandbedingung haben
c krbn.....Liste mit den Knoten die eine 2D-Knotenrandbedingung haben
c an_kno...Anzahl der Knoten im Radialschnitt
c bi.......Anzahl der Radialschnitte
c AUSGABE
c krb1.....Liste mit den Knoten die eine Knotenrandbedingung erhalten
c an3_kr...Anzahl der Knoten die eine Knotenrandbedingung erhalten
c
      implicit none
c
      integer an_krb,krbn(an_krb),an_kno,bi,krb1(an_krb*bi),an3_kr,i,j
c
      an3_kr=0
      do i=1,bi,1
       do j=1,an_krb,1
        an3_kr=an3_kr+1
        krb1(an3_kr)=krbn(j)+an_kno*(i-1) 
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c RB3D_MARK
c ----------------------------------------------------------------------
      subroutine RB3D_MARK(an_kma,kman,kmaw,an_kno,bi,kma1,kma2,an3_km)
c
c Erzeugt 3D-Knotenmarkierungen aus den 2D-Knotenmarkierungen 
c
c EINGABE
c an_kma...Anzahl der Knoten die eine 2D-Knotenmarkierung haben
c kman.....Liste mit den Knoten die eine 2D-Knotenmarkierung haben
c an_kno...Anzahl der Knoten im Radialschnitt
c bi.......Anzahl der Radialschnitte
c AUSGABE
c kma1.....Liste mit den Knoten die eine Knotenmarkierung erhalten
c kma2.....Liste mit den Werten der Knoten
c an3_km...Anzahl der Knoten die eine Knotenmarkierung erhalten
c
c
      implicit none
c
      integer an_kma,kman(an_kma),kmaw(an_kma),an_kno,bi,
     # kma1(an_kma*bi),kma2(an_kma*bi),an3_km,i,j
c
      an3_km=0
      do i=1,bi,1
       do j=1,an_kma,1
        an3_km=an3_km+1
        kma1(an3_km)=kman(j)+an_kno*(i-1)
        kma2(an3_km)=kmaw(j)
       end do
      end do
c
      end
c ----------------------------------------------------------------------
c RB3D_WAND
c ----------------------------------------------------------------------
      subroutine RB3D_WAND(an_wrb,wrba,wrbb,wrbe,a_2del,a_3del,el1,el2,
     # el3,el4,el5,el6,el7,el8,an_kno,bi,wrb1,wrb2,wrb3,wrb4,wrb5,
     # an3_wr)
c
c Erzeugt 3D-Wandrandbedingungen aus 2D-Wandrandbed.
c
c EINGABE
c an_wrb.....Anzahl der Wandrandbedingungen
c wrba..wrbb.Knoten der einzelnen Wandrandbedingungen
c wrbe.......Liste mit den Elementen der Wandrandbedingungen
c a_2del.....Anzahl der 2D-Elemente
c a_3del.....Anzahl der 3D-Elemente
c el1..el8...Knoten der 3D-Elemente
c an_kno.....Anzahl der Knoten im Radialschnitt
c bi.........Anzahl der Radialschnitte
c
c AUSGABE
c wrb1..wrb4.Knoten der einzelnen Wandrandbedingungen
c wrb5.......Liste mit den Elementen der Wandrandbedingungen
c an3_wr.....Anzahl der 3D-Wandrandbedingungen
c
      implicit none
c
      integer  an_wrb,wrba(an_wrb),wrbb(an_wrb),wrbe(an_wrb),a_2del,
     # a_3del,el1(a_3del),el2(a_3del),el3(a_3del),el4(a_3del),
     # el5(a_3del),el6(a_3del),el7(a_3del),el8(a_3del),
     # an_kno,bi,wrb1(an_wrb*(bi-1)),wrb2(an_wrb*(bi-1)),
     # wrb3(an_wrb*(bi-1)),wrb4(an_wrb*(bi-1)),wrb5(an_wrb*(bi-1)),
     # an3_wr,i,j
c 2D-Wandrandbedingungen in 3D
      an3_wr=0
      do i=1,bi-1,1
       do j=1,an_wrb,1     
        an3_wr=an3_wr+1 
        wrb1(an3_wr)=wrba(j)+an_kno*(i-1)
        wrb2(an3_wr)=wrbb(j)+an_kno*(i-1)
        wrb3(an3_wr)=wrbb(j)+an_kno*(i)
        wrb4(an3_wr)=wrba(j)+an_kno*(i)
        wrb5(an3_wr)=wrbe(j)+a_2del*(i-1)
       end do
      end do
c Nabe
      do i=1,a_2del,1
       an3_wr=an3_wr+1 
       wrb1(an3_wr)=el1(i)
       wrb2(an3_wr)=el2(i)
       wrb3(an3_wr)=el3(i)
       wrb4(an3_wr)=el4(i)
       wrb5(an3_wr)=i
      end do 
c Kranz
      do i=(a_3del-a_2del+1),a_3del,1
       an3_wr=an3_wr+1 
       wrb1(an3_wr)=el5(i)
       wrb2(an3_wr)=el6(i)
       wrb3(an3_wr)=el7(i)
       wrb4(an3_wr)=el8(i)
       wrb5(an3_wr)=i
      end do 
c
      end
c ----------------------------------------------------------------------
c RB3D_KBIL
c ----------------------------------------------------------------------
      subroutine RB3D_KBIL(an_kbi,kbia,kbib,kbie,kbiw,a_2del,an_kno,bi,
     # kbi1,kbi2,kbi3,kbi4,kbi5,kbi6,an3_kb)
c
c Erzeugt 3D-Wandrandbedingungen aus 2D-Wandrandbed.
c
c EINGABE
c an_kbi.....Anzahl der Kraft- und Bilanzflaechen
c kbia..kbib.Knoten der einzelnen Kraft- und Bilanzflaechen
c kbie.......Liste mit den Elementen der Kraft- und Bilanzflaechen
c a_2del.....Anzahl der 2D-Elemente
c an_kno.....Anzahl der Knoten im Radialschnitt
c bi.........Anzahl der Radialschnitte
c
c AUSGABE
c kbi1..kbi4.Knoten der einzelnen Kraft- Bilanzflaechen
c kbi5.......Liste mit den Elementen der Kraft- Bilanzflaechen
c kbi6.......Nummer, die die Elementflaechen zu einer Gruppe zusammenf.
c an3_kb.....Anzahl der 3D-Kraft- Bilanzflaechen
c
c
      implicit none
c 
      integer an_kbi,kbia(an_kbi),kbib(an_kbi),kbie(an_kbi),
     # kbiw(an_kbi),a_2del,an_kno,bi,kbi1(an_kbi*(bi-1)),
     # kbi2(an_kbi*(bi-1)),kbi3(an_kbi*(bi-1)),kbi4(an_kbi*(bi-1)),
     # kbi5(an_kbi*(bi-1)),kbi6(an_kbi*(bi-1)),an3_kb,i,j
c
      an3_kb=0
      do i=1,bi-1,1
       do j=1,an_kbi,1     
        an3_kb=an3_kb+1 
        kbi1(an3_kb)=kbia(j)+an_kno*(i-1)
        kbi2(an3_kb)=kbib(j)+an_kno*(i-1)
        kbi3(an3_kb)=kbib(j)+an_kno*(i)
        kbi4(an3_kb)=kbia(j)+an_kno*(i)
        kbi5(an3_kb)=kbie(j)+a_2del*(i-1)
        kbi6(an3_kb)=kbiw(j)
       end do
      end do
c
      end     
c ----------------------------------------------------------------------
c WERT_KRB
c ----------------------------------------------------------------------
      subroutine WERT_KRB(an_krb,krbn,v_max,r_nab,r_kra,alpha,seed,zuli,
     # radius,xk,yk,fr,fp,fz,lmod,krbx,krby,krbz,krbk,krbe,an3_kw)
c
c Berechnet die Vektoren fuer die Knotenrandbedingungen (Eintrittsrb.)
c
c EINGABE
c an_krb.....Anzahl der Knoten die eine 2D-Knotenrandbedingung haben
c krbn.......Liste mit den Knoten die eine 2D-Knotenrandbedingung haben
c v_max......Betrag der max. Eintrittsgeschwindigkeit
c r_nab......Radius Nabe
c r_kra......Radius Kranz
c alpha......Eintrittswinkel
c seed.......Anzahl der Punkte
c zuli.......Zuordnungsliste Punkte-Knoten
c radius.....Radius des Radialschnittes
c xk,yk......2D-Koordinaten der Punkte im r-phi-Koord.system
c fr,fp,fz...Normierungsfaktoren
c lmod.......Modellaenge fuer das k-eps-Modell
c
c AUSGABE
c krbx..krbz.Komponenten der Eintrittsvektoren
c krbk.......k-Wert
c krbe.......eps-Wert
c an3_kw.....Anzahl der Knotenrandbedingung 
c
      implicit none
c
      integer an_krb,krbn(an_krb),seed,zuli(seed),an3_kw,i,kno,j
c
      real v_max,v_betr,r_nab,r_kra,r_null,alpha,radius,xk(seed),
     # yk(seed),fr,fp,fz,lmod,
     # krbx(an_krb),krby(an_krb),krbz(an_krb),krbk(an_krb),krbe(an_krb),
     # pp(2),pz(2),kax(2),kay(2),kaz(2)

      real v_max_leit, v_u_leit, v_u_lauf, v_u_lauf_rel
      real u_temp, alpha_0, phi
      real r_leit, h_leit, v_betr_leit
      real g

c Variablenuebergabe aus Geometrieberechnung
      real r_mitte,t_grad

      real PI, Q, H, N, ETA, VAUBEZ
      real d2, di_da, nlschaufel, n_z2, B0, D1, D2Z, z_axe_la

      common /absol/ Q, H, N, ETA, VAUBEZ
      common /zwis/  d2, di_da, nlschaufel, n_z2, B0, D1, D2Z,
     .               z_axe_la


      PI = 3.14159
      g = 9.81

c Testwerte fuer Leitradgeometrie
      r_leit=d2*1.5
      h_leit=d2/4

      alpha_0=atan(N*Q/(60*g*H*h_leit))

c Umfangsgeschwindigkeit

      u_temp=PI*N*radius/60/1000


c
c Geometrie an der Laufschaufel
      r_kra=d2/2
      r_nab=r_kra*di_da  
    
      r_null=(r_kra-r_nab)/2
      r_mitte=(r_kra+r_nab)/2
      
      t_grad=0.15


c maximale Geschwindigkeit am Leitrad

      v_max_leit=Q*(t_grad+1)/(2*PI*r_leit*h_leit)

      v_betr_leit=v_max_leit*(abs(1-abs(radius/1000-r_nab-r_null)
     #/(1.01*r_null)))**t_grad
      
      v_u_leit=v_betr_leit/tan(alpha_0)
      v_u_lauf=v_u_leit*r_leit/(radius/1000)
      v_u_lauf_rel=v_u_lauf-u_temp


c
c maximale Geschwindigkeit aus Q und Geometrie

      v_max=Q*(t_grad+1)/(4*PI*r_mitte*r_null)

c
c Turbulenzprofil
      
      v_betr=v_max*(abs(1-abs(radius/1000-r_nab-r_null)
     #/(1.01*r_null)))**t_grad

c




      do i=1,an_krb
c Vektoranfangs- und Vektorendpunkt 
       do kno=1,seed
        if (krbn(i).eq.zuli(kno)) goto 1
       end do
1      continue
       pp(1)=xk(kno)
      
       pp(2)=pp(1)+v_u_lauf_rel
       
c Ruecktransformation
              
        kax(1)=(radius/fr)*cos(pp(1)/fp)
        kay(1)=(radius/fr)*sin(pp(1)/fp)
       phi=atan(kay(1)/kax(1))
       if (kax(1).le.0) phi=phi+PI
       
c
       an3_kw=an3_kw+1
       krbx(i)=-v_u_lauf_rel*sin(phi)
       krby(i)=v_u_lauf_rel*cos(phi)
       krbz(i)=-v_betr
       krbk(i)=0.00375*(abs(krbz(i)))**2
       krbe(i)=0.00094*(abs(krbz(i)))**3/lmod
      end do
c
      end
c ----------------------------------------------------------------------
c CRE_3DRB
c ----------------------------------------------------------------------
      subroutine CRE_3DRB(an3_kr,krb1,an3_kw,krbx,krby,krbz,krbk,
     # krbe,an3_wr,wrb1,wrb2,wrb3,wrb4,wrb5,an3_kb,kbi1,kbi2,kbi3,
     # kbi4,kbi5,kbi6,an3_km,kma1,kma2)
c
c Erstellt ein 3D-Randbedingungsfile
c
c EINGABE
c an3_kr......Anzahl der Knoten die eine Knotenrandbedingung erhalten
c krb1........Liste mit den Knoten die eine Knotenrandbedingung erhalten
c an3_kw......Anzahl der Knotenrandbedingung 
c krbx..krbz..Komponenten der Eintrittsvektoren
c krbk........k-Wert
c krbe........eps-Wert
c an3_wr......Anzahl der 3D-Wandrandbedingungen
c wrb1..wrb4..Knoten der einzelnen Wandrandbedingungen
c wrb5........Liste mit den Elementen der Wandrandbedingungen
c an3_kb......Anzahl der 3D-Kraft- Bilanzflaechen
c kbi1..kbi4..Knoten der einzelnen Kraft- Bilanzflaechen
c kbi5........Liste mit den Elementen der Kraft- Bilanzflaechen
c kbi6........Nummer, die die Elementflaechen zu einer Gruppe zusammenf.
c an3_km......Anzahl der Knoten die eine Knotenmarkierung erhalten
c kma1........Liste mit den Knoten die eine Knotenmarkierung erhalten
c kma2........Liste mit den Werten der Knoten
c
      implicit none
c 
      integer an3_kr,krb1(an3_kr),an3_kw,an3_wr,wrb1(an3_wr),
     # wrb2(an3_wr),wrb3(an3_wr),wrb4(an3_wr),wrb5(an3_wr),an3_kb,
     # kbi1(an3_kb),kbi2(an3_kb),kbi3(an3_kb),kbi4(an3_kb),kbi5(an3_kb),
     # kbi6(an3_kb),an3_km,kma1(an3_km),kma2(an3_km),i
c
      real krbx(an3_kw),krby(an3_kw),krbz(an3_kw),krbk(an3_kw),
     # krbe(an3_kw)
c

      character*200 datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
      common /ver2/ datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
   10 FORMAT(4X,I8,4X,I8)



c
      print*, 'CRE_3DRB'
      print*, an3_kr,' Knotenrandbedingungen'
      print*, an3_wr,' Wandrandbedingungen'
      print*, an3_kb,' Kraft- und Bilanzflaechen'
      print*, an3_km,' Knotenmarkierungen'
c
      open(21,file=datei_kart3d_rb)
c
      do i=1,10,1
       write(21,*) 'C'
      end do
c
      write(21,*) (5*an3_kr),an3_wr,' 0 0 0 0 ',an3_kb,an3_km
c
      if (an3_kr.ne.an3_kw) then
       print*, 'CRE_3DRB: Fehler beim Schreiben der Knotenrandbed.'
       return
      elseif (an3_kr.ne.0) then 
       do i=1,an3_kr,1 
        write(21,*) krb1(i),' 1 ',krbx(i)
        write(21,*) krb1(i),' 2 ',krby(i)
        write(21,*) krb1(i),' 3 ',krbz(i)
        write(21,*) krb1(i),' 4 ',krbk(i)
        write(21,*) krb1(i),' 5 ',krbe(i)
       end do
      endif
c
      if (an3_wr.ne.0) then
       do i=1,an3_wr,1
        write(21,*) wrb1(i),wrb2(i),wrb3(i),wrb4(i),' 0 0 0 ',wrb5(i)
       end do
      endif
c
      if (an3_kb.ne.0) then
       do i=1,an3_kb,1
        write(21,*) kbi1(i),kbi2(i),kbi3(i),kbi4(i),kbi5(i),kbi6(i)
       end do
      endif
c
      if (an3_km.ne.0) then
       do i=1,an3_km,1
        write(21,*) kma1(i),kma2(i)
       end do
      end if
      close(21)
c
      end      
c ----------------------------------------------------------------------
c MOVR_KNOT
c ----------------------------------------------------------------------
      subroutine MOVR_KNOT(seed,xk,yk,zk,radius,r_nab,r_kra,apv,rpvN,
     # rpvK,zpv)
c
c Verschiebt Knoten in radialer Richtung nach der durch die Stuetzpunkte
c rpvN,rpvK,zpv gegebene Kontur
c
c EINGABE
c seed.......Anzahl der Gitterpunkte 
c xk,yk,zk...Kart.Koordinaten der Gitterpunkte
c radius.....Radius des aktuellen Schnittes
c r_nab......Radius der Nabe
c r_kra......Radius des Kranzes
c apv........Anzahl Stuetzpunkte der Radialkontur
c rpvN,zpv...Koord. der Stuetzpunkte der Radialkontur an der Nabe
c rpvK,zpv...Koord. derStuetzpunkte der Radialkontur am Kranz
c
c AUSGABE
c xk,yk,zk...Kart.Koordinaten der Gitterpunkte
c
      implicit none
c
      integer DIM1,DIM2,seed,apv,i
      parameter(DIM1=100,DIM2=65535)
c     
      real xk(seed),yk(seed),zk(seed),rz(DIM2),pz(DIM2),radius,
     # r_nab,r_kra,rpvN(apv),rpvK(apv),zpv(apv),akoN(DIM1),bkoN(DIM1),
     # akoK(DIM1),bkoK(DIM1),SP1DVAL,rvN,rvK,gewN,gewK
c
      if (apv.gt.DIM1) then
       print*, 'MOVR_KNOT: DIM1 zu klein'
       return
      elseif (seed.gt.DIM2) then
       print*, 'MOVR_KNOT: DIM2 zu klein'
       return
      endif
c Koeff. 1-D Spline
      call SP1DKO(apv,zpv,rpvN,akoN,bkoN)
      call SP1DKO(apv,zpv,rpvK,akoK,bkoK)
c Trafo ins Zylinderkoord.system
      call RP_TRAFO(seed,xk,yk,rz,pz)
c Verschiebe Gitterpunkte in radialer Richtung
      gewN=1-(radius-r_nab)/(r_kra-r_nab)
      gewK=(radius-r_nab)/(r_kra-r_nab)
      do i=1,seed
       rvN=SP1DVAL(zk(i),apv,zpv,akoN,bkoN)-r_nab
       rvK=SP1DVAL(zk(i),apv,zpv,akoK,bkoK)-r_kra
       rz(i)=rz(i)+rvN*gewN+rvK*gewK
      end do
c Ruecktrafo ins kart. Koord.system
      call RP_RTRAFO(seed,rz,pz,xk,yk)
c
      end
c ----------------------------------------------------------------------
c RP_TRAFO
c ----------------------------------------------------------------------
      subroutine RP_TRAFO(seed,xk,yk,rz,pz)
c
c Transformiert die Koord. aus dem kart. ins zyl. Koordinatensystem
c
c EINGABE
c seed.......Anzahl der Punkte 
c xk,yk......Kart. Koordinaten der Punkte
c
c AUSGABE
c rz,pz......Zyl. Koordinaten der Punkte
c 
c
      implicit none
c
      integer seed,i
c
      real xk(seed),yk(seed),rz(seed),pz(seed)
c
      do i=1,seed
       rz(i)=sqrt(xk(i)**2+yk(i)**2)
       pz(i)=atan2(yk(i),xk(i))
      end do
c
      end
c ----------------------------------------------------------------------
c RP_RTRAFO
c ----------------------------------------------------------------------
      subroutine RP_RTRAFO(seed,rz,pz,xk,yk)
c
c Transformiert die Koord. aus dem zyl. ins kart. Koord.
c
c EINGABE
c seed.......Anzahl der Punkte 
c rz,pz......Zyl. Koordinaten der Punkte
c
c AUSGABE
c xk,yk......Kart. Koordinaten der Punkte
c
      implicit none
c
      integer seed,i
c   
      real xk(seed),yk(seed),rz(seed),pz(seed)
c
      do i=1,seed
       xk(i)=rz(i)*cos(pz(i))
       yk(i)=rz(i)*sin(pz(i))
      end do
c
      end
c ----------------------------------------------------------------------
c SP1DKO
c ----------------------------------------------------------------------
      subroutine SP1DKO(n,xn,yn,a,b)
c
c Berechnet die Koeffizienten eines 1-D Splines
c
c EINGABE
c n.......Anzahl der Stuetzpunkte
c xn,yn...Koord. der Stuetzpunkte
c
c AUSGABE
c a,b.....Koeff. des Splines
c
      implicit none
c
      integer n,i
c
      real xn(n),yn(n),a(n-1),b(n-1)
c
      do i=1,n-1,1
       a(i)=yn(i) 
       b(i)=(yn(i+1)-yn(i))/(xn(i+1)-xn(i))
      end do
c
      end
c ----------------------------------------------------------------------
c SP1DVAL
c ----------------------------------------------------------------------
      real function SP1DVAL(x,n,xn,a,b)
c
c Gibt den zugehoerigen Funktionswert eines 1-D Splines zurueck
c
c EINGABE
c x........x-Koordinate, an der der Funktionswert berechnet werden soll
c n........Anzahl der Stuetzpunkte
c xn.......x-Koordinate der Stuetzpunkte
c a,b......Koeff. des Splines
c
c AUSGABE
c SP1DVAL..Funktionswert zu der x-Koord.
c
      implicit none
c
      integer n,i,k,m
c
      real x,xn(n),a(n-1),b(n-1),xl
c
c Suche Intervall
      i=1
      k=n
1     m=(i+k)/2
      if (x.lt.xn(m)) then
       k=m
      else 
       i=m
      endif
      if (k.gt.i+1) goto 1
c
      xl=x-xn(i)
      SP1DVAL=b(i)*xl+a(i)
c
      end
c ----------------------------------------------------------------------
c INP_MOVR
c ----------------------------------------------------------------------
      subroutine INP_MOVR(mov_fi,DIM_PVR,rsn,rsk,zs,anz)
c
c     Liest die Daten fuer die Radialverschiebung ein
c
c     EINGABE 
c     mov_fi.....Filename mit den Daten
c     DIM_PVR....Vordimensionierung der Variablen
c
c     AUSGABE
c     rsn........Radialkomponente an der Nabe 
c     rsk........Radialkomponente am Kranz
c     zs.........z-Komponente
c     anz........Anzahl der einzulesenden Variablen
c
      implicit none
c 
      integer anz,DIM_PVR,i
c
      real rsn(DIM_PVR),rsk(DIM_PVR),zs(DIM_PVR)
c
      character*200 mov_fi
c
      open(31,file=mov_fi)
      read(31,*) anz
      if (anz.gt.DIM_PVR) then
       print*, 'INPUT_MOV: DIM_PVR zu klein'
       return
      endif
      do i=1,anz
       read(31,*) rsn(i),rsk(i),zs(i)
      end do
      close(31)
c
      end
c ----------------------------------------------------------------------
c T E S T R O U T I N E N
c ----------------------------------------------------------------------
c ----------------------------------------------------------------------
c AUSGABE
c ----------------------------------------------------------------------
      subroutine AUSGABE(name,a,x,y,z)
c
c EINGABE
c name .....Name der Ausgabedatei
c a ........Anzahl der auszugebenden Punkte
c
c AUSGABE
c x,y,z ....Koordinaten der Punkte
c
      implicit none
c
      integer a,i
      integer x(a),y(a),z(a)
      character*80 name
c
      open(12,file=name)
      do i=1,a
       write(12,'(3(F12.3))') x(i),y(i),z(i)
      end do
      close(12)
c
      end
c ----------------------------------------------------------------------
c AUSG_2D
c ----------------------------------------------------------------------
      subroutine AUSG_2D(name,a,x,y)
c
c EINGABE
c name......Name der Ausgabedatei
c a.........Anzahl der auszugebenden Punkte
c
c AUSGABE
c x,y ......Koordinaten der Punkte
c
      implicit none
c
      integer a,i
      real x(a),y(a)
      character*8 name
c
      open(11,file=name)
      do i=1,a
       write(11,'(2F12.3)') x(i),y(i)
      end do
      close(11)
c
      end
c ----------------------------------------------------------------------
c AUSG_2DEL
c ----------------------------------------------------------------------
      subroutine AUSG_2DEL(ap,xk,yk,ae,ela,elb,elc,eld,zuli,ar,reli,
     # file)
c
      implicit none
c
      integer ap,ae,ela(ae),elb(ae),elc(ae),eld(ae),zuli(ap),ar,
     # reli(ar),i,kno,kno1
c
      real xk(ap),yk(ap)
c
      character*4, file
c
      open(20,file=file)
      do i=1,ar
       do kno1=1,ap
        if (ela(reli(i)).eq.zuli(kno1)) goto 10
       end do
       print*, 'AUSG_2DEL:Knoten nicht gefunden'
10     continue
       write(20,*) xk(kno1),yk(kno1)
       do kno=1,ap
        if (elb(reli(i)).eq.zuli(kno)) goto 20
       end do
       print*, 'AUSG_2DEL:Knoten nicht gefunden'
20     continue
       write(20,*) xk(kno),yk(kno)
       do kno=1,ap
        if (elc(reli(i)).eq.zuli(kno)) goto 30
       end do
       print*, 'AUSG_2DEL:Knoten nicht gefunden'
30     continue
       write(20,*) xk(kno),yk(kno)
       do kno=1,ap
        if (eld(reli(i)).eq.zuli(kno)) goto 40
       end do
       print*, 'AUSG_2DEL:Knoten nicht gefunden'
40     continue
       write(20,*) xk(kno),yk(kno)
       write(20,*) xk(kno1),yk(kno1)
       write(20,*)
      end do
      close(20)
c
      end
c ----------------------------------------------------------------------
c AUSG_2DKN
c ----------------------------------------------------------------------
      subroutine AUSG_2DKN(ap,xk,yk,zuli,ark,rkli,file)
c
      implicit none
c
      integer ap,zuli(ap),ark,rkli(ark),i,kno
c
      real xk(ap),yk(ap)
c
      character*4, file
c
      open(21,file=file)
      do i=1,ark
       do kno=1,ap
        if (rkli(i).eq.zuli(kno)) goto 10
       end do
10     continue
       write(21,*) xk(kno),yk(kno)
      end do
      close(21)
c
      end
c ----------------------------------------------------------------------
c SOFTWARE AUS NUMERISCHE ALGORITHMEN, ENGELN-MUELLGES
c ----------------------------------------------------------------------
c ----------------------------------------------------------------------
c ISPLNP
c ----------------------------------------------------------------------
C[BA*)
C[KA{F 10}{Interpolierende Polynomsplines}
C[        {Interpolierende Polynomsplines zur Konstruktion glatter
C[         Kurven}*)
C[FE{F 10.1.2}
C[  {Berechnung der nichtparametrischen kubischen Splines}
C[  {Berechnung der nichtparametrischen kubischen Splines}*)
C[LE*)
C[BE*)
      SUBROUTINE ISPLNP (N,XN,FN,IB,ALPHA,BETA,B,C,D,DUMMY,IERR)
C[BA*)
C[IX{ISPLNP}*)
C[LE*)
C*********************************************************************
C                                                                    *
C  'ISPLNP' berechnet die Koeffizienten B(I), C(I), D(I), I=0(1)N-1, *
C  eines nichtparametrischen kubischen Interpolationssplines         *
C  fr verschiedene Randbedingungen.                                 *
C[BE*)
C  Die Art der Randbedingung ist mittels des Parameters IB vorzuge-  *
C  ben. Die Splinefunktion wird dargestellt in der Form:             *
C                                                                    *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +                 *
C                               + D(I)(X-XN(I))**3                   *
C                                                                    *
C  fr X Element von [XN(I),XN(I+1)], I=0(1)N-1.                     *
C                                                                    *
C                                                                    *
C  VORAUSSETZUNGEN:    1.         N > 2                              *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1             *
C                      3.     FN(0) = FN(N)  , fr IB = 4            *
C                                                                    *
C                                                                    *
C  EINGABEPARAMETER:                                                 *
C  =================                                                 *
C  N  :  Nummer des letzten Knotens                                  *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N                *
C  FN :  1-dim. Feld (0:N); FN(I) = Meáwert an der Stelle XN(I)      *
C                                                                    *
C  IB :  Art der Randbedingungen                                     *
C        IB = 1:  1. Randableitungen vorgegeben                      *
C        IB = 2:  2. Randableitungen vorgegeben                      *
C        IB = 3:  3. Randableitungen vorgegeben                      *
C        IB = 4:  periodische Splinefunktion                         *
C        IB = 5:  'not-a-knot' - Bedingung                           *
C                                                                    *
C  ALPHA :  IB. Randableitung bei X(0) ] nur fr IB = 1, 2, 3; ohne  *
C  BETA  :  IB. Randableitung bei X(N) ] Bedeutung fr IB = 4, 5     *
C                                                                    *
C  (Einen natrlichen Interpolationsspline erh„lt man mit IB = 2     *
C   und  ALPHA = BETA = 0.0)                                         *
C                                                                    *
C                                                                    *
C  HILFSFELDER:                                                      *
C  ============                                                      *
C  DUMMY :  1-dim. Feld (1:5*N+1)                                    *
C                                                                    *
C                                                                    *
C  AUSGABEPARAMETER:                                                 *
C  =================                                                 *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1            *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der            *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                     *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-          *
C                              speicher                              *
C  IERR :  Fehlerparameter                                           *
C          =  0 :  Alles klar                                        *
C          = -1 :  N < 4                                             *
C          = -2 :  IB < 1  oder  IB > 5                              *
C          = -3 :  Monotoniefehler der Sttzstellen XN(I),           *
C                  XN(I) > oder = XN(I+1), I=0(1)N-1                 *
C          = -4 :  FN(0) ungleich FN(N) (nur fr IB=4)               *
C          =  1 :  Abbruch in TRDSY, TRDIG oder CYTSY                *
C                                                                    *
C--------------------------------------------------------------------*
C                                                                    *
C  Ben”tigte Unterprogramme: ISPL1D, ISPL2D, ISPL3D, ISPLPE,         *
C                            ISPLNK, TRDSY, TRDSYS, CYTSY,           *
C                            CYTSYS, TRDIG                           *
C                                                                    *
C                                                                    *
C  Quellen : Engeln-Mllges, G.; Reutter, F., siehe [ENGE87].        *
C                                                                    *
C*********************************************************************
C                                                                    *
C  Autor     : Gnter Palm                                           *
C  Datum     : 15.04.1988                                            *
C  Quellcode : FORTRAN 77                                            *
C                                                                    *
C[BA*)
C*********************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 DUMMY(1:5*N+1)
C
C-----šberprfung der Vorbedingungen------------------------------
C
      IERR = -1
      IF (N .LT. 3) RETURN
      DO 10 I=0,N-1,1
        IF (XN(I) .GE. XN(I+1)) THEN
          IERR = -3
          RETURN
        ENDIF
   10 CONTINUE
C
C-----Berechnung der Splinekoeffizienten--------------------------
C
      IF (IB .EQ. 1) THEN
        CALL ISPL1D (N,XN,FN,ALPHA,BETA,1,B,C,D,DUMMY(1),DUMMY(N+1),
     +               DUMMY(2*N),DUMMY(3*N-1),IERR)
      ELSEIF (IB .EQ. 2) THEN
        CALL ISPL2D (N,XN,FN,ALPHA,BETA,1,B,C,D,DUMMY(1),DUMMY(N+1),
     +               DUMMY(2*N),DUMMY(3*N-1),IERR)
      ELSEIF (IB .EQ. 3) THEN
        CALL ISPL3D (N,XN,FN,ALPHA,BETA,B,C,D,DUMMY(1),DUMMY(N+1),
     +               IERR)
      ELSEIF (IB .EQ. 4) THEN
        CALL ISPLPE (N,XN,FN,1,B,C,D,DUMMY(1),DUMMY(N+2),
     +               DUMMY(2*N+2),DUMMY(3*N+2),DUMMY(4*N+2),IERR)
      ELSEIF (IB .EQ. 5) THEN
        CALL ISPLNK (N,XN,FN,B,C,D,DUMMY(1),DUMMY(N+1),DUMMY(2*N),
     +               IERR)
      ELSE
        IERR = -2
      ENDIF
      RETURN
      END
c ----------------------------------------------------------------------
c TRDIG
c ----------------------------------------------------------------------
C[BA*)
C[LE*)
C[LE*)
C[LE*)
C[FE{F 4.10.1}
C[  {Systeme mit tridiagonaler Matrix}
C[  {Systeme mit tridiagonaler Matrix}*)
C[LE*)
C[BE*)
      SUBROUTINE TRDIG (N,DL,DM,DU,RS,X,MARK)
C[BA*)
C[IX{TRDIG}*)
C[LE*)
C*****************************************************************
C                                                                *
C     L”sung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit tridiagonaler, streng regul„rer Matrix A.              *
C[BE*)
C     Die Matrix A wird durch die drei 1-dim. Felder DL, DM und  *
C     DU beschrieben. Das Gleichungssystem hat die Form:         *
C                                                                *
C     DM(1) * X(1) + DU(1) * X(2)                      = RS(1)   *
C     DL(I) * X(I-1) + DM(I) * X(I) + DU(I) * X(I+1)   = RS(I)   *
C     DL(N) * X(N-1) + DM(N) * X(N)                    = RS(N)   *
C                                                                *
C     fr I=2(1),N-1.                                            *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DL   : 1-dim. Feld (1:N); untere Nebendiagonale            *
C            DL(2), DL(3), ... ,DL(N)                            *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1)                         *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DL   :)                                                    *
C     DM   :)                                                    *
C     DU   :) berschrieben mit Hilfsvektoren                    *
C     RS   :)                                                    *
C     X    : 1-dim. Feld (1:N), das die L”sung des Gleichungs-   *
C            systems enth„lt                                     *
C     MARK : Fehlerparameter                                     *
C            MARK= 1 : alles klar                                *
C            MARK= 0 : Matrix numerisch nicht streng regul„r     *
C            MARK=-1 : Voraussetzung N > 2 verletzt              *
C                                                                *
C     BEMERKUNG: Die Determinante von A kann nach dem Aufruf     *
C                im Falle MARK = 1 wie folgt berechnet werden:   *
C                   DET A = DM(1) * DM(2) * ... * DM(N)          *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Ben”tigte Unterprogramme: TRDIGP, TRDIGS, MACHPD              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Mllges                             *
C  Datum     : 02.05.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C[BA*)
C*****************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DL(1:N),DM(1:N),DU(1:N),RS(1:N),X(1:N)
      MARK = -1
      IF (N .LT. 3) RETURN
C
C  Zerlegung der Matrix A
C
      CALL TRDIGP (N,DL,DM,DU,MARK)
C
C  falls MARK = 1, wird die Vorw„rts- und Rckw„rtselimination
C  durchgefhrt
C
      IF (MARK .EQ. 1) THEN
         CALL TRDIGS (N,DL,DM,DU,RS,X)
      ENDIF
      RETURN
      END
C
C
C[BA*)
C[LE*)
C[BE*)
      SUBROUTINE TRDIGP (N,DL,DM,DU,MARK)
C[BA*)
C[IX{TRDIGP}*)
C[LE*)
C*****************************************************************
C                                                                *
C     Zerlegung einer tridiagonalen, streng regul„ren Matrix A,  *
C     die durch die drei 1-dim. Felder DL, DM und DU beschrieben *
C     wird, in ihre Faktoren  L * R  mit bidiagonaler unterer    *
C     Dreiecksmatrix L und normierter bidiagonalen oberen Drei-  *
C     ecksmatrix R.                                              *
C[BE*)
C     Die Form des Gleichungssystems kann aus der Beschreibung   *
C     der SUBROUTINE TRDIG entnommen werden.                     *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DL   : 1-dim. Feld (1:N); untere Nebendiagonale            *
C            DL(2), DL(3), ... , DL(N)                           *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1)                         *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DL   : untere Nebendiagonale bleibt erhalten, da sie der   *
C            Nebendiagonalen von L entspricht                    *
C     DM   :) berschrieben mit Hilfsfeldern, die die            *
C     DU   :) Zerlegungsmatrizen von A enthalten. Die Neben-     *
C             diagonale der normierten oberen tridiagonalen      *
C             Dreiecksmatrix R wird in DU abgespeichert, die     *
C             Diagonalelemente von L in DM                       *
C     MARK : Fehlerparameter                                     *
C            MARK= 1 : alles klar                                *
C            MARK= 0 : Matrix numerisch nicht streng regul„r     *
C            MARK=-1 : Voraussetzung N > 2 verletzt              *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Ben”tigte Unterprogramme: MACHPD                              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Mllges                             *
C  Datum     : 02.05.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C[BA*)
C*****************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DL(1:N),DM(1:N),DU(1:N)
C
C   šberprfung der Voraussetzung N > 2
C
      MARK = -1
      IF (N .LT. 3) RETURN
C
C   Berechnung der Maschinengenauigkeit
C
      FMACHP = 1.0D0
   10 FMACHP = 0.5D0 * FMACHP
      IF (MACHPD(1.0D0+FMACHP) .EQ. 1) GOTO 10
      FMACHP = FMACHP * 2.0D0
C
C   Festlegung der Schranke fr den relativen Fehler
C
      EPS = 4.0D0 * FMACHP
C
C   Abfrage auf strenge Regularit„t fr N=1
C
      ROW = DABS(DM(1)) + DABS(DU(1))
      IF (ROW .EQ. 0.0D0) THEN
         MARK = 0
         RETURN
      ENDIF
      D = 1.0D0/ROW
      IF (DABS(DM(1))*D .LE. EPS) THEN
         MARK = 0
         RETURN
      ENDIF
C
C   Durchfhrung der Zerlegung bei gleichzeitiger Abfrage
C   auf strenge Regularit„t der Matrix A
C
      DL(1) = 0.0D0
      DU(N) = 0.0D0
      DU(1) = DU(1)/DM(1)
      DO 20 I=2,N,1
         ROW = DABS(DL(I)) + DABS(DM(I)) + DABS(DU(I))
         IF (ROW .EQ. 0.0D0) THEN
            MARK = 0
            RETURN
         ENDIF
         D = 1.0D0/ROW
         DM(I) = DM(I) - DL(I) * DU(I-1)
         IF (DABS(DM(I))*D .LE. EPS) THEN
            MARK = 0
            RETURN
         ENDIF
         IF (I .LT. N) THEN
            DU(I) = DU(I)/DM(I)
         ENDIF
   20 CONTINUE
      MARK=1
      RETURN
      END
C
C
C[BA*)
C[LE*)
C[BE*)
      SUBROUTINE TRDIGS (N,DL,DM,DU,RS,X)
C[BA*)
C[IX{TRDIGS}*)
C[LE*)
C*****************************************************************
C                                                                *
C     L”sung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit tridiagonaler, streng regul„rer Matrix A, nachdem die  *
C     Zerlegungsmatrizen mit Hilfe der SUBROUTINE TRDIGP berech- *
C     net worden sind.                                           *
C[BE*)
C     Hier werden diese Zerlegungsmatrizen als                   *
C     Eingabematrizen verwendet und in den drei 1-dim. Feldern   *
C     DL, DM und DU abgespeichert.                               *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DL   : 1-dim. Feld (1:N); ) die Felder DL, DM und DU ent-  *
C     DM   : 1-dim. Feld (1:N); ) halten die Zerlegung der Matrix*
C     DU   : 1-dim. Feld (1:N); ) A gem„á der Ausgabe von TRDIGP *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     X    : 1-dim. Feld (1:N), das die L”sung des Gleichungs-   *
C            systems enth„lt                                     *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Ben”tigte Unterprogramme: keine                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Mllges                             *
C  Datum     : 02.05.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C[BA*)
C*****************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DL(1:N),DM(1:N),DU(1:N),RS(1:N),X(1:N)
C
C  Vorw„rtselimination
C
      RS(1) = RS(1)/DM(1)
      DO 10 I=2,N,1
         RS(I) = (RS(I) - DL(I) * RS(I-1)) / DM(I)
   10 CONTINUE
C
C  Rckw„rtselimination
C
      X(N) = RS(N)
      DO 20 I=N-1,1,-1
         X(I) = RS(I) - DU(I) * X(I+1)
   20 CONTINUE
      RETURN
      END
c ----------------------------------------------------------------------
c SUBROUTINE ISPLNK
c ----------------------------------------------------------------------
C[BA*)
C[LE*)
C[BE*)
      SUBROUTINE ISPLNK (N,XN,FN,B,C,D,H,DM,RS,IERR)
C[BA*)
C[IX{ISPLNK}*)
C[LE*)
C*******************************************************************
C                                                                  *
C  'ISPLNK' berechnet die Koeffizienten B(I), C(I), D(I),          *
C  I=0(1)N-1, eines kubischen Interpolationssplines mit            *
C  'not-a-knot'-Randbedingung.                                     *
C[BE*)
C                                                                  *
C  Die Splinefunktion wird dargestellt in der Form:                *
C                                                                  *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +               *
C                               + D(I)(X-XN(I))**3                 *
C                                                                  *
C  fr X Element von [XN(I),XN(I+1)], I=0(1)N-1.                   *
C                                                                  *
C                                                                  *
C  VORAUSSETZUNGEN:    1.         N > 2                            *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1           *
C                                                                  *
C                                                                  *
C  BEMERKUNG:  'ISPLNK' sollte nicht direkt, sondern ber das      *
C  ==========  Unterprogramm 'ISPLNP' aufgerufen werden.           *
C              'ISPLNP' berprft auch die Voraussetzungen.        *
C                                                                  *
C                                                                  *
C  EINGABEPARAMETER:                                               *
C  =================                                               *
C  N  :  Nummer des letzten Knotens                                *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N              *
C  FN :  1-dim. Feld (0:N); FN(I) = Meáwert an der Stelle XN(I)    *
C                                                                  *
C                                                                  *
C  HILFSFELDER:                                                    *
C  ============                                                    *
C  H  :   1-dim. Feld (0:N-1)                                      *
C                                                                  *
C  DM :]  1-dim. Felder (1:N-1)                                    *
C  RS :]                                                           *
C                                                                  *
C                                                                  *
C  AUSGABEPARAMETER:                                               *
C  =================                                               *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1          *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der          *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                   *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-        *
C                              speicher                            *
C  IERR :  Fehlerparameter                                         *
C          =  0 :  Alles klar                                      *
C          = -1 :  N < 3                                           *
C          =  1 :  Abbruch in TRDIG                                *
C                                                                  *
C------------------------------------------------------------------*
C                                                                  *
C  Ben”tigte Unterprogramme: TRDIG                                 *
C                                                                  *
C                                                                  *
C  Quellen : Engeln-Mllges, G.; Reutter, F., siehe [ENGE87].      *
C                                                                  *
C*******************************************************************
C                                                                  *
C  Autor     : Gnter Palm                                         *
C  Datum     : 15.04.1988                                          *
C  Quellcode : FORTRAN 77                                          *
C                                                                  *
C[BA*)
C*******************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 H(0:N-1), DM(1:N-1), RS(1:N-1)
C
C-----Berechnung der Hilfsgr”áen----------------------------------
C
      DO 10 I=0,N-1,1
        H(I) = XN(I+1) - XN(I)
   10 CONTINUE
C
C-----Berechnung der Matrixelemente (Ober-, Unter-, Hauptdiag.)---
C     und der rechten Seite des Gleichungssystems A*C=RS mit
C     tridiagonaler Matrix A
C
C     Ober- und Unterdiagonale
C
      D(1) = H(1) - H(0)
      B(2) = H(0)
      DO 20 I=2,N-3,1
        D(I)   = H(I)
        B(I+1) = H(I)
   20 CONTINUE
      D(N-2) = H(N-2)
      B(N-1) = H(N-2) - H(N-1)
C
C     Hauptdiagonale
C
      DM(1) = H(0) + 2.0D0*H(1)
      DO 30 I=2,N-2,1
        DM(I) = 2.0D0*(H(I-1) + H(I))
   30 CONTINUE
      DM(N-1) = 2.0D0*H(N-2) + H(N-1)
C
C     Rechte Seite
C
      DUMMY1 = (FN(2) - FN(1))/H(1)
      RS(1)  = 3.0D0*H(1)/(H(1)+H(0))*(DUMMY1 - (FN(1)-FN(0))/H(0))
      DO 40 I = 2,N-2,1
        DUMMY2 = (FN(I+1) - FN(I))/H(I)
        RS(I)  = 3.0D0*(DUMMY2 - DUMMY1)
        DUMMY1 = DUMMY2
   40 CONTINUE
      RS(N-1) = 3.0D0*H(N-2)/(H(N-2)+H(N-1))*((FN(N)-FN(N-1))/H(N-1)
     +          - DUMMY1)
C
C-----Berechnung der Koeffizienten C(1) bis C(N-1) durch----------
C     L”sen des Gleichungssystems
C
      CALL TRDIG (N-1,B(1),DM,D(1),RS,C(1),IFLAG)
      IF (IFLAG .NE. 1) THEN
        IF (IFLAG .EQ. 0) THEN
          IERR = 1
        ELSE
          IERR = -1
        ENDIF
      ENDIF
      IERR = 0
C
C-----Berechnung der restlichen Splinekoeffizienten---------------
C
      C(0) = C(1) + H(0)/H(1)*(C(1) - C(2))
      C(N) = C(N-1) + H(N-1)/H(N-2)*(C(N-1) - C(N-2))
C
      DO 50 I=0,N-1,1
        B(I) = (FN(I+1)-FN(I))/H(I) - H(I)/3.0D0*(C(I+1)+2.0D0*C(I))
        D(I) = (C(I+1)-C(I))/(3.0D0*H(I))
   50 CONTINUE
      RETURN
      END
c ----------------------------------------------------------------------
c SPVAL
c ----------------------------------------------------------------------
      DOUBLE PRECISION FUNCTION SPVAL (X,N,XN,A,B,C,D)
c
C*****************************************************************
C                                                                *
C  Das Function-Unterprogramm SPVAL berechnet den Funktionswert  *
C  einer kubischen Splinefunktion S.                             *
C  Diese hat die Darstellung                                     *
C                                                                *
C  S(X) = P(I)(X) = A(I) + B(I)*(X-XN(I)) + C(I)*(X-XN(I))**2 +  *
C                                         + D(I)*(X-XN(I))**3    *
C                                                                *
C  für X aus dem Intervall [XN(I),XN(I+1)], I=0(1)N-1.           *
C                                                                *
C  Für X < XN(0) wird das Randpolynom P(0) ausgewertet,          *
C  für X > XN(N) das Randpolynom P(N-1).                         *
C  Es wird keine Plausibilitätsüberprüfung der Eingabewerte      *
C  vorgenommen.                                                  *
C                                                                *
C                                                                *
C  EINGABEPARAMETER:                                             *
C  =================                                             *
C  X  :  Stelle, an der der Funktionswert berechnet werden soll  *
C  N  :  Nummer des letzten Knotens                              *
C  XN :  1-dim. Feld (0:N) mit den Knoten XN(I), I=0(1)N         *
C  A  :  ] 1-dim. Felder (0:N) mit den Splinekoeffizienten A(I), *
C  B  :  ] B(I), C(I), D(I) für I=0(1)N-1 und einem Hilfs-       *
C  C  :  ] speicher für I=N                                      *
C  D  :  ]                                                       *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: keine                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Günter Palm                                       *
C  Datum     : 01.06.1991                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C  Deklarationen
C
      DOUBLE PRECISION XN(0:N), A(0:N), B(0:N), C(0:N), D(0:N)
      SAVE I
C
C  Vorbesetzungen
C
      DATA I /0/
      IF (I .GE. N) I=0
C
C  Falls es sich um einen wiederholten Aufruf des Unterprogramms
C  handelt, wird die Schleife zur Bestimmung des Intervalls
C  [XN(I),XN(I+1)], in dem die Stelle X enthalten ist, nur dann
C  durchlaufen, wenn X nicht in demselben Intervall liegt wie
C  beim letzten Aufruf
C
      IF (X .LT. XN(I)  .OR.  X .GE. XN(I+1)) THEN
        I = 0
        K = N
   10   M = (I+K)/2
        IF (X .LT. XN(M)) THEN
          K = M
        ELSE
          I = M
        ENDIF
        IF (K .GT. I+1) GOTO 10
      ENDIF
C
C  Berechnung des Linearfaktors X-XN(I) für die Polynomauswertung
C
      XL = X - XN(I)
C
C  Berechnung des Funktionswertes durch Hornerklammerung
C
      SPVAL = ((D(I)*XL + C(I))*XL + B(I))*XL + A(I)
      RETURN
      END
C
C
C
C[BA*)
C[LE*)
C[BE*)
      SUBROUTINE ISPL3D (N,XN,FN,ALPHA,BETA,B,C,D,H,RS,IERR)
C[BA*)
C[IX{ISPL3D}*)
C[LE*)
C*******************************************************************
C                                                                  *
C  'ISPL3D' berechnet die Koeffizienten B(I), C(I), D(I),          *
C  I=0(1)N-1, eines kubischen Interpolationssplines mit vorgege-   *
C  bener 3. Randableitung.                                         *
C[BE*)
C                                                                  *
C  Die Splinefunktion wird dargestellt in der Form:                *
C                                                                  *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +               *
C                               + D(I)(X-XN(I))**3                 *
C                                                                  *
C  fr X Element von [XN(I),XN(I+1)], I=0(1)N-1.                   *
C                                                                  *
C                                                                  *
C  VORAUSSETZUNGEN:    1.         N > 2                            *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1           *
C                                                                  *
C                                                                  *
C  BEMERKUNG:  'ISPL3D' sollte nicht direkt, sondern ber das      *
C  ==========  Unterprogramm 'ISPLNP' aufgerufen werden.           *
C              'ISPLNP' berprft auch die Voraussetzungen.        *
C                                                                  *
C                                                                  *
C  EINGABEPARAMETER:                                               *
C  =================                                               *
C  N  :  Nummer des letzten Knotens                                *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N              *
C  FN :  1-dim. Feld (0:N); FN(I) = Meáwert an der Stelle XN(I)    *
C                                                                  *
C  ALPHA :  3. Randableitung bei XN(0)                             *
C  BETA  :  3. Randableitung bei XN(N)                             *
C                                                                  *
C                                                                  *
C  HILFSFELDER:                                                    *
C  ============                                                    *
C  H  :  1-dim. Feld (0:N-1)                                       *
C  RS :  1-dim. Feld (1:N-1)                                       *
C                                                                  *
C                                                                  *
C  AUSGABEPARAMETER:                                               *
C  =================                                               *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1          *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der          *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                   *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-        *
C                              speicher                            *
C  IERR :  Fehlerparameter                                         *
C          =  0 :  Alles klar                                      *
C          = -1 :  N < 3                                           *
C          =  1 :  Abbruch in TRDSY                                *
C                                                                  *
C------------------------------------------------------------------*
C                                                                  *
C  Ben”tigte Unterprogramme: TRDSY                                 *
C                                                                  *
C                                                                  *
C  Quellen : Engeln-Mllges, G.; Reutter, F., siehe [ENGE87].      *
C                                                                  *
C*******************************************************************
C                                                                  *
C  Autor     : Gnter Palm                                         *
C  Datum     : 15.04.1988                                          *
C  Quellcode : FORTRAN 77                                          *
C                                                                  *
C[BA*)
C*******************************************************************
C[BE*)
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 H(0:N-1), RS(1:N-1)
C
C-----Berechnung der Hilfsgr”áen----------------------------------
C
      DO 10 I=0,N-1,1
        H(I) = XN(I+1) - XN(I)
   10 CONTINUE
C
C-----Berechnung der Matrixelemente (Ober- und Hauptdiagonale)----
C     und der rechten Seite des Gleichungssystems A*C=RS mit
C     symmetrischer, tridiagonaler Matrix A
C
C     Oberdiagonale
C
      DO 20 I=1,N-2,1
        D(I) = H(I)
   20 CONTINUE
C
C     Hauptdiagonale
C
      B(1) = 3.0D0*H(0) + 2.0D0*H(1)
      DO 30 I=2,N-2,1
        B(I) = 2.0D0*(H(I-1)+H(I))
   30 CONTINUE
      B(N-1) = 2.0D0*H(N-2) + 3.0D0*H(N-1)
C
C     Rechte Seite
C
      C(0) = 0.5D0*ALPHA*H(0)
      C(N) = 0.5D0*BETA*H(N-1)
C
      DUMMY1 = (FN(2) - FN(1))/H(1)
      RS(1) = 3.0D0*(DUMMY1 - (FN(1) - FN(0))/H(0)) + C(0)*H(0)
      DO 40 I = 2,N-2,1
        DUMMY2 = (FN(I+1) - FN(I))/H(I)
        RS(I)  = 3.0D0*(DUMMY2 - DUMMY1)
        DUMMY1 = DUMMY2
   40 CONTINUE
      RS(N-1) = 3.0D0*((FN(N)-FN(N-1))/H(N-1) - DUMMY1) - C(N)*H(N-1)
C
C-----Berechnung der Koeffizienten C(1) bis C(N-1) durch----------
C     L”sen des Gleichungssystems
C
      CALL TRDSY (N-1,B(1),D(1),RS,C(1),IFLAG)
      IF (IFLAG .NE. 1) THEN
        IF (IFLAG .EQ. -2) THEN
          IERR = -1
        ELSE
          IERR =  1
        ENDIF
        RETURN
      ENDIF
      IERR = 0
C
C-----Berechnung der restlichen Splinekoeffizienten---------------
C
      C(0) = C(1) - C(0)
      C(N) = C(N-1) + C(N)
C
      DO 50 I=0,N-1,1
        B(I) = (FN(I+1)-FN(I))/H(I) - H(I)/3.0D0*(C(I+1)+2.0D0*C(I))
        D(I) = (C(I+1)-C(I))/(3.0D0*H(I))
   50 CONTINUE
      RETURN
      END
C
C
C
      SUBROUTINE PSPTAB (N,NL,TBEG,TEND,T,AX,BX,CX,DX,
     +                   AY,BY,CY,DY,NT,XTAB,YTAB,IERR)
C
C*****************************************************************
C                                                                *
C  Tabellierprogramm für parametrische kubische Splinefunktionen *
C  mit den Komponentenfunktionen SX(T), SY(T) in der Darstellung *
C                                                                *
C  SX := SX(T) = AX(I) + BX(I)(T-T(I)) + CX(I)(T-T(I))**2 +      *
C                                      + DX(I)(T-T(I))**3        *
C                                                                *
C  SY := SY(T) = AY(I) + BY(I)(T-T(I)) + CY(I)(T-T(I))**2 +      *
C                                      + DY(I)(T-T(I))**3        *
C                                                                *
C  für T Element von [T(I),T(I+1)], I=0(1)N-1.                   *
C                                                                *
C                                                                *
C  Das Programm erstellt eine Tabelle der Funktionswerte         *
C  XTAB = SX(TW) und YTAB = SY(TW), TW Element von [TBEG,TEND],  *
C  wobei gilt:                                                   *
C   - ist TBEG < T(0), wird für alle Werte XTAB < T(0) das       *
C     Randpolynom P(0) ausgewertet                               *
C   - ist TEND > T(N), wird für alle Werte XTAB > T(N) das       *
C     Randpolynom P(N-1) ausgewertet                             *
C   - die Intervallgrenzen TBEG und TEND sowie alle dazwischen   *
C     liegenden Stützstellen T(I) werden auf jeden Fall          *
C     tabelliert                                                 *
C   - in jedem Teilintervall [T(I),T(I+1)] wird die Tabelle mit  *
C     äquidistanter Schrittweite H erzeugt, wobei H jeweils von  *
C     der Intervallänge und der gewählten Tabellenlänge NL       *
C     abhängt                                                    *
C   - der Eingabeparameter NL gibt die ungefähre Tabellenlänge   *
C     vor; die tatsächliche Tabellenlänge ist NT+1 (NT ist der   *
C     letzte Tabellenindex). Für NT gilt:  0 < NT < NL+N+3       *
C                                                                *
C                                                                *
C  VORAUSSETZUNGEN:      TBEG < TEND                             *
C  ================      NL > oder = 0                           *
C                                                                *
C                                                                *
C  EINGABEPARAMETER:                                             *
C  =================                                             *
C  N    :  Index des letzten Knotens; T(I), I=0(1)N              *
C  NL   :  Vorgegebene Tabellenlänge zur Dimensionierung der     *
C          Felder XTAB und YTAB                                  *
C  TBEG :  Tabellenanfangswert                                   *
C  TEND :  Tabellenendwert                                       *
C  T    :  1-dim. Feld (0:N) ; T(I) = Knoten, I=0(1)N            *
C  AX   :  1-dim. Feld (0:N) ] Die Feldelemente 0(1)N-1          *
C  BX   :  1-dim. Feld (0:N) ] enthalten die Splinekoeffizienten *
C  CX   :  1-dim. Feld (0:N) ] der Komponentenfunktion SX(T)     *
C  DX   :  1-dim. Feld (0:N) ]                                   *
C  AY   :  1-dim. Feld (0:N) ] Die Feldelemente 0(1)N-1          *
C  BY   :  1-dim. Feld (0:N) ] enthalten die Splinekoeffizienten *
C  CY   :  1-dim. Feld (0:N) ] der Komponentenfunktion SY(T)     *
C  DY   :  1-dim. Feld (0:N) ]                                   *
C                                                                *
C                                                                *
C  AUSDABEPARAMETER:                                             *
C  =================                                             *
C  NT   :  Letzter Tabellenindex; tatsächliche Tabellenlänge-1   *
C  XTAB :  1-dim. Feld (0:NL+N+2) ] Die Feldelemente 0(1)NT      *
C  YTAB :  1-dim. Feld (0:NL+N+2) ] enthalten die Tabellenwerte  *
C  IERR :  Fehlerparameter                                       *
C          = 0 : Alles klar                                      *
C          = 1 : Abbruch wegen TBEG > oder = TEND                *
C          = 2 : Abbruch wegen NL < 0                            *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: keine                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Günter Palm                                       *
C  Datum     : 28.03.1989                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
C-----Deklarationen-----------------------------------------------
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XTAB(0:NL+N+2), YTAB(0:NL+N+2), T(0:N),
     +                 AX(0:N), BX(0:N), CX(0:N), DX(0:N),
     +                 AY(0:N), BY(0:N), CY(0:N), DY(0:N)
C
C-----Überprüfung der Voraussetzungen-----------------------------
C
      IF (TEND .LE. TBEG) THEN
        IERR = 1
        RETURN
      ELSEIF (NL .LT. 0) THEN
        IERR = 2
        RETURN
      ENDIF
      IERR = 0
C
C-----Bestimmung des Intervalls I = IBEG = [T(I),T(I+1)], --------
C     in dem TBEG enthalten ist
C
      I = 0
      K = N
   10 M = (I+K)/2
      IF (TBEG .LT. T(M)) THEN
        K = M
      ELSE
        I = M
      ENDIF
      IF (K .GT. I+1) GOTO 10
      IBEG = I
C
C-----Bestimmung des Intervalls I = IEND = [T(I),T(I+1)], --------
C     in dem TEND enthalten ist
C
      K = N
   20 M = (I+K)/2
      IF (TEND .LT. T(M)) THEN
        K = M
      ELSE
        I = M
      ENDIF
      IF (K .GT. I+1) GOTO 20
      IEND = I
C
C-----Berechnung der Tabellenwerte XTAB(I), YTAB(I), I=0(1)NT-----
C
C     Vorbesetzungen
C
      HP = TEND - TBEG
      FC = NL/HP
      NT = 0
      TW = TBEG
C
      IF (IBEG .NE. IEND) THEN
C
        IF (TBEG .LT. T(0)) THEN
          IP = 0
        ELSE
          IP = 1
        ENDIF
C
        IF (TEND .GT. T(N)) THEN
          IM = 0
        ELSE
          IM = 1
        ENDIF
C
C       Berechnung der Tabellenwerte von TBEG bis T(IBEG+IP)
C
        I = IBEG
        TD = TW - T(I)
        XTAB(NT) = ((DX(I)*TD + CX(I))*TD + BX(I))*TD + AX(I)
        YTAB(NT) = ((DY(I)*TD + CY(I))*TD + BY(I))*TD + AY(I)
        DIF = T(I+IP) - TBEG
        TIV = DIF*FC
        ITV = INT(TIV)
        IF ((TIV-ITV) .GT. 0.0D0) ITV = ITV+1
        IF (ITV .GT. 0) H = DIF/ITV
        DO 30 J = 1,ITV-1,1
          NT = NT + 1
          TW = TW + H
          TD = TW - T(I)
          XTAB(NT) = ((DX(I)*TD + CX(I))*TD + BX(I))*TD + AX(I)
          YTAB(NT) = ((DY(I)*TD + CY(I))*TD + BY(I))*TD + AY(I)
   30   CONTINUE
        NT = NT + 1
        IF ((IEND-IBEG) .NE. 1) THEN
C
C         Berechnung der Tabellenwerte von T(IBEG+IP) bis
C         T(IEND-IM+1)
C
          IBP = IBEG + IP
          IEM = IEND - IM
          DO 40 I = IBP,IEM,1
            TW = T(I)
            XTAB(NT) = AX(I)
            YTAB(NT) = AY(I)
            DIF = T(I+1) - T(I)
            TIV = DIF*FC
            ITV = INT(TIV)
            IF ((TIV-ITV) .GT. 0.0D0) ITV = ITV+1
            IF (ITV .GT. 0) H = DIF/ITV
            DO 50 J = 1,ITV-1,1
              NT = NT + 1
              TW = TW + H
              TD = TW - T(I)
              XTAB(NT) = ((DX(I)*TD+CX(I))*TD+BX(I))*TD+AX(I)
              YTAB(NT) = ((DY(I)*TD+CY(I))*TD+BY(I))*TD+AY(I)
   50       CONTINUE
            NT = NT + 1
   40     CONTINUE
        ENDIF
        TW = T(IEND-IM+1)
      ENDIF
C
C     Berechnung der Tabellenwerte von letzter tabellierter
C     Stelle bis TEND
C
      I = IEND
      TD = TW - T(I)
      XTAB(NT) = ((DX(I)*TD + CX(I))*TD + BX(I))*TD + AX(I)
      YTAB(NT) = ((DY(I)*TD + CY(I))*TD + BY(I))*TD + AY(I)
      DIF = TEND - TW
      TIV = DIF*FC
      ITV = INT(TIV)
      IF ((TIV-ITV) .GT. 0.0D0) ITV = ITV+1
      IF (ITV .GT. 0) H = DIF/ITV
      DO 60 J = 1,ITV-1,1
        NT = NT + 1
        TW = TW + H
        TD = TW - T(I)
        XTAB(NT) = ((DX(I)*TD + CX(I))*TD + BX(I))*TD + AX(I)
        YTAB(NT) = ((DY(I)*TD + CY(I))*TD + BY(I))*TD + AY(I)
   60 CONTINUE
      NT = NT + 1
      TD = TEND - T(I)
      XTAB(NT) = ((DX(I)*TD + CX(I))*TD + BX(I))*TD + AX(I)
      YTAB(NT) = ((DY(I)*TD + CY(I))*TD + BY(I))*TD + AY(I)
      RETURN
      END
C
C
C
      SUBROUTINE PSPPV (N, XN, FN, T, MT, IERR)
C 
C*****************************************************************
C                                                                *
C  'PSPPV' berechnet die Parameterwerte T(I), I=0(1)N, für       *
C  parametrische Splines. Mittels des Übergabeparameters MT      *
C  ist vorzugeben, ob 'PSPPV' die T(I) nach der Sehnenlänge      *
C  oder nach der Bogenlänge berechnet.                           *
C                                                                *
C                                                                *
C  EINGABEPARAMETER:                                             *
C  =================                                             *
C  N  :  Nummer des letzten Knotens                              *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N            *
C  FN :  1-dim. Feld (0:N); FN(i) = Meßwert an der Stelle XN(I)  *
C  MT :  Marke für die Art der Parameterberechnung               *
C        MT =  1: Die Parameterwerte werden nach der Sehnenlänge *
C                 berechnet                                      *
C        MT <> 1: Die Parameterwerte werden nach der Bogenlänge  *
C                 berechnet                                      *
C                                                                *
C                                                                *
C  AUSGABEPARAMETER:                                             *
C  =================                                             *
C  T    :  1-dim. Feld (0:N); T(I) = Parameterwerte              *
C  IERR :  Fehlerparameter                                       *
C          = 0: Alles klar                                       *
C          = 1: Monotoniefehler der Parameterwerte  T(I),        *
C               T(I) > oder = T(I+1), I Element von [0,N-1]      *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: keine                               *
C                                                                *
C  Quellen : Engeln-Müllges, G.; Reutter, F. siehe [ENGE87]      *
C                                                                *
C================================================================*
C                                                                *
C  Autor     : Günter Palm                                       *
C  Datum     : 11.10.1989                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
C-----Deklarationen----------------------------------------------
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), T(0:N)
C
C-----Vorbesetzungen----------------------------------------------
C
      IERR = 1
C
C-----Berechnung der Parameterwerte ... --------------------------
C
      IF (MT .EQ. 1) THEN
C
C       ... nach der Sehnenlänge
C
        T(0) = 0.0D0
        DO 10 I=1,N,1
          DELTX = XN(I) - XN(I-1)
          DELTY = FN(I) - FN(I-1)
          DELTA = DELTX*DELTX + DELTY*DELTY
          IF (DELTA .LE. 0.0D0) RETURN
          T(I) = T(I-1) + DSQRT(DELTA)
   10   CONTINUE
      ELSE
C
C       ... nach der Bogenlänge
C
        T(0) = 0.0D0
        DO 20 I=0,N-2
          A = XN(I+1) - XN(I)
          B = FN(I+1) - FN(I)
          C = XN(I+2) - XN(I+1)
          D = FN(I+2) - FN(I+1)
          E = XN(I+2) - XN(I)
          F = FN(I+2) - FN(I)
          DN = A*D - B*C
          IF (DN .EQ. 0.0D0) THEN
            G = 1.0D0
          ELSE
            DZ = C*E + D*F
            IF (DZ .EQ. 0.0D0) THEN
              G = 1.57D0
            ELSE
              DZ = DZ/DN
              G = DSQRT(1.0D0+DZ*DZ) * DATAN(1.0D0/DABS(DZ))
            ENDIF
          ENDIF
          DT = G * DSQRT(A*A + B*B)
          IF (DT .LE. 0.0D0) RETURN
          T(I+1) = T(I) + DT
   20   CONTINUE
        G =  A
        A = -C
        C = -G
        G =  B
        B = -D
        D = -G
        E = -E
        F = -F
        DN = A*D - B*C
        IF (DN .EQ. 0.0D0) THEN
          G = 1.0D0
        ELSE
          DZ = C*E + D*F
          IF (DZ .EQ. 0.0D0) THEN
            G = 1.57D0
          ELSE
            DZ = DZ/DN
            G = DSQRT(1.0D0+DZ*DZ) * DATAN(1.0D0/DABS(DZ))
          ENDIF
        ENDIF
        DT = G * DSQRT(A*A + B*B)
        IF (DT .LE. 0.0D0) RETURN
        T(N) = T(N-1) + DT
      ENDIF
C
      IERR = 0
      RETURN
      END
C
C
C
      SUBROUTINE ISPLPA (N,XN,FN,T,MT,IB,ALPHA,BETA,BX,CX,DX,
     +                   BY,CY,DY,DUMMY,IERR)
C
C*****************************************************************
C                                                                *
C  'ISPLPA' berechnet die Koeffizienten BX(I), CX(I), DX(I),     *
C  BY(I), CY(I), DY(I), I=0(1)N-1, eines parametrischen kubi-    *
C  schen Interpolationssplines für verschiedene Randbedingungen. *
C  Die Art der Randbedingung ist mittels des Parameters IB vor-  *
C  zugeben. Die parametrische Funktion mit den Kurvenparametern  *
C  T(I), I=0(1)N, setzt sich vektoriell aus den Komponentenfunk- *
C  tionen SX und SY der folgenden Form zusammen:                 *
C                                                                *
C  SX := SX(T) = XN(I) + BX(I)(T-T(I)) + CX(I)(T-T(I))**2 +      *
C                                      + DX(I)(T-T(I))**3        *
C                                                                *
C  SY := SY(T) = FN(I) + BY(I)(T-T(I)) + CY(I)(T-T(I))**2 +      *
C                                      + DY(I)(T-T(I))**3        *
C                                                                *
C  für T Element von [T(I),T(I+1)], I=0(1)N-1.                   *
C                                                                *
C  SX und SY sind nichtparametrische kubische Splines.           *
C                                                                *
C                                                                *
C  VORAUSSETZUNGEN:    1.         N > 2                          *
C  ================    2.      T(I) < T(I+1), I=0(1)N-1          *
C                      3.     FN(0) = FN(N) , für IB = 4         *
C                                                                *
C                                                                *
C  EINGABEPARAMETER:                                             *
C  =================                                             *
C  N  :  Nummer des letzten Knotens                              *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N            *
C  FN :  1-dim. Feld (0:N); FN(I) = Meßwert an der Stelle XN(I)  *
C  T  :  1-dim. Feld (0:N); zu den XN(I), FN(I) gehörende        *
C                           Parameterwerte                       *
C  MT :  Marke für die Vorgabe der Kurvenparameter T(I)          *
C        MT = 0 :  Der Benutzer gibt die Parameterwerte T(I),    *
C                  I=0(1)N, selbst vor                           *
C        MT = 1 :  Keine Vorgabe der Parameterwerte. Die Werte   *
C                  werden im Unterprogramm PSPPV nach der        *
C                  Sehnenlänge berechnet                         *
C        MT = 2 :  Keine Vorgabe der Parameterwerte. Die Werte   *
C                  werden im Unterprogramm PSPPV nach der        *
C                  Bogenlänge berechnet                          *
C                                                                *
C  IB : Art der Randbedingungen                                  *
C       IB = 1: 1. Randableitungen nach dem Parameter vorgegeben *
C       IB = 2: 2. Randableitungen nach dem Parameter vorgegeben *
C       IB = 3: 1. Randableitungen DY/DX vorgegeben              *
C       IB = 4: periodische Splinefunktion                       *
C                                                                *
C  ALPHA : ] 1-dim. Felder (1:2)                                 *
C  BETA  : ]                                                     *
C             für IB = 1 : ] IB. Ableitungen nach dem Parameter  *
C                 IB = 2 : ]       ALPHA(1)=SX(IB)(T(0))         *
C                                  ALPHA(2)=SY(IB)(T(0))         *
C                                  BETA(1) =SX(IB)(T(N))         *
C                                  BETA(2) =SY(IB)(T(N))         *
C             für IB = 3 :  1. Randableitung DY/DX               *
C                           ALPHA(1) = DY/DX (XN(0))             *
C                           BETA(1)  = DY/DX (XN(N))             *
C                           ALPHA(2) : ohne Bedeutung            *
C                           BETA(2)  : ohne Bedeutung            *
C               Ist der Betrag von ALPHA(1) oder BETA(1)         *
C               > oder = 1.E10, wird der entsprechende           *
C               Tangentialvektor wie folgt berechnet:            *
C                . .                                             *
C               (X,Y) = (0,DSIGN(1,FN(1)-FN(0))   (linker Rand)  *
C                . .                                             *
C               (X,Y) = (0,DSIGN(1,FN(N)-FN(N-1)) (rechter Rand) *
C                                                                *
C             für IB = 4 : ohne Bedeutung                        *
C                                                                *
C  (Einen natürlichen parametrischen Interpolationsspline er-    *
C  hält man mit IB=2 und ALPHA(1)=ALPHA(2)=BETA(1)=BETA(2)=0.0)  *
C                                                                *
C                                                                *
C  HILFSFELDER:                                                  *
C  ============                                                  *
C  DUMMY :  1-dim. Feld (1:5*N+1)                                *
C                                                                *
C                                                                *
C  AUSGABEPARAMETER:                                             *
C  =================                                             *
C  XN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1        *
C  BX :  1-dim. Feld (0:N)]    sind die Koeffizienten der        *
C  CX :  1-dim. Feld (0:N)]    Komponentenfunktion SX            *
C  DX :  1-dim. Feld (0:N)]                                      *
C                                                                *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1        *
C  BY :  1-dim. Feld (0:N)]    sind die Koeffizienten der        *
C  CY :  1-dim. Feld (0:N)]    Komponentenfunktion SY            *
C  DY :  1-dim. Feld (0:N)]                                      *
C                              Die Feldelemente BX(N), CX(N),    *
C                              DX(N), BY(N), CY(N) und DY(N)     *
C                              sind Hilfsspeicher                *
C  IERR :  Fehlerparameter                                       *
C          =  0 :  Alles klar                                    *
C          = -1 :  N < 3                                         *
C          = -2 :  IB < 1  oder  IB > 4                          *
C          = -3 :  Monotoniefehler der Parameterwerte T(I),      *
C                  T(I) > oder = T(I+1), I=0(1)N-1               *
C          = -4 :  FN(0) ungleich FN(N) (nur für IB = 4)         *
C          =  1 :  Abbruch in TRDSY oder CYTSY                   *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: ISPL1D, ISPL2D, ISPLPE, PSPPV       *
C                                                                *
C                                                                *
C  Quellen : Engeln-Müllges, G.; Reutter, F., siehe [ENGE87].    *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Günter Palm                                       *
C  Datum     : 15.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), T(0:N), BX(0:N), CX(0:N),
     +                 DX(0:N), BY(0:N), CY(0:N), DY(0:N),
     +                 ALPHA(2), BETA(2), DUMMY(1:5*N+1)
C
C-----Überprüfung der Vorbedingungen------------------------------
C
      IERR = -1
      IF (N .LT. 3) RETURN
      IF (IB .LT. 1  .OR.  IB .GT. 4) THEN
        IERR = -2
        RETURN
      ENDIF
C
C-----Ermittlung und/oder Überprüfung der Parameterwerte----------
C
      IF (MT .GT. 0) THEN
C
C       Berechnung der Parameterwerte im Unterprogramm PSPPV
C
        CALL PSPPV (N,XN,FN,T,MT,IERR)
        IF (IERR .NE. 0) THEN
          IERR = -3
          RETURN
        ENDIF
      ELSE
C
C       Überprüfung der Vorgegebenen Parameterwerte
C
        IERR = -3
        DO 20 I=0,N-1,1
          IF (T(I+1) .LE. T(I)) RETURN
   20   CONTINUE
      ENDIF
C
C-----Berechnung der Splinekoeffizienten, wobei beim 2. Aufruf----
C     der Unterprogramme die erneute Zerlegung der Matrix des
C     Gleichungssystems zur Berechnung der CX(I), CY(I) umgangen
C     wird
C
      IF (IB .EQ. 1) THEN
        CALL ISPL1D (N,T,XN,ALPHA(1),BETA(1),1,BX,CX,DX,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
        IF (IERR .NE. 0) RETURN
        CALL ISPL1D (N,T,FN,ALPHA(2),BETA(2),2,BY,CY,DY,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
      ELSEIF (IB .EQ. 2) THEN
        CALL ISPL2D (N,T,XN,ALPHA(1),BETA(1),1,BX,CX,DX,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
        IF (IERR .NE. 0) RETURN
        CALL ISPL2D (N,T,FN,ALPHA(2),BETA(2),2,BY,CY,DY,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
      ELSEIF (IB .EQ. 3) THEN
C
C       Berechnung der Tangentialvektoren der 1. Ableitungen
C       für SX und SY
C
        UB = 1.0D10
        IF (DABS(ALPHA(1)) .GE. UB) THEN
          ALPHAX = 0.0D0
          ALPHAY = DSIGN(1.0D0,FN(1)-FN(0))
        ELSE
          ROOT   = DSQRT(1.0D0/(1.0D0 + ALPHA(1)*ALPHA(1)))
          ALPHAX = DSIGN(ROOT,XN(1)-XN(0))
          ALPHAY = ALPHAX*ALPHA(1)
        ENDIF
        IF (DABS(BETA(1)) .GE. UB) THEN
          BETAX = 0.0D0
          BETAY = DSIGN(1.0D0,FN(N)-FN(N-1))
        ELSE
          ROOT  = DSQRT(1.0D0/(1.0D0 + BETA(1)*BETA(1)))
          BETAX = DSIGN(ROOT,XN(N)-XN(N-1))
          BETAY = BETAX*BETA(1)
        ENDIF
        CALL ISPL1D (N,T,XN,ALPHAX,BETAX,1,BX,CX,DX,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
        IF (IERR .NE. 0) RETURN
        CALL ISPL1D (N,T,FN,ALPHAY,BETAY,2,BY,CY,DY,DUMMY(1),
     +               DUMMY(N+1),DUMMY(2*N),DUMMY(3*N-1),IERR)
      ELSE
        CALL ISPLPE (N,T,XN,1,BX,CX,DX,DUMMY(1),DUMMY(N+2),
     +               DUMMY(2*N+2),DUMMY(3*N+2),DUMMY(4*N+2),IERR)
        IF (IERR .NE. 0) RETURN
        CALL ISPLPE (N,T,FN,2,BY,CY,DY,DUMMY(1),DUMMY(N+2),
     +               DUMMY(2*N+2),DUMMY(3*N+2),DUMMY(4*N+2),IERR)
      ENDIF
      RETURN
      END
C
C
C
      SUBROUTINE ISPL1D (N,XN,FN,ALPHA,BETA,MREP,B,C,D,
     +                   H,DU,DM,RS,IERR)
C
C*******************************************************************
C                                                                  *
C  'ISPL1D' berechnet die Koeffizienten B(I), C(I), D(I),          *
C  I=0(1)N-1, eines kubischen Interpolationssplines mit vorgege-   *
C  bener 1. Randableitung.                                         *
C                                                                  *
C  Die Splinefunktion wird dargestellt in der Form:                *
C                                                                  *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +               *
C                               + D(I)(X-XN(I))**3                 *
C                                                                  *
C  für X Element von [XN(I),XN(I+1)], I=0(1)N-1.                   *
C                                                                  *
C                                                                  *
C  VORAUSSETZUNGEN:    1.         N > 2                            *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1           *
C                                                                  *
C                                                                  *
C  BEMERKUNG:  'ISPL1D' sollte nicht direkt, sondern über das      *
C  ==========  Unterprogramm 'ISPLNP' - im Falle parametrischer    *
C              Splines über das Unterprogramm 'ISPLPA' - aufge-    *
C              rufen werden. 'ISPLNP' bzw. 'ISPLPA' überprüfen     *
C              auch die Voraussetzungen.                           *
C                                                                  *
C                                                                  *
C  EINGABEPARAMETER:                                               *
C  =================                                               *
C  N  :  Nummer des letzten Knotens                                *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N              *
C  FN :  1-dim. Feld (0:N); FN(I) = Meßwert an der Stelle XN(I)    *
C                                                                  *
C  ALPHA :  1. Randableitung bei XN(0)                             *
C  BETA  :  1. Randableitung bei XN(N)                             *
C                                                                  *
C  MREP  : Marke für wiederholten Aufruf des Unterprogramms        *
C          MREP = 1: Es wird eine vollständige Berechnung der      *
C                    Matrixelemente zur Berechnung der C(I) und    *
C                    deren Zerlegung im Unterprogramm TRDSY        *
C                    durchgeführt                                  *
C          MREP = 2: Es wird nur die rechte Seite des Gleichungs-  *
C                    systems neu berechnet. Mit den Werten der     *
C                    Felder DU und DM aus dem 1. Aufruf wird im    *
C                    Unterprogramm TRDSYS die Lösung berechnet     *
C                    (Umgehung der erneuten Zerlegung bei para-    *
C                    metrischen Splines).                          *
C                    Die Elemente der Felder H, DU und DM dürfen   *
C                    hierbei nach dem 1. Aufruf nicht verändert    *
C                    werden!                                       *
C                                                                  *
C                                                                  *
C  HILFSFELDER:                                                    *
C  ============                                                    *
C  H  :   1-dim. Feld (0:N-1)                                      *
C  DU : ]                                                          *
C  DM : ] 1-dim. Felder (1:N-1)                                    *
C  RS : ]                                                          *
C                                                                  *
C                                                                  *
C  AUSGABEPARAMETER:                                               *
C  =================                                               *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1          *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der          *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                   *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-        *
C                              speicher                            *
C  IERR :  Fehlerparameter                                         *
C          =  0 :  Alles klar                                      *
C          = -1 :  N < 3                                           *
C          = -5 :  Falscher Wert für MREP                          *
C          =  1 :  Abbruch in TRDSY                                *
C                                                                  *
C------------------------------------------------------------------*
C                                                                  *
C  Benötigte Unterprogramme: TRDSY, TRDSYS                         *
C                                                                  *
C                                                                  *
C  Quellen : Engeln-Müllges, G.; Reutter, F., siehe [ENGE87].      *
C                                                                  *
C*******************************************************************
C                                                                  *
C  Autor     : Günter Palm                                         *
C  Datum     : 15.04.1988                                          *
C  Quellcode : FORTRAN 77                                          *
C                                                                  *
C*******************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 H(0:N-1), DU(1:N-1), DM(1:N-1), RS(1:N-1)
C
C-----Überprüfung der Marke für wiederholten Aufruf---------------
C
      IERR = -5
      IF (MREP .NE. 1  .AND.  MREP .NE. 2) RETURN
C
C-----Berechnung der Hilfsgrößen und der Matrixelemente-----------
C     (Ober- und Hauptdiagonale) des Gleichungssystems, wenn
C     es sich um den 1. Aufruf handelt
C
      IF (MREP .EQ. 1) THEN
C
C       Hilfsgrößen
C
        DO 10 I=0,N-1,1
          H(I) = XN(I+1) - XN(I)
   10   CONTINUE
C
C       Oberdiagonale
C
        DO 20 I=1,N-2,1
          DU(I) = H(I)
   20   CONTINUE
C
C       Hauptdiagonale
C
        DM(1) = 1.5D0*H(0) + 2.D0*H(1)
        DO 30 I=2,N-2,1
          DM(I) = 2.0D0*(H(I-1)+H(I))
   30   CONTINUE
        DM(N-1) = 2.0D0*H(N-2) + 1.5D0*H(N-1)
      ENDIF
C
C-----Berechnung der rechten Seite--------------------------------
C
      DUMMY1 = (FN(2) - FN(1))/H(1)
      RS(1)  = 3.0D0*(DUMMY1 - 0.5D0*(3.0D0*(FN(1)-FN(0))/H(0)
     +         - ALPHA))
      DO 40 I = 2,N-2,1
        DUMMY2 = (FN(I+1) - FN(I))/H(I)
        RS(I)  = 3.0D0*(DUMMY2 - DUMMY1)
        DUMMY1 = DUMMY2
   40 CONTINUE
      RS(N-1) = 3.0D0*(0.5D0*(3.0D0*(FN(N)-FN(N-1))/H(N-1)-BETA)
     +          - DUMMY1)
C
C-----Berechnung der Koeffizienten C(1) bis C(N-1) durch----------
C     Lösen des Gleichungssystems ...
C
      IF (MREP .EQ. 1) THEN
C
C       ... mit Zerlegung
C
        CALL TRDSY (N-1,DM,DU,RS,C(1),IFLAG)
        IF (IFLAG .NE. 1) THEN
          IF (IFLAG .EQ. -2) THEN
            IERR = -1
          ELSE

            IERR =  1
          ENDIF
          RETURN
        ENDIF
      ELSE
C
C       ... ohne Zerlegung
C
        CALL TRDSYS (N-1,DM,DU,RS,C(1))
      ENDIF
      IERR = 0
C
C-----Berechnung der restlichen Splinekoeffizienten---------------
C
      C(0) = 1.5D0*((FN(1)-FN(0))/H(0) - ALPHA)/H(0) - 0.5D0*C(1)
      C(N) = 1.5D0*(BETA-(FN(N)-FN(N-1))/H(N-1))/H(N-1) - 0.5D0*C(N-1)
C
      DO 50 I=0,N-1,1
        B(I) = (FN(I+1)-FN(I))/H(I) - H(I)/3.0D0*(C(I+1)+2.0D0*C(I))
        D(I) = (C(I+1)-C(I))/(3.0D0*H(I))
   50 CONTINUE
      RETURN
      END
C
C
C
      SUBROUTINE ISPL2D (N,XN,FN,ALPHA,BETA,MREP,B,C,D,
     +                   H,DU,DM,RS,IERR)
C
C*******************************************************************
C                                                                  *
C  'ISPL2D' berechnet die Koeffizienten B(I), C(I), D(I),          *
C  I=0(1)N-1, eines kubischen Interpolationssplines mit vorgege-   *
C  bener 2. Randableitung.                                         *
C                                                                  *
C  Die Splinefunktion wird dargestellt in der Form:                *
C                                                                  *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +               *
C                               + D(I)(X-XN(I))**3                 *
C                                                                  *
C  für X Element von [XN(I),XN(I+1)], I=0(1)N-1.                   *
C                                                                  *
C                                                                  *
C  VORAUSSETZUNGEN:    1.         N > 2                            *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1           *
C                                                                  *
C                                                                  *
C  BEMERKUNG:  'ISPL2D' sollte nicht direkt, sondern über das      *
C  ==========  Unterprogramm 'ISPLNP' - im Falle parametrischer    *
C              Splines über das Unterprogramm 'ISPLPA' - aufge-    *
C              rufen werden. 'ISPLNP' bzw. 'ISPLPA' überprüfen     *
C              auch die Voraussetzungen.                           *
C                                                                  *
C                                                                  *
C  EINGABEPARAMETER:                                               *
C  =================                                               *
C  N  :  Nummer des letzten Knotens                                *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N              *
C  FN :  1-dim. Feld (0:N); FN(I) = Meßwert an der Stelle XN(I)    *
C                                                                  *
C  ALPHA :  2. Randableitung bei XN(0)                             *
C  BETA  :  2. Randableitung bei XN(N)                             *
C                                                                  *
C  (Einen natürlichen Interpolationsspline erhält man mit          *
C   ALPHA = BETA = 0.0)                                            *
C                                                                  *
C  MREP  : Marke für wiederholten Aufruf des Unterprogramms        *
C          MREP = 1: Es wird eine vollständige Berechnung der      *
C                    Matrixelemente zur Berechnung der C(I) und    *
C                    deren Zerlegung im Unterprogramm TRDSY        *
C                    durchgeführt                                  *
C          MREP = 2: Es wird nur die rechte Seite des Gleichungs-  *
C                    systems neu berechnet. Mit den Werten der     *
C                    Felder DU und DM aus dem 1. Aufruf wird im    *
C                    Unterprogramm TRDSYS die Lösung berechnet     *
C                    (Umgehung der erneuten Zerlegung bei para-    *
C                    metrischen Splines).                          *
C                    Die Elemente der Felder H, DU und DM dürfen   *
C                    hierbei nach dem 1. Aufruf nicht verändert    *
C                    werden!                                       *
C                                                                  *
C                                                                  *
C  HILFSFELDER:                                                    *
C  ============                                                    *
C  H  :   1-dim. Feld (0:N-1)                                      *
C  DU : ]                                                          *
C  DM : ] 1-dim. Felder (1:N-1)                                    *
C  RS : ]                                                          *
C                                                                  *
C                                                                  *
C  AUSGABEPARAMETER:                                               *
C  =================                                               *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1          *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der          *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                   *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-        *
C                              speicher                            *
C  IERR :  Fehlerparameter                                         *
C          =  0 :  Alles klar                                      *
C          = -1 :  N < 3                                           *
C          = -5 :  Falscher Wert für MREP                          *
C          =  1 :  Abbruch in TRDSY                                *
C                                                                  *
C------------------------------------------------------------------*
C                                                                  *
C  Benötigte Unterprogramme: TRDSY, TRDSYS                         *
C                                                                  *
C                                                                  *
C  Quellen : Engeln-Müllges, G.; Reutter, F., siehe [ENGE87].      *
C                                                                  *
C*******************************************************************
C                                                                  *
C  Autor     : Günter Palm                                         *
C  Datum     : 15.04.1988                                          *
C  Quellcode : FORTRAN 77                                          *
C                                                                  *
C*******************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 H(0:N-1), DU(1:N-1), DM(1:N-1), RS(1:N-1)
C
C-----Überprüfung der Marke für wiederholten Aufruf---------------
C
      IERR = -5
      IF (MREP .NE. 1  .AND.  MREP .NE. 2) RETURN
C
C-----Berechnung der Hilfsgrößen und der Matrixelemente-----------
C     (Ober- und Hauptdiagonale) des Gleichungssystems, wenn
C     es sich um den 1. Aufruf handelt
C
      IF (MREP .EQ. 1) THEN
C
C       Hilfsgrößen
C
        DO 10 I=0,N-1,1
          H(I) = XN(I+1) - XN(I)
   10   CONTINUE
C
C       Oberdiagonale
C
        DO 20 I=1,N-2,1
          DU(I) = H(I)
   20   CONTINUE
C
C       Hauptdiagonale
C
        DO 30 I=1,N-1,1
          DM(I) = 2.0D0*(H(I-1) + H(I))
   30   CONTINUE
      ENDIF
C
C-----Berechnung der rechten Seite--------------------------------
C
      C(0) = 0.5D0*ALPHA
      C(N) = 0.5D0*BETA
C
      DUMMY1 = (FN(2) - FN(1))/H(1)
      RS(1)  = 3.0D0*(DUMMY1 - (FN(1) - FN(0))/H(0)) - H(0)*C(0)
      DO 40 I=2,N-2,1
        DUMMY2 = (FN(I+1) - FN(I))/H(I)
        RS(I)  = 3.0D0*(DUMMY2 - DUMMY1)
        DUMMY1 = DUMMY2
   40 CONTINUE
      RS(N-1) = 3.0D0*((FN(N)-FN(N-1))/H(N-1) - DUMMY1) - C(N)*H(N-1)
C
C-----Berechnung der Koeffizienten C(1) bis C(N-1) durch----------
C     Lösen des Gleichungssystems ...
C
      IF (MREP .EQ. 1) THEN
C
C       ... mit Zerlegung
C
        CALL TRDSY (N-1,DM,DU,RS,C(1),IFLAG)
        IF (IFLAG .NE. 1) THEN
          IF (IFLAG .EQ. -2) THEN
            IERR = -1
          ELSE
            IERR =  1
          ENDIF
          RETURN
        ENDIF
      ELSE
C
C       ... ohne Zerlegung
C
        CALL TRDSYS (N-1,DM,DU,RS,C(1))
      ENDIF
      IERR = 0
C
C-----Berechnung der restlichen Splinekoeffizienten---------------
C
      DO 50 I=0,N-1,1
        B(I) = (FN(I+1)-FN(I))/H(I) - H(I)/3.0D0*(C(I+1) + 2.0D0*C(I))
        D(I) = (C(I+1)-C(I))/(3.0D0*H(I))
   50 CONTINUE
      RETURN
      END
C
C
C
      SUBROUTINE ISPLPE (N,XN,FN,MREP,B,C,D,H,DU,DM,RC,RS,IERR)
C
C*******************************************************************
C                                                                  *
C  'ISPLPE' berechnet die Koeffizienten B(I), C(I), D(I),          *
C  I=0(1)N-1, eines periodischen kubischen Interpolationssplines.  *
C                                                                  *
C  Die Splinefunktion wird dargestellt in der Form:                *
C                                                                  *
C  S(X) = FN(I) + B(I)(X-XN(I)) + C(I)(X-XN(I))**2 +               *
C                               + D(I)(X-XN(I))**3                 *
C                                                                  *
C  für X Element von [XN(I),XN(I+1)], I=0(1)N-1.                   *
C                                                                  *
C                                                                  *
C  VORAUSSETZUNGEN:    1.         N > 2                            *
C  ================    2.     XN(I) < XN(I+1), I=0(1)N-1           *
C                      3.     FN(0) = FN(N)                        *
C                                                                  *
C                                                                  *
C  BEMERKUNG:  'ISPLPE' sollte nicht direkt, sondern über das      *
C  ==========  Unterprogramm 'ISPLNP' - im Falle parametrischer    *
C              bzw. transformiert-parametrischer Splines über      *
C              das Unterprogramm 'ISPLPA' bzw. 'ISPLTR' - aufge-   *
C              rufen werden. Diese Programme überprüfen auch       *
C              die Voraussetzungen 1 und 2.                        *
C                                                                  *
C                                                                  *
C  EINGABEPARAMETER:                                               *
C  =================                                               *
C  N  :  Nummer des letzten Knotens                                *
C  XN :  1-dim. Feld (0:N); XN(I) = Knoten, I = 0(1)N              *
C  FN :  1-dim. Feld (0:N); FN(I) = Meßwert an der Stelle XN(I)    *
C                                                                  *
C  MREP :  Marke für wiederholten Aufruf des Unterprogramms        *
C          MREP = 1: Es wird eine vollständige Berechnung der      *
C                    Matrixelemente zur Berechnung der C(I) und    *
C                    deren Zerlegung im Unterprogramm CYTSY        *
C                    durchgeführt                                  *
C          MREP = 2: Es wird nur die rechte Seite des Gleichungs-  *
C                    systems neu berechnet. Mit den Werten der     *
C                    Felder DU, DM und RC aus dem 1. Aufruf wird   *
C                    im Unterprogramm CYTSYS die Lösung berechnet  *
C                    (Umgehung der erneuten Zerlegung bei para-    *
C                    metrischen Splines).                          *
C                    Die Elemente der Felder H, DU, DM und RC      *
C                    dürfen hierbei nach dem 1. Aufruf nicht       *
C                    verändert werden!                             *
C                                                                  *
C                                                                  *
C  HILFSFELDER:                                                    *
C  ============                                                    *
C  H  :   1-dim. Feld (0:N)                                        *
C  DU : ]                                                          *
C  DM : ] 1-dim. Felder (1:N)                                      *
C  RC : ]                                                          *
C  RS : ]                                                          *
C                                                                  *
C                                                                  *
C  AUSGABEPARAMETER:                                               *
C  =================                                               *
C  FN :  1-dim. Feld (0:N)]    Die Feldelemente 0 bis N-1          *
C  B  :  1-dim. Feld (0:N)]    sind die Koeffizienten der          *
C  C  :  1-dim. Feld (0:N)]    Splinefunktion S.                   *
C  D  :  1-dim. Feld (0:N)]    B(N), C(N), D(N) sind Hilfs-        *
C                              speicher                            *
C  IERR :  Fehlerparameter                                         *
C          =  0 :  Alles klar                                      *
C          = -1 :  N < 3                                           *
C          = -4 :  FN(0) ungleich FN(N)                            *
C          = -5 :  Falscher Wert für MREP                          *
C          =  1 :  Abbruch in CYTSY                                *
C                                                                  *
C------------------------------------------------------------------*
C                                                                  *
C  Benötigte Unterprogramme: CYTSY, CYTSYS                         *
C                                                                  *
C                                                                  *
C  Quellen : Engeln-Müllges, G.; Reutter, F., siehe [ENGE87].      *
C                                                                  *
C*******************************************************************
C                                                                  *
C  Autor     : Günter Palm                                         *
C  Datum     : 15.04.1988                                          *
C  Quellcode : FORTRAN 77                                          *
C                                                                  *
C*******************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION XN(0:N), FN(0:N), B(0:N), C(0:N), D(0:N),
     +                 H(0:N), DU(1:N), DM(1:N), RC(1:N),RS(1:N)
C
C-----Überprüfung der Vorbedingung der Periodizität---------------
C
      IERR = -4
      IF (FN(N) .NE. FN(0)) RETURN
C
C-----Überprüfung der Marke für wiederholten Aufruf---------------
C
      IERR = -5
      IF (MREP .NE. 1  .AND.  MREP .NE. 2) RETURN
C
C-----Berechnung der Hilfsgrößen und der Matrixelemente-----------
C     (Ober- und Hauptdiagonale) des Gleichungssystems, wenn
C     es sich um den 1. Aufruf handelt
C
      IF (MREP .EQ. 1) THEN
C
C       Hilfsgrößen
C
        DO 10 I=0,N-1,1
          H(I) = XN(I+1) - XN(I)
   10   CONTINUE
        H(N) = H(0)
C
C       Oberdiagonale
C
        DO 20 I=1,N-1,1
          DU(I) = H(I)
   20   CONTINUE
        DU(N) = H(0)
C
C       Hauptdiagonale
C
        DO 30 I=1,N,1
          DM(I) = 2.0D0*(H(I-1) + H(I))
   30   CONTINUE
      ENDIF
C
C-----Berechnung der rechten Seite--------------------------------
C
      DUMMY1 = (FN(1) - FN(0))/H(0)
      DO 40 I=1,N-1,1
        DUMMY2 = (FN(I+1) - FN(I))/H(I)
        RS(I)  = 3.0D0*(DUMMY2 - DUMMY1)
        DUMMY1 = DUMMY2
   40 CONTINUE
      RS(N) = 3.0D0*((FN(1)-FN(0))/H(0) - DUMMY1)
C
C-----Berechnung der Koeffizienten C(1) bis C(N-1) durch----------
C     Lösen des Gleichungssystems ...
C
      IF (MREP .EQ. 1) THEN
C
C       ... mit Zerlegung
C
        CALL CYTSY (N,DM,DU,RC,RS,C(1),IFLAG)
        IF (IFLAG .NE. 1) THEN
          IF (IFLAG .EQ. -2) THEN
            IERR = -1
          ELSE
            IERR =  1
          ENDIF
          RETURN
        ENDIF
      ELSE
C
C       ... ohne Zerlegung
C
        CALL CYTSYS (N,DM,DU,RC,RS,C(1))
      ENDIF
      IERR = 0
C
C-----Berechnung der restlichen Splinekoeffizienten---------------
C
      C(0) = C(N)
C
      DO 50 I=0,N-1,1
        B(I) = (FN(I+1)-FN(I))/H(I) - H(I)/3.0D0*(C(I+1)+2.0D0*C(I))
        D(I) = (C(I+1)-C(I))/(3.0D0*H(I))
   50 CONTINUE
      RETURN
      END
C
C
C
      SUBROUTINE TRDSY (N,DM,DU,RS,X,MARK)
C
C*****************************************************************
C                                                                *
C     Lösung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit tridiagonaler, symmetrischer, positiv definiter        *
C     Matrix A.                                                  *
C     Die Matrix A wird durch die beiden 1-dim. Felder           *
C     DM und DU beschrieben. Das Gleichungssystem hat die Form:  *
C                                                                *
C     DM(1) * X(1) + DU(1) * X(2)                      = RS(1)   *
C     DU(I-1) * X(I-1) + DM(I) * X(I) + DU(I) * X(I+1) = RS(I)   *
C     DU(N-1) * X(N-1) + DM(N) * X(N)                  = RS(N)   *
C                                                                *
C     für I=2(1),N-1.                                            *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1)                         *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DM   :)                                                    *
C     DU   :) überschrieben mit Hilfsvektoren                    *
C     RS   :)                                                    *
C     X    : 1-dim. Feld (1:N), das die Lösung des Gleichungs-   *
C            systems enthält                                     *
C     MARK : Fehlerparameter                                     *
C            MARK= 1 : alles klar                                *
C            MARK= 0 : Matrix numerisch nicht streng regulär     *
C                      (d.h. A ist nicht positiv definit)        *
C            MARK=-1 : Matrix A ist nicht positiv definit        *
C            MARK=-2 : Voraussetzung N > 2 verletzt              *
C                                                                *
C     BEMERKUNG: Die Determinante von A kann nach dem Aufruf     *
C                im Falle MARK = 1 wie folgt berechnet werden:   *
C                   DET A = DM(1) * DM(2) * ... * DM(N)          *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: TRDSYP, TRDSYS, MACHPD              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 25.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N),RS(1:N),X(1:N)
      MARK = -2
      IF (N .LT. 3) RETURN
C
C  Zerlegung der Matrix A
C
      CALL TRDSYP (N,DM,DU,MARK)
C
C  falls MARK = 1, wird die Vorwärts- und Rückwärtselimination
C  durchgeführt
C
      IF (MARK .EQ. 1) THEN
         CALL TRDSYS (N,DM,DU,RS,X)
      ENDIF
      RETURN
      END
C
C
C
      SUBROUTINE TRDSYP (N,DM,DU,MARK)
C
C*****************************************************************
C                                                                *
C     Zerlegung einer tridiagonalen, symmetrischen, positiv      *
C     definiten Matrix A, die durch die beiden 1-dim. Felder     *
C     DM und DU beschrieben wird, in ihre Faktoren               *
C     R(TRANSP) * D * R  nach dem Cholesky-Verfahren für tridia- *
C     gonale Matrizen.                                           *
C     Die Form des Gleichungssystems kann aus der Beschreibung   *
C     der SUBROUTINE TRDSY entnommen werden.                     *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1);                        *
C            die untere Nebendiagonale braucht aus Symmetrie-    *
C            gründen nicht eingegeben zu werden                  *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DM   :) überschrieben mit Hilfsfeldern, die die            *
C     DU   :) Zerlegungsmatrizen von A enthalten. Die Neben-     *
C             diagonale der normierten oberen tridiagonalen      *
C             Dreiecksmatrix R wird in DU abgespeichert, die     *
C             Diagonalmatrix in DM.                              *
C     MARK : Fehlerparameter                                     *
C            MARK= 1 : alles klar                                *
C            MARK= 0 : Matrix numerisch nicht streng regulär     *
C                      (d.h. A ist nicht positiv definit)        *
C            MARK=-1 : Matrix A nicht positiv definit            *
C            MARK=-2 : Voraussetzung N > 2 verletzt              *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: MACHPD                              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 25.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N)
C
C   Berechnung der Maschinengenauigkeit
C
      FMACHP = 1.0D0
   10 FMACHP = 0.5D0 * FMACHP
      IF (MACHPD(1.0D0+FMACHP) .EQ. 1) GOTO 10
      FMACHP = FMACHP * 2.0D0
C
C   Festlegung der Schranke für den relativen Fehler
C
      EPS = 4.0D0 * FMACHP
C
C   Überprüfung der Voraussetzung N > 2
C
      MARK = -2
      IF (N .LT. 3) RETURN
      DU(N) = 0.0D0
C
C   Abfrage auf positive Definitheit und strenge Regularität der
C   Matrix A für N=1.
C
      ROW = DABS(DM(1)) + DABS(DU(1))
      IF (ROW .EQ. 0.0D0) THEN
         MARK = 0
         RETURN
      ENDIF
      D = 1.0D0/ROW
      IF (DM(1) .LT. 0.0D0) THEN
         MARK = -1
         RETURN
      ELSEIF (DABS(DM(1))*D .LE. EPS) THEN
         MARK = 0
         RETURN
      ENDIF
C
C   Durchführung der Zerlegung bei gleichzeitiger Abfrage auf
C   positive Definitheit und strenge Regularität der Matrix A
C
      DUMMY = DU(1)
      DU(1) = DU(1)/DM(1)
      DO 20 I=2,N,1
         ROW = (DABS (DM(I)) + DABS(DU(I)) + DABS(DUMMY))
         IF (ROW .EQ. 0.0D0) THEN
            MARK = 0
            RETURN
         ENDIF
         D = 1.0D0/ROW
         DM(I) = DM(I) - DUMMY * DU(I-1)
         IF (DM(I) .LT. 0.0D0) THEN
            MARK = -1
            RETURN
         ELSEIF (DABS(DM(I))*D .LE. EPS) THEN
            MARK = 0
            RETURN
         ENDIF
         IF (I .LT. N) THEN
            DUMMY = DU(I)
            DU(I) = DU(I)/DM(I)
         ENDIF
   20 CONTINUE
      MARK=1
      RETURN
      END
C
C
C
      SUBROUTINE TRDSYS (N,DM,DU,RS,X)
C
C*****************************************************************
C                                                                *
C     Lösung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit tridiagonaler, symmetrischer, positiv definiter        *
C     Matrix A, nachdem die Zerlegungsmatrizen mit Hilfe der     *
C     SUBROUTINE TRDSYP berechnet worden sind.                   *
C     Hier werden diese Zerlegungsmatrizen als Eingabematrizen   *
C     verwendet und in den beiden 1-dim. Feldern DM und DU       *
C     abgespeichert.                                             *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); Diagonalmatrix D                 *
C     DU   : 1-dim. Feld (1:N); obere Dreiecksmatrix R           *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     X    : 1-dim. Feld (1:N), das die Lösung des Gleichungs-   *
C            systems enthält                                     *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: keine                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 25.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N),RS(1:N),X(1:N)
C
C  Vorwärtselimination
C
      DUMMY = RS(1)
      RS(1) = DUMMY/DM(1)
      DO 10 I=2,N,1
         DUMMY = RS(I) - DU(I-1) * DUMMY
         RS(I) = DUMMY/DM(I)
   10 CONTINUE
C
C  Rückwärtselimination
C
      X(N) = RS(N)
      DO 20 I=N-1,1,-1
         X(I) = RS(I) - DU(I) * X(I+1)
   20 CONTINUE
      RETURN
      END
C
C
C
      SUBROUTINE CYTSY (N,DM,DU,CR,RS,X,MARK)
C
C*****************************************************************
C                                                                *
C     Lösung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit zyklisch tridiagonaler, symmetrischer, positiv         *
C     definiter Matrix A.                                        *
C     Die Matrix A wird durch die beiden 1-dim. Felder DM und DU *
C     beschrieben. Das Gleichungssystem hat die Form:            *             *
C                                                                *
C     DM(1) * X(1) + DU(1) * X(2) + DU(N) * X(N)       = RS(1)   *
C     DU(I-1) * X(I-1) + DM(I) * X(I) + DU(I) * X(I+1) = RS(I)   *
C     DU(N) * X(1) + DU(N-1) * X(N-1) + DM(N) * X(N)   = RS(N)   *
C                                                                *
C     für I=2(1),N-1.                                            *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1); das Nichtdiagonal-     *
C            element A(1,N) ist durch DU(N) gegeben              *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DM   :)                                                    *
C     DU   :) überschrieben mit Hilfsvektoren                    *
C     CR   :)                                                    *
C     RS   :)                                                    *
C     X    : 1-dim. Feld (1:N), das die Lösung des Gleichungs-   *
C            systems enthält                                     *
C     MARK : Fehlerparameter                                     *
C            MARK=-2 : Voraussetzung N > 2 verletzt              *
C            MARK=-1 : Matrix A ist nicht positiv definit        *
C            MARK= 0 : Matrix numerisch nicht streng regulär     *
C                      (d.h. A ist nicht positiv definit)        *
C            MARK= 1 : alles klar                                *
C                                                                *
C     BEMERKUNG: Die Determinante von A kann nach dem Aufruf     *
C                im Falle MARK = 1 wie folgt berechnet werden:   *
C                   DET A = DM(1) * DM(2) * ... * DM(N)          *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: CYTSYP, CYTSYS, MACHPD              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 27.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N),CR(1:N),RS(1:N),X(1:N)
      MARK = -2
      IF (N .LT. 3) RETURN
C
C  Faktorisierung der Matrix A
C
      CALL CYTSYP (N,DM,DU,CR,MARK)
C
C  falls MARK = 1, wird die Vorwärts- und Rückwärtselimination
C  durchgeführt
C
      IF (MARK .EQ. 1) THEN
         CALL CYTSYS (N,DM,DU,CR,RS,X)
      ENDIF
      RETURN
      END
C
C
C
      SUBROUTINE CYTSYP (N,DM,DU,CR,MARK)
C
C*****************************************************************
C                                                                *
C     Zerlegung einer zyklisch tridiagonalen, symmetrischen,     *
C     positiv definiten Matrix A, die durch die beiden 1-dim.    *
C     Felder DM und DU beschrieben wird, in ihre Faktoren        *
C     R(TRANSP) * D * R nach dem Cholesky-Verfahren für tridia-  *
C     gonale Matrizen.                                           *
C     Die Form des Gleichungssystems kann aus der Beschreibung   *
C     der SUBROUTINE CYTSY entnommen werden.                     *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); Hauptdiagonale                   *
C            DM(1), DM(2), ... , DM(N)                           *
C     DU   : 1-dim. Feld (1:N); obere Nebendiagonale             *
C            DU(1), DU(2), ... , DU(N-1); das Nichtdiagonal-     *
C            element A(1,N) ist durch DU(N) gegeben.             *
C            Die untere Nebendiagonale braucht aus Symmetrie-    *
C            gründen nicht eingegeben zu werden.                 *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     DM   :) überschrieben mit Hilfsfeldern, die die            *
C     DU   :) Zerlegungsmatrizen von A enthalten. Die Neben-     *
C     CR   :) diagonalen der normierten oberen tridiagonalen     *
C             Dreiecksmatrix R wird in DU abgespeichert, die     *
C             Diagonalmatrix in DM und die rechte Spalte in CR.  *
C     MARK : Fehlerparameter                                     *
C            MARK=-2 : Voraussetzung N > 2 verletzt              *
C            MARK=-1 : Matrix A nicht positiv definit            *
C            MARK= 0 : Matrix numerisch nicht streng regulär     *
C                      (d.h. A ist nicht positiv definit)        *
C            MARK= 1 : alles klar                                *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: MACHPD                              *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 27.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N),CR(1:N)
C
C   Berechnung der Maschinengenauigkeit
C
      FMACHP = 1.0D0
   10 FMACHP = 0.5D0 * FMACHP
      IF (MACHPD(1.0D0+FMACHP) .EQ. 1) GOTO 10
      FMACHP = FMACHP * 2.0D0
C
C   Festlegung der Schranke für den relativen Fehler
C
      EPS = 4.0D0 * FMACHP
C
C   Überprüfung der Voraussetzung N > 2
C
      MARK = -2
      IF (N .LT. 3) RETURN
C
C   Abfrage auf positive Definitheit und strenge Regularität
C   der Matrix A für N=1
C
      ROW = DABS(DM(1)) + DABS(DU(1)) + DABS(DU(N))
      IF (ROW .EQ. 0.0D0) THEN
         MARK = 0
         RETURN
      END IF
      D = 1.0D0/ROW
      IF (DM(1) .LT. 0.0D0) THEN
         MARK = -1
         RETURN
      ELSEIF (DABS(DM(1))*D .LE. EPS) THEN
         MARK = 0
         RETURN
      END IF
C
C   Durchführung der Zerlegung bei gleichzeitiger Abfrage
C   auf positive Definitheit und strenge Regularität der Matrix A
C
      DUMMY = DU(1)
      DU(1) = DU(1)/DM(1)
      CR(1) = DU(N)/DM(1)
      DO 20 I=2,N-1,1
         ROW = DABS(DM(I)) + DABS(DU(I)) + DABS(DUMMY)
         IF (ROW .EQ. 0.0D0) THEN
            MARK = 0
            RETURN
         END IF
         D = 1.0D0/ROW
         DM(I) = DM(I) - DUMMY * DU(I-1)
         IF (DM(I) .LT. 0.0D0) THEN
            MARK = -1
            RETURN
         ELSEIF (DABS(DM(I))*D .LE. EPS) THEN
            MARK = 0
            RETURN
         ENDIF
         IF (I .LT. (N-1)) THEN
            CR(I) = -DUMMY * CR(I-1)/DM(I)
            DUMMY = DU(I)
            DU(I) = DU(I)/DM(I)
         ELSE
            DUMMY2 = DU(I)
            DU(I) = (DU(I) - DUMMY * CR(I-1))/DM(I)
         ENDIF
   20 CONTINUE
      ROW = DABS(DU(N)) + DABS(DM(N)) + DABS(DUMMY2)
      IF (ROW .EQ. 0.0D0) THEN
         MARK = 0
         RETURN
      END IF
      D = 1.0D0/ROW
      DM(N) = DM(N) - DM(N-1) * DU(N-1) * DU(N-1)
      DUMMY = 0.0D0
      DO 30 I=1,N-2,1
         DUMMY = DUMMY + DM(I) * CR(I) * CR(I)
   30 CONTINUE
      DM(N) = DM(N) - DUMMY
      IF (DM(N) .LT. 0) THEN
         MARK = -1
         RETURN
      ELSEIF (DABS(DM(N))*D .LE. EPS) THEN
         MARK = 0
         RETURN
      ENDIF
      MARK = 1
      RETURN
      END
C
C
C
      SUBROUTINE CYTSYS (N,DM,DU,CR,RS,X)
C
C*****************************************************************
C                                                                *
C     Lösung eines linearen Gleichungssystems                    *
C                  A * X = RS                                    *
C     mit zyklisch tridiagonaler, symmetrischer, positiv         *
C     definiter Matrix A, nachdem die Zerlegungsmatrizen mit     *
C     Hilfe der SUBROUTINE CYTSYP berechnet worden sind.         *
C     Hier werden diese Zerlegungsmatrizen als Eingabematrizen   *
C     verwendet und in den drei 1-dim. Feldern DM, CR und DU     *
C     abgespeichert.                                             *
C                                                                *
C                                                                *
C     EINGABEPARAMETER:                                          *
C     =================                                          *
C     N    : Anzahl der Gleichungen, N > 2                       *
C     DM   : 1-dim. Feld (1:N); ) die Felder DU, DM und CR       *
C     DU   : 1-dim. Feld (1:N); ) enthalten die Zerlegung der    *
C     CR   : 1-dim. Feld (1:N); ) Matrix A                       *
C     RS   : 1-dim. Feld (1:N); rechte Seite                     *
C                                                                *
C                                                                *
C     AUSGABEPARAMETER:                                          *
C     =================                                          *
C     X    : 1-dim. Feld (1:N), das die Lösung des Gleichungs-   *
C            systems enthält                                     *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: keine                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Gisela Engeln-Müllges                             *
C  Datum     : 27.04.1988                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION DM(1:N),DU(1:N),CR(1:N),RS(1:N),X(1:N)
C
C  Vorwärtselimination
C
      DUMMY = RS(1)
      RS(1) = DUMMY/DM(1)
      SUM = CR(1)*DUMMY
      DO 10 I=2,N-1,1
         DUMMY = RS(I)-DU(I-1)*DUMMY
         RS(I) = DUMMY/DM(I)
         IF (I .NE. (N-1)) SUM = SUM + CR(I)*DUMMY
   10 CONTINUE
      DUMMY = RS(N)-DU(N-1)*DUMMY
      DUMMY = DUMMY-SUM
      RS(N) = DUMMY/DM(N)
C
C  Rückwärtselimination
C
      X(N) = RS(N)
      X(N-1) = RS(N-1)-DU(N-1)*X(N)
      DO 30 I=N-2,1,-1
         X(I) = RS(I)-DU(I)*X(I+1)-CR(I)*X(N)
   30 CONTINUE
      RETURN
      END
C
C
C
      INTEGER FUNCTION MACHPD(X)
C
C***************************************************************
C                                                              *
C     Berechnung der Maschinengenauigkeit                      * 
C                                                              *
C***************************************************************
C
      DOUBLE PRECISION X
      MACHPD=0
      IF (1.0D0 .LT. X) MACHPD=1
      RETURN
      END
C
C
C
      SUBROUTINE PSPVAL (TV,N,T,AX,BX,CX,DX,AY,BY,CY,DY,SX,SY)
C
C*****************************************************************
C                                                                *
C  Auswertungsprogramm für parametrische kubische Spline-        *
C  funktionen                                                    *
C  mit den Komponentenfunktionen SX(T), SY(T) in der Darstellung *
C                                                                *
C  SX := SX(T) = AX(I) + BX(I)(T-T(I)) + CX(I)(T-T(I))**2 +      *
C                                      + DX(I)(T-T(I))**3        *
C                                                                *
C  SY := SY(T) = AY(I) + BY(I)(T-T(I)) + CY(I)(T-T(I))**2 +      *
C                                      + DY(I)(T-T(I))**3        *
C                                                                *
C  für T Element von [T(I),T(I+1)], I=0(1)N-1.                   *
C                                                                *
C  Das Programm berechnet den Funktionswert der Komponentenfunk- *
C  tionen SX(T) und SY(T) an der Stelle T=TV.                    *
C                                                                *
C                                                                *
C  EINGABEPARAMETER:                                             *
C  =================                                             *
C  TV :  Stelle, an der die Splinefunktionen SX(T) und SY(T)     *
C        ausgewertet werden sollen.                              *
C  N  :  Nummer des letzten Knotens; T(I), I=0(1)N               *
C  T  :  1-dim. Feld (0:N) ;  T(I) = Knoten, I=0(1)N             *
C  AX :  1-dim. Feld (0:N) ]  Die Feldelemente 0 bis N-1         *
C  BX :  1-dim. Feld (0:N) ]  enthalten die Splinekoeffizienten  *
C  CX :  1-dim. Feld (0:N) ]  der Komponentenfunktion SX(T)      *
C  DX :  1-dim. Feld (0:N) ]                                     *
C  AY :  1-dim. Feld (0:N) ]  Die Feldelemente 0 bis N-1         *
C  BY :  1-dim. Feld (0:N) ]  enthalten die Splinekoeffizienten  *
C  CY :  1-dim. Feld (0:N) ]  der Komponentenfunktion SY(T)      *
C  DY :  1-dim. Feld (0:N) ]                                     *
C                                                                *
C                                                                *
C  AUSGABEPARAMETER:                                             *
C  =================                                             *
C  SX :  Funktionswert der Splinefunktion SX(T) an der Stelle TV *
C  SY :  Funktionswert der Splinefunktion SY(T) an der Stelle TV *
C                                                                *
C----------------------------------------------------------------*
C                                                                *
C  Benötigte Unterprogramme: SPVAL                               *
C                                                                *
C*****************************************************************
C                                                                *
C  Autor     : Günter Palm                                       *
C  Datum     : 01.06.1991                                        *
C  Quellcode : FORTRAN 77                                        *
C                                                                *
C*****************************************************************
C
C-----Deklarationen-----------------------------------------------
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DOUBLE PRECISION T(0:N), AX(0:N), BX(0:N), CX(0:N), DX(0:N),
     +                         AY(0:N), BY(0:N), CY(0:N), DY(0:N)
C
C-----Berechnung der Funktionswerte der Komponentenfunktionen-----
C     SX(T) und SY(T)
C
      SX = SPVAL (TV,N,T,AX,BX,CX,DX)
      SY = SPVAL (TV,N,T,AY,BY,CY,DY)
C
      RETURN
      END
C
C
C
