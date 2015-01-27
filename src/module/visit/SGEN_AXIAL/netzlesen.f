c ----------------------------------------------------------------------
c Einlesen der Rechennetzes aus Geo-File
c ----------------------------------------------------------------------

      SUBROUTINE netzoeffnen()

      implicit none

c ----------------------------------------------------------------------
c LESE GEO_FILE
c ----------------------------------------------------------------------
 
      call LESE_3DGEO()


c ----------------------------------------------------------------------
c LESE RB_FILE
c ----------------------------------------------------------------------
c      call LESE_3DRB(an3_kr,e(ikrb1),an3_kw,f(ikrbx),
c     #  f(ikrby),f(ikrbz),f(ikrbk),f(ikrbe),an3_wr,e(iwrb1),e(iwrb2),
c     #  e(iwrb3),e(iwrb4),e(iwrb5),an3_kb,e(ikbi1),e(ikbi2),e(ikbi3),
c     #  e(ikbi4),e(ikbi5),e(ikbi6),an3_km,e(ikma1),e(ikma2))

c ----------------------------------------------------------------------
c PROGRAMM ENDE
c ----------------------------------------------------------------------

      end









c ----------------------------------------------------------------------
c LESE_3DGEO
c ----------------------------------------------------------------------
      subroutine LESE_3DGEO()
c
c Liest ein 3D-Geofile ein
c
c EINGABE
c x,y,z.....Koordinaten der Knoten
c anz_kno...Anzahl der Knoten
c t.........Anzahl der Radialschnitte
c el1..el8..Knotennummern je 3D-Element
c anz_elm...Anzahl der Elemente
c
      parameter (DIM_F=1500000,DIM_E=3000000)

      common /ver2/  datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
      common /akno/ an_kno, bi, ixkn, iykn, izkn
      common /anet/ f(DIM_F),e(DIM_E)
c
      integer an_kno,bi,ixkn,iykn,izkn
      integer a_3Del,iel1,iel2,iel3,iel4,iel5,iel6,iel7,iel8
      integer i
      integer anz_kno_mal_t
      integer zero
      real f(DIM_F)
      integer e(DIM_E)
c     
      character*200 datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb

c
c                  
      open(16,file=datei_kart3d_geo)
c      
      do i=1,10,1
       read(16,*) 
      end do
      read(16,*) anz_kno_mal_t,a_3Del

      an_kno=anz_kno_mal_t
      bi=1

      ixkn=1
      iykn=ixkn+an_kno
      izkn=iykn+an_kno

      iel1=1
      iel2=iel1+a_3Del
      iel3=iel2+a_3Del
      iel4=iel3+a_3Del
      iel5=iel4+a_3Del
      iel6=iel5+a_3Del
      iel7=iel6+a_3Del
      iel8=iel7+a_3Del

c
c Eingabe der Knoten      
      do i=1,anz_kno_mal_t,1
       read(16,*) zero,f(ixkn+i-1),f(iykn+i-1),f(izkn+i-1)
      end do
c
c Eingabe der Elemente
      do i=1,anz_elm
       read(16,*) e(iel1+i-1),e(iel2+i-1),e(iel3+i-1),e(iel4+i-1),
     #  e(iel5+i-1),e(iel6+i-1),e(iel7+i-1),e(iel8+i-1)
      end do
c
      close(16)
c
      end














c ----------------------------------------------------------------------
c CRE_3DRB
c ----------------------------------------------------------------------
      subroutine LESE_3DRB(an3_kr,krb1,an3_kw,krbx,krby,krbz,krbk,
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

c 
      integer an3_kr,krb1(an3_kr),an3_kw,an3_wr,wrb1(an3_wr),
     # wrb2(an3_wr),wrb3(an3_wr),wrb4(an3_wr),wrb5(an3_wr),an3_kb,
     # kbi1(an3_kb),kbi2(an3_kb),kbi3(an3_kb),kbi4(an3_kb),kbi5(an3_kb),
     # kbi6(an3_kb),an3_km,kma1(an3_km),kma2(an3_km),i
     
      integer zero,eins,zwei,drei,vier,fuenf
c
      real krbx(an3_kw),krby(an3_kw),krbz(an3_kw),krbk(an3_kw),
     # krbe(an3_kw)
c

      character*200 datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb
      common /ver2/ datei_ax_stf, datei_kart3d_geo, datei_kart3d_rb

c
c
      open(21,file=datei_kart3d_rb)
c
      do i=1,10,1
       read(21,*) 
      end do
c  
      print*, '1'
      read(21,*) an3_kr,an3_wr,zero,zero,zero,zero,an3_kb,an3_km
      an3_kr=an3_kr/5
      an3_kw=an3_kr
      print*, an3_kb

c      integer krb1(an3_km)

c
      
       do i=1,an3_kr,1
        read(21,*) 
        read(21,*) 
        read(21,*) 
        read(21,*) 
        read(21,*) 
 
c        print*, '1,5'
c        read(21,*) krb1(i)
c	print*, '1,5'

c        read(21,*) krb1(i),zwei,krby(i)
c	print*, '1,5'

c        read(21,*) krb1(i),drei,krbz(i)
c        read(21,*) krb1(i),vier,krbk(i)
c        read(21,*) krb1(i),fuenf,krbe(i)
c	print*, '1,5'
       end do
      print*, '2'

c
      if (an3_wr.ne.0) then
       do i=1,an3_wr,1
        read(21,*) wrb1(i),wrb2(i),wrb3(i),wrb4(i),
     #             zero,zero,zero,wrb5(i)
       end do
      endif
c
      print*, '3'

      if (an3_kb.ne.0) then
       do i=1,an3_kb,1
        read(21,*) kbi1(i)
c       ,kbi2(i),kbi3(i),kbi4(i),kbi5(i),kbi6(i)
       end do
      endif
c
      print*, '4'

      if (an3_km.ne.0) then
       do i=1,an3_km,1
        read(21,*) 
c          kma1(i)
c        ,kma2(i)
       end do
      end if
      print*, '5'
c
      end      
