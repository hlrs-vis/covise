C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_ADR(lnods,elmat,elmat_adr,elmat_stu,nl_elmat, 
     *                    elfla_kno,elfla_ele,nelfla,
     *                    nelfla_max,nkd_obe,
     *                    elpar,zeiger,kern_ele,kern_adr,
     *                    coord,coord_num)
  
      implicit none     

      include 'mpif.h'
      include 'common.zer'

      integer lnods,elmat,elmat_adr,elmat_stu,nl_elmat, 
     *        elfla_kno,elfla_ele,nelfla,nelfla_max,nkd_obe,
     *        elpar,zeiger,kern_ele,kern_adr,
     *        coord_num

      integer i,j,k,kk,iii,ifla,ielem,inach,luerr,
     *        nnn,nfla_frei,nfla_proz,nfla_alle,igeb,
     *        anz_frei,fla_1(6,4),anz

      real    coord

      dimension lnods(nelem_max,nkd),elpar(nelem_max),
     *          elmat(nl_elmat),elmat_adr(nelem+1),
     *          elmat_stu(nelem+1),
     *          elfla_kno(nelfla_max,nkd_obe),elfla_ele(nelfla_max),
     *          zeiger(npoin),kern_ele(nelem),kern_adr(ngebiet+1),
     *          coord(npoin_max,ncd),coord_num(npoin_max)
c     *****************************************************************


c     *****************************************************************
c     BERECHNEN DER FREIEN FLAECHEN DER KERNELEMENTE:

c     Initialisierungen:                 
      do 40 i=1,npoin
         zeiger(i)=0
 40   continue

      do 142 i=1,nelfla_max
        elfla_ele(i)=0
        do 143 j=1,nrbknie
           elfla_kno(i,j)=0
 143     continue
 142  continue

      nfla_alle=0
      nfla_frei=0
c     goto 199
      do 100 ielem=1,nelem
         nnn=elmat_stu(ielem)-elmat_adr(ielem)+1

         IF (nnn.lt.nkd_fla) THEN
c           Das Element ielem besitzt freie Flaechen.
            anz_frei=nkd_fla-nnn

c           Flaechen von Element ielem:
            do 101 ifla=1,nkd_fla
              do 102 k=1,nrbknie
                   fla_1(ifla,k)=lnods(ielem,fla_zeig(ifla,k))
 102          continue 
 101        continue

            anz=0
c           Schleife ueber alle Flaechen:
            do 110 ifla=1,nkd_fla

c              Schleife ueber die Nachbarelemente von Element ielem:
               do 120 j=elmat_adr(ielem),elmat_stu(ielem)

                   inach=abs(elmat(j))

c                  Markieren der Knoten von Nachbarelement inach:
                   do 130 k=1,nkd
                       zeiger(lnods(inach,k))=1
 130               continue

                   nnn=0
                   do 140 k=1,nrbknie
                      if (zeiger(fla_1(ifla,k)).eq.1) then
                         nnn=nnn+1
                      endif
 140               continue 

c                  Loeschen der markierten Knoten von Element inach:
                   do 150 k=1,nkd
                       zeiger(lnods(inach,k))=0
 150               continue

                   if (nnn.eq.nrbknie) then
c                    Flaeche ifla ist in Element inach enthalten
c                    Untersuche naechste Flaeche:
                     goto 109
                   endif
 120           continue

c              Flaeche ifla ist in keinem Nachbarelement enthalten:
               nfla_alle=nfla_alle+1
               nfla_frei=nfla_frei+1
               anz=anz+1
               elfla_ele(nfla_alle)=ielem
               do 160 k=1,nrbknie
                  elfla_kno(nfla_alle,k)=fla_1(ifla,k)
 160           continue

 109        continue
 110        continue

            if (anz.ne.anz_frei) then
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler in Routine OBER_ADR '
               write(luerr,*)'Element ',ielem,' besitzt ',anz_frei
               write(luerr,*)'frei Flaechen. Bei der Berechnung wurden'
               write(luerr,*)'aber ',anz,' freie Flaechen    '
               write(luerr,*)'festgestellt.                      '
               write(luerr,*)'anz     =',anz            
               write(luerr,*)'anz_frei=',anz_frei
               write(luerr,*)'nnn     =',elmat_stu(ielem)-
     *                                   elmat_adr(ielem)+1
               write(luerr,*) (elmat(k),k=elmat_adr(ielem),
     *                        elmat_stu(ielem))
               call erro_ende(myid,parallel,luerr)
            endif
         ENDIF 

 100  continue                     
c199  continue
c     *****************************************************************



c     *****************************************************************
c     BERECHNEN DER FLAECHEN ZWISCHEN KERN- UND UEBERLAPPELEMENTEN:

c     Initialisieren des Hilfsfeldes:
      do 510 i=1,npoin
         zeiger(i)=0
 510  continue

      nfla_proz=0

c     goto 599
      do 500 igeb=1,ngebiet

         do 200 iii=kern_adr(igeb),kern_adr(igeb+1)-1

            ielem=kern_ele(iii)

c           Flaechen von Element ielem:
            do 201 ifla=1,nkd_fla
              do 202 k=1,nrbknie
                   fla_1(ifla,k)=lnods(ielem,fla_zeig(ifla,k))
 202          continue 
 201        continue

c           Schleife ueber die Nachbarelemente von Kernelement ielem:
            do 210 i=elmat_adr(ielem),elmat_stu(ielem)
                 inach=elmat(i)

                 IF (elpar(inach).ne.igeb) THEN
c                   Das Element inach grenzt an das Kernelement an

c                    Markieren der Knoten von Element inach:
                     do 220 k=1,nkd
                        zeiger(lnods(inach,k))=1
 220                 continue

c                    Schleife ueber alle Flaechen von Kernelement ielem
                     do 230 ifla=1,nkd_fla
                        nnn=0
                        do 240 k=1,nrbknie

                           if (zeiger(fla_1(ifla,k)).eq.1) then
                              nnn=nnn+1
                           endif
 240                    continue

                        if (nnn.eq.nrbknie) then
c                          Flaeche ifla ist in Element inach enthalten
                           nfla_alle=nfla_alle+1
                           nfla_proz=nfla_proz+1
                           elfla_ele(nfla_alle)=ielem
                           do 250 k=1,nrbknie
                              elfla_kno(nfla_alle,k)=fla_1(ifla,k)
 250                       continue
   
c                          Verlasse Schleife 230
                           goto 231
   
                        endif
 230                 continue

                     call erro_init(myid,parallel,luerr)
                     write(luerr,*)'Fehler in Routine OBER_ADR '
                     write(luerr,*)'Element ',ielem,' grenzt stumpf an '
                     write(luerr,*)'das Ueberlappelement ',inach,' an. '
                     write(luerr,*)'Aber es konnte keine gemeinsame  '
                     write(luerr,*)'Flaeche gefunden werden.      '
                     write(luerr,*)'Knoten von Element ',ielem,':'
                     write(luerr,299) (lnods(ielem,kk),kk=1,nkd)
                     write(luerr,*)'Knoten von Element ',inach,':'
                     write(luerr,299) (lnods(inach,kk),kk=1,nkd)
                     call erro_ende(myid,parallel,luerr)
 231                 continue
 299                 format(8(i7,1x))

c                    Loeschen der Markierung von Element inach:       
                     do 203 k=1,nkd
                        zeiger(lnods(inach,k))=0
 203                 continue

                 ENDIF
 210        continue

 200     continue                     

 500  continue

c599  continue
c     *****************************************************************


c     *****************************************************************
c     DIMENSIONSKONTROLLE:

      if (nfla_alle.ne.nelfla) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine OBER_ADR'
         write(luerr,*)'Die in Routine OBER_DIM berechnet Anzahl an'
         write(luerr,*)'Oberflaechenelementen stimmt mit der in Routine'
         write(luerr,*)'OBER_ADR geschriebenen Anzahl nicht ueberein.  '
         write(luerr,*)'Berechnete   Anzahl:',nelfla         
         write(luerr,*)'Geschriebene Anzahl:',nfla_alle
         call erro_ende(myid,parallel,luerr)
      endif
c     *****************************************************************


      return
      end

