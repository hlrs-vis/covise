C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE KERN_DATA(knpar,coord_num,lnods,
     *                     kern_kn,kern_kn_adr,
     *                     kern_el,kern_el_adr,nkern_max)

      implicit none     

      include 'common.zer'

      integer  knpar,coord_num,lnods,
     *         kern_kn,kern_kn_adr,
     *         kern_el,kern_el_adr,
     *         nkern_max

      integer i,k,nnn,igeb,nkn,nel,ikn,luerr,ncor,
     *        nkern_kn,nkern_el

      dimension knpar(npoin_max),coord_num(npoin_max),
     *          lnods(nelem_max,nkd)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)
c     *****************************************************************


c     *****************************************************************
c     BESTIMMUNG DER KERN-KNOTEN PRO GEBIET:

      kern_kn_adr(1)=1
      nkern_kn=0
      do 100 igeb=1,ngebiet
         nnn=0
         do 110 i=1,npoin
            if (knpar(i).eq.igeb) then
               nnn=nnn+1
               nkern_kn=nkern_kn+1
               kern_kn(nkern_kn)=i   
            endif
 110     continue 

         kern_kn_adr(igeb+1)=kern_kn_adr(igeb)+nnn

 100  continue 
c     *****************************************************************

c     **************************************************************
c     BESTIMMUNG DER KERN-ELEMENTE:

      kern_el_adr(1)=1 

      nkern_el=0
      do 400 igeb=1,ngebiet

         ncor=0
         do 410 i=1,nelem
            nnn=0
            do 420 k=1,nkd
               ikn=lnods(i,k)
               if (knpar(ikn).eq.igeb) then
                  nnn=nnn+1
               endif
 420        continue 

            if (nnn.eq.nkd) then
               ncor=ncor+1
               nkern_el=nkern_el+1
               kern_el(nkern_el)=i
            endif 
 410     continue

         kern_el_adr(igeb+1)=kern_el_adr(igeb)+ncor

 400  continue
c     **************************************************************


c     **************************************************************
c     DIMENSIONSKONTROLLEN:

      if (nkern_kn.gt.nkern_max.or.nkern_el.gt.nkern_max) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine KERN_DATA'
           write(luerr,*)'Die zuvor bestimmte maximale Anzahl an '
           write(luerr,*)'Kern-Daten ist falsch.                 '
           write(luerr,*)'Bestimmte  Anzahl nkern_max:',nkern_max
           write(luerr,*)'Benoetigte Anzahl nkern_kn :',nkern_kn 
           write(luerr,*)'Benoetigte Anzahl nkern_el :',nkern_el 
           call erro_ende(myid,parallel,luerr)
      endif

      if (nkern_kn.ne.npoin) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine KERN_DATA'
           write(luerr,*)'Die Gesamtanzahl Kernknoten muss gleich'
           write(luerr,*)'der eingelesenen Knotenanzahl sein.    '
           write(luerr,*)'Gesamtanzahl Kernknoten :',nkern_kn      
           write(luerr,*)'Eingelesene Knotenanzahl:',npoin        
           call erro_ende(myid,parallel,luerr)
      endif
c     **************************************************************


c     **************************************************************
c     AUSDRUCK:
      
c     write(lupar,*)'             '         
c     write(lupar,*)'GEBIETS-DIMENSIONEN:'         
c     write(lupar,*)'Gebiet       nkern_kn     nkern_el'
c     do 750 igeb=1,ngebiet
c           nkn=kern_kn_adr(igeb+1)-kern_kn_adr(igeb)
c           nel=kern_el_adr(igeb+1)-kern_el_adr(igeb)
c           write(lupar,766) igeb,nkn,nel 
c750  continue
c766  format(i4,10x,i7,6x,i7)
 
c     write(lupar,*)'             '         
c     write(lupar,*)'KNOTEN-DATEN:'         
c     do 700 igeb=1,ngebiet
c        write(lupar,*)'Kern-Knoten von Gebiet',igeb
c        do 710 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
c           write(lupar,777) coord_num(kern_kn(i)) 
c710     continue
c700  continue
 
 
c     write(lupar,*)'             '         
c     write(lupar,*)'ELEMENT-DATEN:'         
c     do 701 igeb=1,ngebiet
c        write(lupar,*)'Kern-Elemente von Gebiet',igeb
c        do 711 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
c           write(lupar,777) kern_el(i)
c711     continue
c701  continue
c777  format(i7,5x,3(i7,1x))
 
c     write(6,*)'Stop in KERN_DATA'
c     stop
c     **************************************************************

      return
      end
