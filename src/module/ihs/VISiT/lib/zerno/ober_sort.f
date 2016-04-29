C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_SORT(elfla_kno,elfla_ele,elfla_adr,nelfla,
     *                     nelfla_max,nkd_obe,elpar,permut,
     *                     help_kno,help_ele,help_num)
  
      implicit none     

      include 'common.zer'

      integer    elfla_adr,elfla_kno,elfla_ele,nelfla_max,nelfla,
     *           help_kno,help_ele,help_num,permut,elpar,nkd_obe

      integer i,k,kflag,igeb,anz,luerr

      dimension permut(nelfla_max),elpar(nelem_max),
     *          help_kno(nelfla_max,nkd_obe),help_ele(nelfla_max),
     *          help_num(nelfla_max)

      dimension elfla_adr(ngebiet+1),elfla_ele(nelfla_max),
     *          elfla_kno(nelfla_max,nkd_obe)
c     *****************************************************************


c     *****************************************************************
c     SORTIEREN DER ELEMENTRANDBEDINGUNGEN NACH DER GEBIETSNUMMER:

      do 100 i=1,nelfla
         permut(i)=i
         help_num(i)=elpar(elfla_ele(i))
         help_ele(i)=elfla_ele(i)
         do 101 k=1,nkd_obe
            help_kno(i,k)=elfla_kno(i,k)
 101     continue
 100  continue

      kflag=2
      call isort(help_num,permut,nelfla,kflag)

      do 200 i=1,nelfla
         elfla_ele(i)=help_ele(permut(i))
         do 201 k=1,nkd_obe
            elfla_kno(i,k)=help_kno(permut(i),k)
 201     continue
 200  continue
c     *****************************************************************

c     *****************************************************************
c     BESTIMMUNG DES ADRESSFELDES:

      elfla_adr(1)=1
      do 300 igeb=1,ngebiet
         anz=0
         do 310 i=1,nelfla
            if (elpar(elfla_ele(i)).eq.igeb) then
               anz=anz+1
            else if (elpar(elfla_ele(i)).gt.igeb) then
               goto 309
            endif
 310     continue

 309     continue
         elfla_adr(igeb+1)=elfla_adr(igeb)+anz

 300  continue
c     *****************************************************************


c     *****************************************************************
c     KONTROLLE:

      do 400 igeb=1,ngebiet
        do 410 i=elfla_adr(igeb),elfla_adr(igeb+1)-1
           if (elpar(elfla_ele(i)).ne.igeb) then
              call erro_init(myid,parallel,luerr)
              write(luerr,*)'Fehler in Routine OBER_SORT'
              write(luerr,*)'Das Adressfeld elfla_adr ist falsch belegt'
              write(luerr,*)'Das Oberflaechenelement mit der '
              write(luerr,*)'Nummer ',elfla_ele(i),' sollte eigentlich '
              write(luerr,*)'die Gebietsnummer ',igeb,' besitzen.'
              write(luerr,*)'Tatsaechlich besitzt das Element aber die'
              write(luerr,*)'Gebietsnummer ',elpar(elfla_ele(i))       
              call erro_ende(myid,parallel,luerr)
           endif
 410    continue
 400  continue
c     *****************************************************************

c     write(lupar,*)' Nr   geb      ele'
c     do 500 igeb=1,ngebiet
c       write(lupar,*)'Gebiet ',igeb 
c       do 510 i=elfla_adr(igeb),elfla_adr(igeb+1)-1
c          write(lupar,533) i,elpar(elfla_ele(i)),elfla_ele(i)
c510    continue
c500  continue
c533  format(4(i7,1x)) 

      return
      end

