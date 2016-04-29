C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                     coord_num,lnods,typ_name)

      implicit none

      include 'common.zer'

      integer el_num,help1(8),coord_zeig,lnods_zeig,
     *        lnods,coord_num,
     *        luerr,nnn,j,k,el_adr,kn_num,node

      character*35 typ_name

      dimension coord_zeig(knmax_num),lnods_zeig(elmax_num),
     *          coord_num(npoin_max),lnods(nelem_max,nkd)
c     ****************************************************************
 
c      write(*,*)'in rel_check'

c     ****************************************************************
c     KONTROLLE DER ELEMENT-NUMMER:

      if (el_num.gt.elmax_num .or. lnods_zeig(el_num).eq.0) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'FEHLER IN SUBROUTINE REL_CHECK          '
         write(luerr,*)'Fehler bei ',typ_name                       
         write(luerr,*)'Die angegebene Elementnummer ', el_num
         write(luerr,*)'existiert nicht im Geometrie-File.'
         write(luerr,*)'Max. Elementnummer im Geometrie-File:',elmax_num
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************

c     ****************************************************************
c     KONTROLLE DER KNOTEN-NUMMERN:

      do 100 j=1,nrbknie
         kn_num=help1(j)
         if (kn_num.gt.knmax_num .or. coord_zeig(kn_num).eq.0) THEN
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'FEHLER IN SUBROUTINE REL_CHECK          '
             write(luerr,*)'Fehler bei ',typ_name                       
             write(luerr,*)'Die angegebene Knotennummer ', kn_num
             write(luerr,*)'existiert nicht im Geometrie-File.'
             write(luerr,*)'Max. Knotennummer im Geometrie-File:',
     *                      knmax_num
             call erro_ende(myid,parallel,luerr)
         endif
 100  continue
c     ****************************************************************

c     ****************************************************************
c     KONTROLLE OB ALLE KNOTEN IM ANGEGEBENEN ELEMENT ENTHALTEN SIND:

      el_adr=lnods_zeig(el_num)
      nnn=0
      do 200 j=1,nrbknie
          node=coord_zeig(help1(j))
          do 210 k=1,nkd
            if (node.eq.lnods(el_adr,k)) then
               nnn=nnn+1
               goto 201
            endif
 210      continue
 201      continue
 200  continue

      if (nnn.ne.nrbknie) then
          call erro_init(myid,parallel,luerr)
          write(luerr,*)'FEHLER IN SUBROUTINE REL_CHECK          '
          write(luerr,*)'Fehler bei ',typ_name                       
          write(luerr,*)'Nicht alle Knoten der Randbedingung sind '
          write(luerr,*)'im dazugehoerigen Element des Geometrie-Files'
          write(luerr,*)'enthalten.                        '
          write(luerr,*)'Elementnummer:',el_num                
          write(luerr,*)'Randknoten   :',(help1(j),j=1,nrbknie)       
          write(luerr,*)'Elementknoten:',
     *                  (coord_num(lnods(el_adr,j)),j=1,nkd)
          call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************
      return
      end
