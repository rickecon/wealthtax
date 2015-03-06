***********************************************************
* Piketty Wealth Tax Project
* 
***********************************************************
* This program takes SCF data and calculates some tabs
* on wealth by age
*
*
*************************************************************

#delimit ;
set more off;
capture clear all;
capture log close;
set memory 8000m;


set matsize 800 ;

use "rscfp2013.dta", clear ;
append using "rscfp2010.dta" ; /* append 2010 data  - note it's already in 2013 dollars*/
append using "rscfp2007.dta" ; /* append 2007 data  - note it's already in 2013 dollars*/


/* calculate the distribution of wealth by age */
collapse (p1) p1_wealth=networth (p2) p2_wealth=networth (p3) p3_wealth=networth (p4) p4_wealth=networth (p5) p5_wealth=networth (p6) p6_wealth=networth (p7) p7_wealth=networth  (p8) p8_wealth=networth  (p9) p9_wealth=networth  (p10) p10_wealth=networth
         (p11) p11_wealth=networth (p12) p12_wealth=networth (p13) p13_wealth=networth (p14) p14_wealth=networth (p15) p15_wealth=networth (p16) p16_wealth=networth (p17) p17_wealth=networth  (p18) p18_wealth=networth  (p19) p19_wealth=networth  (p20) p20_wealth=networth
         (p21) p21_wealth=networth (p22) p22_wealth=networth (p23) p23_wealth=networth (p24) p24_wealth=networth (p25) p25_wealth=networth (p26) p26_wealth=networth (p27) p27_wealth=networth  (p28) p28_wealth=networth  (p29) p29_wealth=networth  (p30) p30_wealth=networth
         (p31) p31_wealth=networth (p32) p32_wealth=networth (p33) p33_wealth=networth (p34) p34_wealth=networth (p35) p35_wealth=networth (p36) p36_wealth=networth (p37) p37_wealth=networth  (p38) p38_wealth=networth  (p39) p39_wealth=networth  (p40) p40_wealth=networth
         (p41) p41_wealth=networth (p42) p42_wealth=networth (p43) p43_wealth=networth (p44) p44_wealth=networth (p45) p45_wealth=networth (p46) p46_wealth=networth (p47) p47_wealth=networth  (p48) p48_wealth=networth  (p49) p49_wealth=networth  (p50) p50_wealth=networth
         (p51) p51_wealth=networth (p52) p52_wealth=networth (p53) p53_wealth=networth (p54) p54_wealth=networth (p55) p55_wealth=networth (p56) p56_wealth=networth (p57) p57_wealth=networth  (p58) p58_wealth=networth  (p59) p59_wealth=networth  (p60) p60_wealth=networth
         (p61) p61_wealth=networth (p62) p62_wealth=networth (p63) p63_wealth=networth (p64) p64_wealth=networth (p65) p65_wealth=networth (p66) p66_wealth=networth (p67) p67_wealth=networth  (p68) p68_wealth=networth  (p69) p69_wealth=networth  (p70) p70_wealth=networth
         (p71) p71_wealth=networth (p72) p72_wealth=networth (p73) p73_wealth=networth (p74) p74_wealth=networth (p75) p75_wealth=networth (p76) p76_wealth=networth (p77) p77_wealth=networth  (p78) p78_wealth=networth  (p79) p79_wealth=networth  (p80) p80_wealth=networth
         (p81) p81_wealth=networth (p82) p82_wealth=networth (p83) p83_wealth=networth (p84) p84_wealth=networth (p85) p85_wealth=networth (p86) p86_wealth=networth (p87) p87_wealth=networth  (p88) p88_wealth=networth  (p89) p89_wealth=networth  (p90) p90_wealth=networth
         (p91) p91_wealth=networth (p92) p92_wealth=networth (p93) p93_wealth=networth (p94) p94_wealth=networth (p95) p95_wealth=networth (p96) p96_wealth=networth (p97) p97_wealth=networth  (p98) p98_wealth=networth  (p99) p99_wealth=networth, by(age) ;


format * %23.5f ;
outsheet using "scf2007to2013_wealth_age_all_percentiles.csv", comma replace ;


capture log close ;
