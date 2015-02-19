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
cd "~/Econ/Research/PikettyWealthTax/data/" ;
log using "~/Econ/Research/PikettyWealthTax/data/scf_wealth_tabs.log", replace ;

local datapath "~/Econ/Research/PikettyWealthTax/data/" ;


set matsize 800 ;

use "`datapath'/rscfp2013.dta", clear ;
append using "`datapath'/rscfp2010.dta" ; /* append 2010 data  - note it's already in 2013 dollars*/
append using "`datapath'/rscfp2007.dta" ; /* append 2007 data  - note it's already in 2013 dollars*/


/* calculate the distribution of wealth by age */
collapse (mean) mean_wealth=networth (sd) sd_wealth=networth (median) median_wealth=networth (p10) p10_wealth=networth (p90) p90_wealth=networth (p95) p95_wealth=networth (p96) p96_wealth=networth (p98) p98_wealth=networth (p99) p99_wealth=networth (count) num_obs=networth[aweight=wgt], by(age) ;
format * %23.5f ;
outsheet using "`datapath'/scf2007to2013_wealth_age.csv", comma replace ;


replace mean_wealth = mean_wealth/1000 ;
replace median_wealth = median_wealth/1000 ;
replace p10_wealth = p10_wealth/1000 ;
replace p90_wealth = p90_wealth/1000 ;
replace p90_wealth = p90_wealth/1000 ;
replace p95_wealth = p95_wealth/1000 ;
replace p96_wealth = p96_wealth/1000 ;
replace p98_wealth = p98_wealth/1000 ;
replace p99_wealth = p99_wealth/1000 ;
format * %23.0f ;
/* plot some graphs of the life-cycle profile of wealth */
/* plot life-cycle profiles in dollas per hour */
twoway (connected mean_wealth age, msize(small) mcolor(blue) lcolor(blue) msymbol(D))
(connected p10_wealth age, msize(small) mcolor(red) lcolor(red) msymbol(O))
(connected median_wealth age, msize(small) mcolor(orange) lcolor(organge) msymbol(S))
(connected p90_wealth age, msize(small) mcolor(green) lcolor(green) msymbol(T)),
	title("Life-Cylce Wealth Profiles") 
	xtitle("Age") 
	ytitle("Wealth (1000s of 2013$)")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) */
	ylabel(0(1000)3000,grid)
	legend(label(1 "Mean")) 
	legend(label(2 "10th Percentile")) 
	legend(label(3 "50th Percentile")) 
	legend(label(4 "90th Percentile")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`datapath'/LCP_wealth_scf.pdf", replace;

/* plot some graphs of the life-cycle profile of wealth */
/* plot life-cycle profiles in dollas per hour */
twoway (connected mean_wealth age, msize(small) mcolor(blue) lcolor(blue) msymbol(D))
(connected p95_wealth age, msize(small) mcolor(red) lcolor(red) msymbol(O))
(connected p96_wealth age, msize(small) mcolor(orange) lcolor(organge) msymbol(S))
(connected p98_wealth age, msize(small) mcolor(black) lcolor(black) msymbol(X))
(connected p99_wealth age, msize(small) mcolor(green) lcolor(green) msymbol(T)),
	title("Life-Cylce Wealth Profiles") 
	xtitle("Age") 
	ytitle("Wealth (1000s of 2013$)")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) */
	ylabel(0(1000)3000,grid)
	legend(label(1 "Mean")) 
	legend(label(2 "95th Percentile")) 
	legend(label(3 "96th Percentile")) 
	legend(label(4 "98th Percentile")) 
	legend(label(5 "99th Percentile")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`datapath'/LCP_wealth_scf_top5.pdf", replace;

capture log close ;