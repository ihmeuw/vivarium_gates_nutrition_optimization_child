**v0.14.0 - 10/23/26**

 - Port the LBWSG PAF simulation (``data/lbwsg_paf.yaml``) to vivarium-public-health 6
 - Weight each sex and neonatal age group's LBWSG PAF by that group's own exposure instead of exposure summed over both sexes and both neonatal age groups
 - Read the LBWSG PAF key from tracked simulation outputs, putting input draw k in artifact draw k, and regenerate Ethiopia's PAFs on the GBD 2023 artifact

**v0.13.3 - 10/22/26**

 - Cap pyarrow below 26 so the simulation environment can import it next to numpy 1.26 and psimulate can write results
 - Require vivarium_cluster_tools 4.5.0 for the cluster extra

**v0.13.2 - 10/22/26**

 - Compute the proportion of pregnant women with hemoglobin below 100 g/L from a GBD pull instead of a stored CSV

**v0.13.1 - 10/21/26**

 - Fix LBWSG Ordering
 - Use real worse-MAM data instead of data values

**v0.13.0 - 10/20/26**

 - Use the pytest-vivarium plugin for the shared test options and markers

**v0.12.3 - 10/16/26**

 - Migrate artifact building off ``get_draws`` to vivarium_gbd_access 7 and vivarium_inputs 9.1.1 (the first release that fills in the TMREL residual exposure category); fix the LBWSG exposure key and restrict it to the neonatal age groups
 - Drop the duplicated late-neonatal rows from the wasting relative-risk CSV before the neonatal RR-1 fill (MIC-7638)
 - Adopt the shared environment tooling; raise the floors to vivarium-engine 5.11.0 and vivarium-public-health 6.6.4; describe every pipeline producer and modifier
 - Delete the unused GBD 2021 data helpers and fix subnational location resolving
 - Fix ``--national`` artifact builds: accept a location name in ``get_national_location_id``

**v0.12.1 - 02/05/25**

 - Add python versions file

**v0.12.0 - 01/09/24**

 - Wasting transitions among 1-5 months, including LBWSG-dependent initialization
 - MAM treatment also targeted to "worse" MAM category
 - Replicate model for Nigeria and Pakistan
 - MMS shift and wasting transition rate data updates

**v0.9.0 - 10/12/23**

 - Add MAM targeting scenario

**v0.8.0 - 09/18/23**

 - Update effect of BEP on birthweight to account for maternal BMI status

**v0.7.1 - 09/18/23**

 - Remove explicit support for Python 3.7 and 3.8
 - Refactor all components to subclass Component 

**v0.7.0 - 09/18/23**

 - Add SQ-LNS intervention

**v0.6 - 09/18/23**

 - Update wasting exposure model (use transition rate data and new coverage/effectiveness values)

**v0.5.3 - 09/15/23**

 - Fix CGF PAF csv

**v0.5.2 - 09/14/23**

 - CGF Risk Effects Bug Fixes: Fix CGF Relative Risks

**v0.5.1 09/13/23**

 - CGF Risk Effects Bug Fixes: Include Underweight

**v0.5.0 - 09/07/23**

 - Update CGF Risk Effects

**v0.4.1 - 09/07/23**

 - Use updated underweight exposure distribution data (lookup.csv)

**v0.3.2 - 09/06/23**

 - Update malaria EMR to be calculated instead of taken from GBD

**v0.3.1 - 09/01/23**

 - Update malaria prevalence to be calculated instead of taken from GBD

**v0.4.0 - 09/01/23**

 - Add underweight exposure

**v0.3.0 - 08/30/23**

 - Include malaria

**v0.2.0 - 08/29/23**

 - Add Dynamic Child Wasting Model with GBD 2021 data
 - Re-bin to 2021 age groups 

**v0.1.1 - 08/24/23**

 - Include effects of antenatal supplementation on gestational age

**v0.1.0 - 08/21/23**

 - Replicate IV iron child model

**v0.0.0 - 08/07/23**

 - Initial release
