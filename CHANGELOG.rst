**v0.13.0 - 10/05/26**

 - Pin vivarium_inputs 9.x and vivarium_gbd_access 7.x in the data extra; add the ``[tool.uv]`` override block
 - Import from vivarium.gbd_mapping instead of the removed gbd_mapping shim
 - Delete the unused GBD 2021 data-processing helpers from data/utilities.py and ``_load_em_from_meid`` from the loader
 - Select subnational locations with ``gbd.get_most_detailed_locations`` instead of a substring match on ``path_to_top_parent``
 - Let ``get_national_location_id`` accept a location name, id, or list of ids and walk parents with ``utility_data.get_location_id_parents``; fixes ``--national`` artifact builds
 - Replace the deprecated ``utility_data.get_location_id`` with ``resolve_location`` / ``resolve_locations``
 - Add unit tests for ``fetch_subnational_ids`` and ``get_national_location_id``

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
