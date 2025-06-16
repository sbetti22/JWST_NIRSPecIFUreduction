# JWST_NIRSPecIFUreduction

JWST NIRSpec IFU data reduction, reference differential imaging, and disk modeling code for disk reflectance spectra.  
Based in part on jwstIFURDI (https://github.com/ChenXie-astro/jwstIFURDI/) 

author: Sarah Betti

## Installation 
```
% pip install git+https://github.com/sbetti22/JWST_NIRSPecIFUreduction.git#egg=JWST_NIRSPecIFUreduction
```

## Requirements
  - numpy
  - scipy
  - pathlib
  - matplotlib
  - scikit-image
  - [jwst](https://jwst-pipeline.readthedocs.io/en/latest/)
  - [h5py](https://docs.h5py.org/en/latest/index.html)
  - [astropy](https://www.astropy.org)
  - [vip_hci](https://vip.readthedocs.io/en/latest/index.html)

## To Run
The example notebook ```data_reduction_betapic.ipynb``` shows how you can use the package to run your data through the pipeline.  For each step, a ```run_<step>``` function is called.  For more control, you can utilize the individual functions that are called within the ```run_<step>``` function to individualize your reduction. 

## Credits
Please also cite [Xie et al. 2025](https://www.nature.com/articles/s41586-025-08920-4) when using this code, if you used the centering, masking, or PSF subtraction.  

```
@article{Xie2025,
	author = {Xie, Chen and Chen, Christine H. and Lisse, Carey M. and Hines, Dean C. and Beck, Tracy and Betti, Sarah K. and Pinilla-Alonso, Noem{\'\i} and Ingebretsen, Carl and Worthen, Kadin and G{\'a}sp{\'a}r, Andr{\'a}s and Wolff, Schuyler G. and Bolin, Bryce T. and Pueyo, Laurent and Perrin, Marshall D. and Stansberry, John A. and Leisenring, Jarron M.},
	date = {2025/05/01},
	date-added = {2025-05-14 11:29:27 -0400},
	date-modified = {2025-05-14 11:29:27 -0400},
	doi = {10.1038/s41586-025-08920-4},
	id = {Xie2025},
	isbn = {1476-4687},
	journal = {Nature},
	number = {8063},
	pages = {608--611},
	title = {Water ice in the debris disk around HD 181327},
	url = {https://doi.org/10.1038/s41586-025-08920-4},
	volume = {641},
	year = {2025},
	bdsk-url-1 = {https://doi.org/10.1038/s41586-025-08920-4}}
}
