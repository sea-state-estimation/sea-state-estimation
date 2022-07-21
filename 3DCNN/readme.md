### 3DCNN.py General Idea:
Finding ocean characteristics (as regression) from video using a 3DCNN.
Architecture based on publications 
- S. Ji, W. Xu, M. Yang, and K. Yu, ‘‘3D convolutional neural networks for human action recognition’’ IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 1, pp. 221–231, Jan. 2010.
- H. Wu et al.: ‘‘iOceanSee: A Novel Scheme for Ocean State Estimation‘‘ IEEE Access, VOLUME 8, 2020


### Model Summary: see in model_summary.txt

### extract-data.py Idea:
Extract data from downloaded .csv-files into numpy-arrays


### Settings for download data from bsh:
https://seastate.bsh.de/rave/index.jsf?content=seegang
- Stations and parameters: 
  - Tick only 'DWR'
  - Tick under 'FINO3 Platform' only
    - 'VAVH - Average height highest 1/3 wave (H1/3) [m]' and
    - 'VAVT - Average period highest 1/3 wave (T1/3) [s]'
- Queries:
  - Start date:
    - Select the first date from the period, you want to download the data (day included)
  - End date:
    - Select the first date after the period, you want to download the data (day excluded)
  - Quality flags:
    - Do not change anything
  - Export:
    - Format: select 'CSV'	
    - Decimal separator: select 'Period'	
    - Line break: select 'Windows'
    - Partitioning:	select 'Daily'
    - Filename:	type a name, if you want
    - Email notification: tick, if you want
- press 'Request data'

Source to download the requested files:
https://seastate.bsh.de/rave/index.jsf?content=download
