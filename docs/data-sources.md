# Data sources

The frozen model uses Lake Erie observations from 2015–2020 for development and
2021 for validation and threshold selection. Its input window covers the seven
complete days before each target date. The examples distributed here are synthetic.
No measured observation rows or row-level derivatives are included.

## Stone Laboratory / Ohio Sea Grant

The exact source is `waterqualitydata2013-2022.xlsx`, available from the
[Stone Lab water-quality page](https://ohioseagrant.osu.edu/research/live/water)
through its [2013–2022 workbook download](https://ohioseagrant.osu.edu/download/h6hnx).
The workbook's SHA-256 is
`07475cd1600cf169f1c9e35fd066f73938b0abf76f75448041b7f8c98715cdb6`.
The model uses the `SL Buoy` and misspelled `SL Bouy` program records for
laboratory variables and measured total microcystin labels.

Archive citation: Chaffin, Justin (2023). *Total microcystins, chlorophyll, and
other water quality data collected in Lake Erie*, NCEI 0276941,
[doi:10.25921/7c57-gf98](https://doi.org/10.25921/7c57-gf98), SL Buoy subset for
2015–2021. The current archive extends beyond the original workbook's coverage;
use the workbook and checksum above for this release.

No dataset-specific open redistribution license was identified for this exact
workbook. Obtain it from the provider and retain its terms and citation.

## OSUGI / OSU Gibralter Island Buoy

Ohio State University collected the sensor observations, assembled by the Great
Lakes Observing System (GLOS). Original monthly NetCDF files are available from
the [NOAA NCEI OSUGI archive](https://www.ncei.noaa.gov/data/oceans/ioos/glos/OSUGI/).
[GLOS ERDDAP obs_56](https://seagull-erddap.glos.org/erddap/info/obs_56/index.html)
provides station metadata and a separate access interface.

Archive citation: Smith, Joeseph and GLOS (2015),
[NCEI 0130545](https://www.ncei.noaa.gov/archive/accession/0130545), OSUGI subset
for 2015–2021, within the GLOS collection
[doi:10.25921/wjq2-ty20](https://doi.org/10.25921/wjq2-ty20).

The OSUGI metadata permits free use and redistribution with stated limitations
concerning legal use and warranties. This software provides
[source URLs and checksums](osugi-sources.json) for the 316 original NetCDF files
needed for 2015–2021; it does not bundle those observations.

The MIT license covers the team's software and does not relicense the source
datasets. Source links and terms were checked on 2026-10-20.
