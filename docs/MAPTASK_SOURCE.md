# HCRC Map Task corpus source and attribution

The repository includes three unmodified archives of the **HCRC Map Task Corpus**, used as
the study's second domain. They are in [`data/maptask/archives/`](../data/maptask/archives/).

Source links:

- [HCRC Map Task Corpus home page](https://groups.inf.ed.ac.uk/maptask/)
- [Download page (annotations, maps, transcripts, audio)](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html)
- [Corpus design description](https://groups.inf.ed.ac.uk/maptask/maptask-description.html)

Copyright (c) 2007 Human Communication Research Centre, University of Edinburgh and
University of Glasgow. The downloads are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
They are redistributed here unchanged, with this attribution. The corpus is not
relicensed under this project's licenses, and no endorsement is implied. The corpus
maintainers ask users, as a courtesy, to tell them the topic of their work at
maptask@cogsci.ed.ac.uk.

Archives (downloaded and verified on November 16, 2026; changes: none):

| File | Source URL | Size (bytes) | SHA256 |
|---|---|---|---|
| `hcrcmaptask.nxtformatv2-1.zip` | https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip | 12,206,088 | `e90e083179c4fcdb7292d2a38c3238401412bbb4c3706b987befdc02e4c8af30` |
| `transcripts.zip` | https://groups.inf.ed.ac.uk/maptask/transcripts/transcripts.zip | 281,582 | `74355f7f7bea02260f96cee7be0f561f79f0578e9d610d99912a50fcaceae083` |
| `maps.tar.gz` | https://groups.inf.ed.ac.uk/maptask/interface/maps.tar.gz | 795,420 | `7ac98ad22bc1a7a4375d6c5f611c0743dbdbfe9de7d4e9bd86646f2ffc1c808c` |

What each contains:

- `hcrcmaptask.nxtformatv2-1.zip`: annotations v2.1 in NITE XML Toolkit format for all
  128 dialogues: timed words, dialogue moves, games, transactions, landmark references,
  per-map landmark inventories, and path deviation scores.
- `transcripts.zip`: plain-text transcripts of the 128 dialogues.
- `maps.tar.gz`: the 16 giver and 16 follower maps as GIF images.

The audio recordings are not included (too large); they are on the download page.

Derived data: [`data/maptask/maps.json`](../data/maptask/maps.json) is this project's text
transcription of the 16 maps (landmark positions and the giver's route), made from
`maps.tar.gz` and checked against the corpus landmark inventory. It is an adaptation of the
corpus and carries the same CC BY 4.0 attribution.

To unpack the archives for the code (into the gitignored `data/maptask_raw/`):

```bash
python -m maptask.corpus --download
```

This uses the committed archives and only downloads from Edinburgh if one is missing.

Citation:

- Anderson, A. H., Bader, M., Bard, E. G., Boyle, E., Doherty, G., Garrod, S., Isard, S.,
  Kowtko, J., McAllister, J., Miller, J., Sotillo, C., Thompson, H. S., & Weinert, R.
  (1991). The HCRC Map Task Corpus. *Language and Speech*, 34(4), 351–366.
- Carletta, J., Isard, A., Isard, S., Kowtko, J. C., Doherty-Sneddon, G., & Anderson, A. H.
  (1997). The reliability of a dialogue structure coding scheme. *Computational
  Linguistics*, 23(1), 13–31. (The dialogue move coding.)
