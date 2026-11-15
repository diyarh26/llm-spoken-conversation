# Switchboard Dialog Act corpus source and attribution

The repository includes the unmodified `swda.zip` distribution maintained by
Christopher Potts. It is an annotated text corpus with caller metadata and parse
information, not the full Switchboard audio release.

Source links:

- [Christopher Potts corpus overview and downloads](https://compprag.christopherpotts.net/swda.html)
- [Upstream public GitHub repository](https://github.com/cgpotts/swda)
- [Original course-site archive](https://compprag.christopherpotts.net/code-data/swda.zip)
- [Upstream dataset card](https://huggingface.co/datasets/cgpotts/swda)

The originating course page and dataset card carry a
[Creative Commons Attribution NonCommercial ShareAlike 3.0 notice](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Preserve attribution, use the material noncommercially under those terms, and retain
the same license for adaptations. The upstream GitHub repository also contains a
GPL version 2 license; an unmodified copy is preserved in
[SWDA_UPSTREAM_LICENSE.txt](SWDA_UPSTREAM_LICENSE.txt). These upstream notices are
retained separately from this project's MIT code and CC BY 4.0 generated-data notices.
We do not relicense the corpus under the project's licenses or imply endorsement.

Archive provenance:

- Archive source: `https://compprag.christopherpotts.net/code-data/swda.zip`.
- Verification date: November 15, 2026.
- Size: 14,040,987 bytes.
- SHA256: `625742aff16f6ffa1ff2f30631d6c637483099d2962efa0c9513367204baa5e2`.
- Changes to archive: none.

The course-site archive and the current upstream GitHub archive have different
checksums. This project preserves the course-site copy used in the original
experiment; it does not silently replace it with the current GitHub copy. The
preserved GitHub license text was obtained at upstream repository revision
`3baa1fe49ebfa0fdb340eb6f69ee770836901cd9`.

The upstream README requests citation of Jurafsky, Shriberg and Biasca (1997),
*Switchboard SWBD DAMSL Shallow Discourse Function Annotation Coders Manual*,
Shriberg et al. (1998), *Can Prosody Aid the Automatic Classification of Dialog Acts
in Conversational Speech*, and Stolcke et al. (2000), *Dialogue Act Modeling for
Automatic Tagging and Recognition of Conversational Speech*. Credit Christopher
Potts for the merged distribution and follow the upstream README's citation details.

After cloning this project, extract the archive from the repository root:

```bash
python -m zipfile -e swda.zip data/switchboard
```

This creates `data/switchboard/swda/swda-metadata.csv` and 1,155 conversation CSVs.
The extracted copy remains gitignored to avoid storing the same data twice.
Public availability is not a general permission to redistribute arbitrary internet
datasets: this notice records the sources and terms for this particular distribution.
