# Third-party source

The integrated dependency code is included to preserve the implementation used in the final experiment. No ownership of upstream models or code is claimed.

| Directory | Source | License / notes |
| --- | --- | --- |
| `third_party/aeroact` | https://github.com/return-sleep/AeroAct, commit `0417d39d4a739ed985ef33b9cb9be4327543640e` plus documented experiment compatibility changes | Apache-2.0; see its `LICENSE` and existing per-file notices. Dataset/media examples and obsolete cloud launchers are omitted. |
| `third_party/vipe` | https://github.com/nv-tlabs/vipe, exact experiment working-source files | Apache-2.0 for ViPE; retain its `LICENSE`, `THIRD_PARTY_LICENSES.md`, and nested notices. Includes the existing MoGe2 integration. File hashes identify the actual source rather than claiming an unverified upstream commit. |

The AeroAct changes are recorded in `provenance/upstream_source_difference_audit.json` and `patches/aeroact-upstream-0417.patch`. The ViPE source retains its existing integration notes, including `vipe/priors/depth/moge2/VENDORED.md` and third-party notices. Model weights, datasets, simulator binaries and runtime archives are obtained separately under their own terms.

The project-specific scripts have not been assigned a new license by this repository packaging step.
