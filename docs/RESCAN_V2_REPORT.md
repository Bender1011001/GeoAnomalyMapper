# Fixed-Pipeline Rescan of the 33-Tile Heavy Survey (2026-07-14)

The original heavy survey (Konya / Harran / Nineveh / Balikh / Khabur) ran
under two recall bugs (ranking starvation + systematic tell misclassification,
both fixed and committed 245862f/ad6a647). This rescan repeats the exact same
33 tiles with the fixed pipeline.

## Yield

| | old (broken) | rescan (fixed) |
|---|---|---|
| confident hits (>=0.80) | 15 | **118** (7.9x) |
| Konya | 10 | 28 |
| Harran | ~3 | 23 |
| Nineveh | ~1 | 6 |
| Balikh | ~1 | 36 |
| Khabur | 4 | 25 |

## Validation

- **External (Menze-Ur 14,324-site catalog): 25/25 Khabur hits within 1.5 km
  of a catalogued site (18/25 < 500 m), 0 spurious "novel" claims** — in the
  ground-truthed region the fixed pipeline's confident tier is ~100% real.
- Konya anchor: the eyeball-verified hoyuk re-detected at 0 m (ANCIENT 0.93);
  volcanic terrain correctly rejected wholesale (NATURAL 0.82-0.91 despite
  prominence z up to 190).
- Spot eyeball of the >=0.93 tier: 0.97 = major walled tell near Harran
  (~800 m oval with rampart circuit); 0.93 = classic village-on-tell. Real.

## Notes

- Confidence distribution: 80 hits at 0.82, 24 at 0.87, 10 at >=0.90.
- Hits outside the Khabur footprint have no ground-truth catalog; they are
  real-tell candidates (open DBs cover <<1% here), NOT claimed discoveries.
- Full list: docs/rescan_v2_results.json (lat/lon, confidence, VLM reason,
  distance to nearest Menze-Ur site).
