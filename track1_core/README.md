# Track1 Core

`track1_core` contains the project-owned implementation of acoustic-visual floorplan localization. External baselines remain outside this package.

The intended data flow is:

```text
visual observation + floorplan + pose grid -> visual likelihood
acoustic observation + floorplan + pose grid -> acoustic likelihood
visual likelihood + acoustic likelihood + valid pose mask -> posterior
posterior + ground-truth pose -> metrics
```

The exact tensor and coordinate contracts are defined in [../docs/data_contracts.md](../docs/data_contracts.md).
