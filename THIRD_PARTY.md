# Sources and attribution

- **MaleCNS v1.0**: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory
  of Molecular Biology, and Google Research. The downloaded connectome data are
  licensed **CC BY 4.0**: https://male-cns.janelia.org/download/ and
  https://creativecommons.org/licenses/by/4.0/. Source URLs, byte sizes and
  SHA-256 values are recorded by `fly_arena/data.py` and in each prepared graph's
  `manifest.json`. Transformation: retain annotated non-glial entries and all
  edges between them; reorder into CSR; derive approximate input/output ports
  and simplified transmitter signs. This repository does not redistribute the large
  original files. Preview footage is generated using these transformed data.
- **NeuroMechFly / FlyGym 2.1.0**, NeLy-EPFL contributors, Apache 2.0:
  https://github.com/NeLy-EPFL/flygym and https://neuromechfly.org/.
  Walking interface follows the official turning-controller tutorial:
  https://neuromechfly.org/tutorials/4d_turning_controller/.
  The package supplies the fly body, step recordings and hybrid controller;
  it is installed as an external dependency. FlyGym's license is included in
  `licenses/FlyGym-APACHE-2.0.txt` for convenience.
- **MuJoCo 3.9.0**, Google DeepMind and contributors, Apache 2.0:
  https://github.com/google-deepmind/mujoco.
- Numerical/data/image dependencies retain their own licenses in their
  installed distributions: NumPy, Numba, PyArrow, SciPy, Pillow, ImageIO,
  imageio-ffmpeg and FlyGym's transitive dependencies. No compiled FFmpeg binary
  is included in this repository.

No source files or dependencies from DoomFly are included. The LIF dynamics,
retinal sampling and descending-command adapter in this project are independent
experimental implementation choices. Janelia does not supply or validate these
dynamics, nor does this project establish natural fly behavior or learning.
