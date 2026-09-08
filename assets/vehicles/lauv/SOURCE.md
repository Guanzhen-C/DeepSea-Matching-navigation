# LAUV model source

This directory contains the visual meshes for the Light Autonomous Underwater
Vehicle (LAUV) developed by the Laboratório de Sistemas e Tecnologia
Subaquática (LSTS), University of Porto, and OceanScan-MST.

- Upstream: <https://github.com/uuvsimulator/lauv_gazebo>
- Vendored revision: `3529e028b2e7fcf33e92db0a319ce71afb3d9ea6`
- License: Apache License 2.0; see `LICENSE` and `NOTICE` in this directory.
- Published vehicle basis: A. Sousa et al., "LAUV: The man-portable
  autonomous underwater vehicle," IFAC Proceedings Volumes 45(5), 2012.

OceanSim loads the original Collada geometry at its native 1.08 m length and
bakes the supplied body texture into OpenUSD vertex colours at scene startup.
The vehicle is not geometrically enlarged for presentation.

