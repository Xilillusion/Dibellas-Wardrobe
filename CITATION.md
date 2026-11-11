## Acknowledgements & Third-Party Dependencies

This project stands on the shoulders of the incredible *The Elder Scrolls V: Skyrim* modding community:

* **[nifly / PyNifly](https://github.com/BadDogSkyrim/PyNifly)**: Developed by BadDog, ousnius, and contributors. Used for parsing Skyrim `.nif` meshes, UVs, and skin weights.
* **[BodySlide and Outfit Studio](https://github.com/ousnius/BodySlide-and-Outfit-Studio)**: Developed by ousnius. The OpenGL fragment shader (`default.frag`) and normal/specular texture mapping logic were adapted from Outfit Studio's rendering pipeline to accurately reproduce Skyrim's material properties.
* **[BulletSharp](https://github.com/AndresTraks/BulletSharp)** and **[Bullet Physics](https://github.com/bulletphysics/bullet3)**: Developed by Andres Traks and Erwin Coumans (zlib license). Power the HDT-SMP cloth simulation in `smp_bridge/`.
* **HDT-SMP / Faster HDT-SMP (FSMP)**: The SMP XML format and constraint semantics (6DOF spring constraints, templates, collision filtering) follow the plugin's behaviour.

Licensed under the GNU General Public License v3.0 (GPL-3.0), compatible with its upstream dependencies.