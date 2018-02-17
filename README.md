## skylibs

Tools used for LDR/HDR environment map (IBL) handling and IO.


### envmap

`envmap.EnvironmentMap` Environment map class. Converts easily between those formats:

- latlong (equirectangular
- angular 
- sphere
- cube
- skyangular
- skylatlong

Available methods:

- `.copy()`: Deepcopy the instance.
- `.solidAngles()`: Computes the per-pixel solid angles of the current representaiton.
- `.convertTo(targetFormat)`: Convert to the `targetFormat`.
- `.rotate(format, rotation)`: Rotate the environment map using format DCM. Soon will support Euler Angles, Euler Vector and Quaternions.
- `.resize(targetSize)`: Resize the environment map. Be cautious, this function does not ensure energy is preserved!
- `.toIntensity()`: Convert to grayscale.
- `.getMeanLightVectors(normals)`: Compute the mean light vector of the environment map for the given normals.

Internal functions:
- `.imageCoordinates()`: returns the (u, v) coordinates at teach pixel center.
- `.worldCoordinates()`: returns the (x, y, z) world coordinates for each pixel center.
- `.interpolate(u, v, valid, method='linear')`: interpolates

### hdrio

`imread` and `imwrite`/`imsave` supporting the folloring formats:

- exr (ezexr)
- cr2, nef, raw (dcraw)
- hdr, pic (custom, beta)
- tiff (tifffile or scipy)
- All the formats supported by `scipy.io`

### ezexr

Internal exr reader and writer.

### tools3d

- `getMaskDerivatives(mask)`: creates the dx+dy from a binary `mask`.
- `NfromZ`: derivates the normals from a depth map `surf`.
- `ZfromN`: Integrates a depth map from a normal map `normals`.
- `display.plotDepth`: Creates a 3-subplot figure that shows the depth map `Z` and two side views.
- `spharm.FSHT` Fash Spherical Harmonic Transform
- `spharm.iFSHT` inverse Fash Spherical Harmonic Transform
- `spharm.sphericalHarmonicTransform` Spherical Harmonic Transform
- `spharm.inverseSphericalHarmonicTransform` inverse Spherical Harmonic Transform


### hdrtools

Tonemapping using `pfstools`.



## Roadmap

- Improved display for environment maps (change intensity with keystroke/button)
- Standalone `ezexr` on all platforms
- add `worldCoordinates()` output in spherical coordinates instead of (x, y, z)
- Add assert that data is float32 in convertTo/resize (internal bugs in scipy interpolation)
- bugfix: `.rotate()` not working on grayscale (2D) data
- bugfix: `.convertTo()` not working on grayscale (2D) data

