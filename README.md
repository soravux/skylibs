## skylibs

Tools used for LDR/HDR environment map (IBL) handling and IO.


### envmap



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

