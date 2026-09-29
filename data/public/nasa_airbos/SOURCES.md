# NASA AirBOS 4 background-oriented schlieren images (public reference set)

Six official NASA images from the AirBOS 4 flight series (NASA Ames / Armstrong, Black Mountain Supersonic Corridor near Edwards AFB, December 2018; publicly released 2019): processed air-to-air BOS results showing supersonic USAF T-38s and their shock waves. All files were downloaded as the `~orig` (highest-resolution) asset from the images.nasa.gov asset manifest (`https://images-api.nasa.gov/asset/{nasa_id}`) on 2026-09-29.

**Used for qualitative illustration only; no NASA endorsement implied.**

Credit: NASA (NASA Ames Research Center). Images acquired by JT Heineck; schlieren data processed by Neal Smith (per NASA's descriptions).

## Usage rights

Source: NASA Images and Media Usage Guidelines, https://www.nasa.gov/nasa-brand-center/images-and-media/ (retrieved 2026-09-29). Quoted verbatim:

> NASA content – images, audio, video, and media files used in the rendition of 3-dimensional models, such as texture maps and polygon data in any format – generally are not subject to copyright in the United States. You may use this material for educational or informational purposes, including photo collections, textbooks, public exhibits, computer graphical simulations and Internet Web pages. This general permission extends to personal Web pages.

> NASA content used in a factual manner that does not imply endorsement may be used without needing explicit permission. NASA should be acknowledged as the source of the material. NASA occasionally uses copyright-protected material of third parties with permission on its website. Those images will be marked identified as copyright protected with the name of the copyright holder. NASA’s use does not convey any rights to others to use the same material. Those wishing to use copyright protected material of third parties must contact the copyright holder directly.

> The NASA Insignia, Logotype, identifiers, and imagery are not in the public domain. The use of the Insignia, Logotype and NASA identifiers is protected by law, and imagery is made available for use consistent with Media Usage Guidelines.

> If the NASA material is to be used for commercial purposes, including advertisements, it must not explicitly or implicitly convey NASA’s endorsement of commercial goods or services.

From the same page, "Artificial Intelligence (AI) Applications" section (relevant because these images are run through a model):

> Attribution: NASA cannot verify the accuracy of information once it’s incorporated into a large language model. Therefore, attribution of the information directly to NASA is not permitted. Similarly, NASA does not permit our Insignia (i.e., NASA Meatball, Worm logotype, or Seal) to appear with AI generated imagery or in the training of AI tools.

> Source Disclosure (No Permission Implied): As a statement of fact, you can acknowledge your AI tool includes NASA source material, but do not imply any review or permission was granted by NASA for its specific use.

> Endorsement: Use of NASA content to train AI models does not constitute endorsement by NASA.

Rights check per image: each image's NASA API record (title, description, photographer, keywords) and its embedded metadata (`metadata.json`: EXIF/XMP/IPTC) were inspected. None carries a third-party copyright notice; the only credit fields present are `photographer: JT Heineck` (NASA Ames photographer) and, for ACD21-0016-001, `IPTC/XMP Credit: NASA/Ames`. No NASA insignia (meatball/worm/seal) appears in any image; the aircraft carry U.S. Air Force markings.

## Notes for the downstream (grayscale |displacement|) model

- The three `ACD21-*` JPEGs are colormapped presentation composites (RGB). Converting to grayscale will not recover displacement magnitude (non-monotonic colormaps).
- The three `AD21-*` TIFFs are single-channel 8-bit grayscale renderings. Shocks appear as signed dark/bright bands on a mid-gray background, i.e. a directional (knife-edge-like) displacement component rather than |d|. They are the closest match to the model's input domain, but a sign/offset mismatch is expected.
- In all six, the aircraft are photographic cut-outs pasted over the BOS field ("plane_drop"), so aircraft pixels are not BOS data.
- Pairs: ACD21-0016-00N and AD21-0016-00N show the same pass (color vs grayscale). They are not raw camera frames; no raw BOS frames are published on images.nasa.gov.

## Files

### `ACD21-0016-002_F4_P3_RGB_planedrop_cleaned.jpg`

- NASA ID: `ACD21-0016-002_F4_P3_RGB_planedrop_cleaned`
- Title: AirBOS4
- Date created (NASA): 2018-12-18
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/ACD21-0016-002_F4_P3_RGB_planedrop_cleaned
- File: https://images-assets.nasa.gov/image/ACD21-0016-002_F4_P3_RGB_planedrop_cleaned/ACD21-0016-002_F4_P3_RGB_planedrop_cleaned~orig.jpg
- Asset manifest: https://images-api.nasa.gov/asset/ACD21-0016-002_F4_P3_RGB_planedrop_cleaned
- Dimensions: 3600 x 3241 px, RGB, 8-bit; 5,028,047 bytes
- SHA-256: `6a504d2736b591763da12277b81f14c82915cd92e3f8c22418c449f7a29a74f7`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.01 pass. This data is the first time shockwave interactions between two full scale aircraft traveling faster than the speed of sound have been imaged and shown with schlieren visualization. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith"
- Alt text (description_508): "Schlieren visualization of T-3"
- What it shows: Processed BOS composite, pseudocolor (blue/cyan background, yellow/red/dark-blue contour bands). Two T-38s in echelon formation viewed from above (plan view), photographic cut-outs of the aircraft pasted over the BOS field. Bow and tail shock families from both aircraft clearly visible, as are the exhaust plumes; the trailing aircraft's shocks interact with the lead aircraft's shock system. This is the widely published 2019 'two T-38s' image. Colormapped, not a raw frame, and not a magnitude map.

### `AD21-0016-002_F4_P3_CAM_46_plane_drop_v.tif`

- NASA ID: `AD21-0016-002_F4_P3_CAM_46_plane_drop_v`
- Title: AirBOS4
- Date created (NASA): 2018-12-12
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/AD21-0016-002_F4_P3_CAM_46_plane_drop_v
- File: https://images-assets.nasa.gov/image/AD21-0016-002_F4_P3_CAM_46_plane_drop_v/AD21-0016-002_F4_P3_CAM_46_plane_drop_v~orig.tif
- Asset manifest: https://images-api.nasa.gov/asset/AD21-0016-002_F4_P3_CAM_46_plane_drop_v
- Dimensions: 2438 x 1388 px, Gray, 8-bit; 6,200,848 bytes
- SHA-256: `12eb8459a6fd499e3ac635c133e1d5ddfd35a4d55c3926ffd74e33b9ec0b45a0`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.02 pass. The interaction of the shockwave of the trailing aircraft with the exhaust plume of the lead aircraft shows a shockwave reflection. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith."
- Alt text (description_508): "Schlieren visualization of T-38"
- What it shows: Grayscale (8-bit, single channel) BOS rendering of the same pass/scene as ACD21-0016-002 (camera 46). Shocks appear as signed dark/bright bands on mid-gray: this looks like a single directional displacement component (knife-edge-like rendering), not a displacement magnitude. Aircraft are photographic cut-outs pasted in. Closest of the set to the downstream model input, but polarity/sign semantics differ from |d|.

### `ACD21-0016-003_F4_P4_4-15_12Hprint.jpg`

- NASA ID: `ACD21-0016-003_F4_P4_4-15_12Hprint`
- Title: AirBOS4
- Date created (NASA): 2018-12-12
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/ACD21-0016-003_F4_P4_4-15_12Hprint
- File: https://images-assets.nasa.gov/image/ACD21-0016-003_F4_P4_4-15_12Hprint/ACD21-0016-003_F4_P4_4-15_12Hprint~orig.jpg
- Asset manifest: https://images-api.nasa.gov/asset/ACD21-0016-003_F4_P4_4-15_12Hprint
- Dimensions: 3600 x 2880 px, RGB, 8-bit; 3,356,448 bytes
- SHA-256: `7eca4563a39e67ddd930fa39d1ef6b9adb88af5f8f9bc92ff5f3f51cc71577ce`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.02 pass. The interaction of the shockwave of the trailing aircraft with the exhaust plume of the lead aircraft shows a shockwave reflection. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith."
- Alt text (description_508): "Schlieren visualization of T-3"
- What it shows: Processed BOS composite, pseudocolor (orange/red/yellow, dark bands). Two T-38s in echelon formation, plan view, aircraft cut-outs pasted in. Strong curved shock interactions: trailing aircraft shocks bend where they meet the lead aircraft's shocks/plume. Colormapped composite.

### `AD21-0016-003_F4_P4_plane_drop_v.tif`

- NASA ID: `AD21-0016-003_F4_P4_plane_drop_v`
- Title: AirBOS4
- Date created (NASA): 2018-12-12
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/AD21-0016-003_F4_P4_plane_drop_v
- File: https://images-assets.nasa.gov/image/AD21-0016-003_F4_P4_plane_drop_v/AD21-0016-003_F4_P4_plane_drop_v~orig.tif
- Asset manifest: https://images-api.nasa.gov/asset/AD21-0016-003_F4_P4_plane_drop_v
- Dimensions: 2374 x 1280 px, Gray, 8-bit; 5,636,440 bytes
- SHA-256: `abb9b754a5ff94d208da4817d52331e9b42f95202f2f7a425e30522c2a855a07`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.01 pass. This data is the first time shockwave interactions between two full scale aircraft traveling faster than the speed of sound have been imaged and shown with schlieren visualization. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith."
- Alt text (description_508): "Schlieren visualization of T-38"
- What it shows: Grayscale (8-bit, single channel) BOS rendering of the same scene as ACD21-0016-003. Signed dark/bright shock bands on mid-gray (directional/knife-edge style), aircraft cut-outs pasted in, exhaust plumes visible. Shocks and curved interaction clearly visible.

### `ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint.jpg`

- NASA ID: `ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint`
- Title: AirBOS4
- Date created (NASA): 2018-12-11
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint
- File: https://images-assets.nasa.gov/image/ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint/ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint~orig.jpg
- Asset manifest: https://images-api.nasa.gov/asset/ACD21-0016-001_F3_P3_Knife_red_planedrop_12Hprint
- Dimensions: 2700 x 3600 px, RGB, 8-bit; 3,341,055 bytes
- SHA-256: `825b4f4fc63e89cc50bd756865b7f3e90d795f00b0bf1c27deb7c9e4fa270295`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.01 pass. This data is the first time shockwave interactions between two full scale aircraft traveling faster than the speed of sound have been imaged and shown with schlieren visualization. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith"
- Alt text (description_508): "Schlieren visualization of T-38"
- What it shows: Processed BOS composite, pseudocolor (orange/yellow, 'Knife_red'), portrait crop and rotated relative to the grayscale version. Single T-38 in side/banked view (USAF TPS 'ED' tail markings visible), cut-out pasted in. Multiple nested oblique shocks (bow, canopy, wing, tail) and exhaust plume visible. Note: NASA's description text speaks of two aircraft, but only one aircraft appears in this frame.

### `AD21-0016-001_F3_P3_knife_plane_drop_v.tif`

- NASA ID: `AD21-0016-001_F3_P3_knife_plane_drop_v`
- Title: AirBOS4
- Date created (NASA): 2018-12-12
- Center / location: ARC / NASA Ames Research Center
- Photographer / credit (as given by NASA): JT Heineck
- Page: https://images.nasa.gov/details/AD21-0016-001_F3_P3_knife_plane_drop_v
- File: https://images-assets.nasa.gov/image/AD21-0016-001_F3_P3_knife_plane_drop_v/AD21-0016-001_F3_P3_knife_plane_drop_v~orig.tif
- Asset manifest: https://images-api.nasa.gov/asset/AD21-0016-001_F3_P3_knife_plane_drop_v
- Dimensions: 2560 x 1600 px, Gray, 8-bit; 7,129,516 bytes
- SHA-256: `a17d6b83765c9de914bc438350c408f2fb2daffcab16a9c20bfe38314ab4c0c5`
- NASA description (verbatim): "Composite image of Background Oriented Schlieren (BOS) data (contour) with a cut-out images of the T-38’s during a Mach Number 1.02 pass. The interaction of the shockwave of the trailing aircraft with the exhaust plume of the lead aircraft shows a shockwave reflection. Original recording of the pass taken in the Black Mountain Supersonic Corridor at near Edwards AFB in December of 2018. Image acquired by JT Heineck, schlieren data processed by Neal Smith."
- Alt text (description_508): "Schlieren visualization of T-38"
- What it shows: Grayscale (8-bit, single channel) synthetic knife-edge BOS rendering of the same pass as ACD21-0016-001, landscape orientation, with dark vignetted borders at the image edges. Single T-38 with its nested shock system and plume. Signed dark/bright bands (directional), not magnitude.
