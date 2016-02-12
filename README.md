# Loki
--
**L**ibrary **o**f **K**orrelated **I**ntensity tools (still thinking of a better name...)

![](https://derpicdn.net/img/2012/8/27/83960/thumb.png)

###About
LOKI is a functional group of tools useful for doing angular correlations on SACLA MPCCD detector images.

Typically, our group (Doniach group / stanford Univeristy / Applied Physics) will 

* Interpolate the MPCCD images to form polar images
* Parameterize the polar images
* Correlate the difference of selected polar images (this minimizes artifactual detector signals)

###Analysis flow
Consider you have a SACLA h5 file (converted on the SACLA servers using the in-house SACLA DataConvert program). Typically this file will consist of all the exposures in an experimental run R.

Then, a typical workflow will look like:

1. Use the function ```process_sacla.interpolate_run``` to convert SACLA h5 files into Loki-style polar image h5 files. Examples of how to go from SACLA h5 files to Loki-style files can be found in the tutorial pdf.
 
2. Use ```make_db.MakeDatabase``` to parameterize the experimental run R. Requires Loki-style polar image file as input. Makes a Loki-style pandas database.

3.  Use ```make_tag_pairs.MakeTagPairs``` to group exposures. Requires Loki-style pandas database.

4. Use ```cor_tag_pairs.CorTagPairs``` to record exposure difference and compute the so-called difference correlations. Requires all outputs from steps 1-3.


>#######dermen
>>originator: Derek Mendez 
>>>contact: dermendarko@gmail.com
>>>> Doniach group Stanford applied physics

![](http://pre11.deviantart.net/f054/th/pre/i/2012/187/6/e/loki_vector_by_flaredragon497-d567hsv.png)