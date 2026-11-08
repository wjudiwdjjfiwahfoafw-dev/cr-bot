# Additional gameplay card references

The six `*_nejc.png` images are small portrait crops, used as fixed template-matching
references (not model training data):

* `skeletons_nejc.png`, `ice_spirit_nejc.png`: source image
  [0107QX4PFEolHM7kd0qo](https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection/images/0107QX4PFEolHM7kd0qo).
* `cannon_nejc.png`, `musketeer_nejc.png`: source image
  [06y1LirJBimsZnYxJjXU](https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection/images/06y1LirJBimsZnYxJjXU).
* `hog_rider_nejc.png`, `ice_golem_nejc.png`: source image
  [08lTSURxJpeILlssmSGE](https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection/images/08lTSURxJpeILlssmSGE).

Attribution: Nejc Zavodnik, *Clash Royale troop detection*, Roboflow Universe;
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The images were cropped from
spectator card portraits and saved as PNG. This attribution does not imply endorsement.
These older portraits complement, rather than replace, the existing templates. Their utility
varies by game version and UI; use `scripts/capture_templates.py` for your own device.
