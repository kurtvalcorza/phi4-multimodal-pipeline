"""Captioning dataset contract for fine-tuning: the pinned VizWiz-Captions sample, validation, seeded
image-disjoint splitting, BYOD loaders and JSONL export.

The default dataset is **real** and out of the base model's distribution: VizWiz-Captions (Gurari et al.,
ECCV 2020; CC BY 4.0) — photographs taken by blind people, each with five crowd-written captions, from a
population and a caption style (what is held, what the label says, how the picture is framed) that COCO
captions do not cover. `microsoft/Phi-4-multimodal-instruct` was instruction-tuned on broad web and synthetic
image-text data and never on VizWiz; its default captions are long and descriptive where VizWiz references are
short and practical, which is exactly the style gap a bounded adaptation can be measured on. (Module ported from
the fleet's `blip-captioning-pipeline`, whose E2E row pinned this sample first.)
The sample comes from one pinned parquet shard of the Hub mirror `mm-eval/VizWiz-Captions` (a re-conversion
of the official `val.json` with the rejected and pre-canned captions dropped, so an image can carry fewer
than five references): the four text columns of all 1,550 rows are read **column-only** over HTTPS range
requests through `pyarrow` (about 0.5 MB), and the image column of **row group 0 only** (336 photographs,
about 84 MB) is read the same way — the shard's declared size and SHA-256 checked against the pins before
any byte is read, the decoded text columns' SHA-256 and every photograph's SHA-256 and size checked after.
Images without a surviving reference caption are excluded from the sample.

A record is ``{id, image_id, image, captions, category}`` — the path of the digest-verified photograph, its
one or more reference captions and a category (`text` when the VizWiz annotators flagged text in the
photograph, `no-text` otherwise; BYOD records may carry any label, `other` by default). Every record's image
is its own (one row per photograph), so a split by image is a split by record; BYOD records may share an
`image_id` and are then kept together.
"""

from __future__ import annotations

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import io
import json
import random
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MAX_PREFIX_CHARS, MODEL_ID, validate_image

CORPUS_NAME = "VizWiz-Captions"
CORPUS_REPO = "mm-eval/VizWiz-Captions"
CORPUS_REVISION = "c4a6d897836e7885d0095134f92d392e4e770539"
CORPUS_RELEASE = (
    "VizWiz-Captions v1 (2020) as re-converted on the Hugging Face Hub, dataset revision c4a6d897"
)
CORPUS_LICENSE = "CC BY 4.0 (Gurari et al. 2020; vizwiz.org/tasks-and-datasets/image-captioning)"
CORPUS_TEXT_COLUMNS = ("id", "answer", "question_type", "text_detected")
CORPUS_IMAGE_COLUMN = "media"
CORPUS_FILE: dict[str, Any] = {
    "path": "data/val-00004-of-00005.parquet",
    "bytes": 392_245_504,
    "sha256": "4492465a41d32b3c12b8b7b6a0cf7e0a0e202a5b825b006ca0c85dcdf24efd3e",
    "rows": 1_550,
    "text_sha256": "9799ebb13cf6a7e7c76afdd180892e21499ecefff697212ce4e505c7fc207d6e",
    "row_group": 0,
    "row_group_rows": 336,
}
DEFAULT_CACHE_DIR = Path("weights") / "vizwiz-captions"
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 208, "validation": 40, "test": 70}  # the 318 row-group-0 photographs with a caption
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MIN_CAPTIONS = 1
MAX_CAPTION_CHARS = 500
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# image id -> (sha256, bytes) of the JPEG carried by row group 0 of the pinned shard (336 photographs).
IMAGE_PINS: dict[str, tuple[str, int]] = {
    "29631": ("635042d060e1f42bdebf7d16af7b7b1f78763482b21959dd6ab3e642fb131522", 219537),
    "29632": ("595732108a90de2ebb653d8bac8ab1d88d837624103c0a3344161f57ff84bd7b", 208376),
    "29633": ("93ff2bd506c778c53715a965a83d8d55a6d3a46b4b06dd91108689284ba3aa5f", 14682),
    "29634": ("b7bfa7344d17b18dcc03f3fdcecb874963a3584d14b92210717f351155e42d0a", 232493),
    "29635": ("f807878774e35d16396c8d6600c67b8bd6cb93a89328da8d29bfd8c34386ec85", 268910),
    "29636": ("fa7f75a23aaedfeb26032a7bce86bdfaace2d204695efafc17b87a0907bae611", 82416),
    "29637": ("407efaaabffc133f142c0c690318a16812c39fc30902ac95d86b3b8640b511c9", 506359),
    "29638": ("486f02412386b81757936e3e9782efe234fe69760d24fc70734a0b60d7b284b0", 443910),
    "29639": ("37cfae24bbfb0faf1d6fadda974b2824bf8976ffe7fac02f740eee75027d95b3", 349992),
    "29640": ("da467dffb38a6d407cee8fe1b1b690a3d8af70f378941e0dd992284f9d2d8f5e", 269887),
    "29641": ("4f2e2bea724a6b3b3ead65718da3b3d17bcd34f56913c0652ba91d89acd6967e", 257024),
    "29642": ("afbdef671c1faa2f191466f7320b07a18c583afe2c3e856eccefec46532f29db", 225839),
    "29643": ("cdb9b4f804e4ee61754d82560940c72fe21a3747fac6c75648faa613231da76a", 261053),
    "29644": ("632a0a1662b58ad6b13cefef34a51898b93fcdf88abebc9217e312691128f569", 193354),
    "29645": ("889242e35ee151e13fe6243567126d0476c9221a3985ecf8dce4f6b415f2b59e", 258707),
    "29646": ("0454ddcec779bfbfafd093518f67165b2a7a092e81917e78c213aa18f6dbb8cb", 346643),
    "29647": ("d0ef537515306a3df5c77d85dd0d2666c4aa0da2f72502cc82500cf6524d1e19", 271956),
    "29648": ("ceb4c0b7b00bbb1dfbdacec67ea4d8fc8d59754a9856c0eb87a52a554a570664", 321526),
    "29649": ("dccb89f70f8be06bdcff3e4947422fb5a69cf5069b00e707c22e92090fe91397", 272027),
    "29650": ("f5e41db77978764a79010a335c6e21d5a053892517dd34a4dfd315ae17eeeb1a", 434003),
    "29651": ("72be2c699582d963754a4986c56c22701511cef3c802c48e0d7aa558f9107a09", 292153),
    "29652": ("7db7f94532355552e80fbeb015d9e4db978398cc57512a9c96bd0db44a50c0c4", 338389),
    "29653": ("0a84818b7f47bf36c5e6e4001481dfc8faec6bf232792ce521b75db2cf50ec86", 188622),
    "29654": ("15eca622fe300527a6c92df119b8a2f3947df676d627047ea50e96de39303d19", 49771),
    "29655": ("b2e54c27133737e21ca33e858ef52dd66b7a68bd9144f947094702295511a891", 39778),
    "29656": ("719f5852b6092754ac26f3feb6d6d974da19a848146ea3bced950e71dd55cc0b", 144496),
    "29657": ("568dd4519806b64e63d40f063205e485c1fc9f42a6719fff7acbe59d94c01148", 199012),
    "29658": ("dac90383b9f44fbff68426da47687c124f2511152ce4d3e94c307855a43e3871", 337908),
    "29659": ("61be884a769cb69f89e847cb643c4a78c01e747934d56186a53a8b3e6daf244f", 30681),
    "29660": ("08fac9a7c1c30e0a1aa92f7295f3366454d4629a78bd3678b86090ffff8fc373", 205850),
    "29661": ("72e87e97b362c7fe3412676ca230cb5765cd6d9091d530e4f1f894458344c97d", 330913),
    "29662": ("61bb302f64ec861c9628b1efc04c065a1fffe4c9fa6688b16d06ce946eec0c1f", 182155),
    "29663": ("a417180d3f937c01b9269cab5e46e096a05714c0c2fdf630bf7430cb351e44e9", 258175),
    "29664": ("3913b82cecc6a49c057c9cc4dde219920f62aa9c43f2b1843c58cebdefa5ee20", 217507),
    "29665": ("e4b02231db046fc3b6b25cfd434ba4ed25ba74f98941cedc7127fa3d5579c0e5", 673261),
    "29666": ("1d6e584c39a7b469a97f7125ab91dbfed50ce151c76c0037365754b3da1935ec", 415821),
    "29667": ("994e2c3cf86c546c10480b01064a9af915add972553ad8b4645a3e5a183969fe", 459367),
    "29668": ("f6c0fc8835171e91cf8368be8c33e8af565c037e89fdd61293e92dc61f7a57ca", 283625),
    "29669": ("3a465d5426ce6c766a9aaee0aa135ff5b438197587a21111bb06f32d827a4229", 155717),
    "29670": ("e60a335c69895e52db1f1265533f773977117819d10cffc7c7a6fe5b77fd911b", 218339),
    "29671": ("45a785cec7d063f545f72b8e5f1c3ce1820c43f62bd725fdaff2070db47468ff", 301005),
    "29672": ("20df609ab7455f25c26a8ad7e79224e921dd3656de31c8e9757b0bcda5a3e6e8", 271522),
    "29673": ("9009def8616940b157d2133306edd1a2f7f39b129a158a9a19ebd88069f246d8", 36083),
    "29674": ("5160d7be46d40cd6cb3b122d94b48e8791ed0a939408884b130b11ce3eff64aa", 311110),
    "29675": ("ea0dcee84b2b3ccaf9371e8ffecc779f6a7acf8a1303b3c3438c3b507a3fde8a", 331823),
    "29676": ("799e06916b505131dbc20db33c6aef34f9c325a347819de063c3faa599c1e607", 209338),
    "29677": ("67fe5425d3f0e7aa2a260d854cf25ea1c1c41f9cb30dc3691b16700966011c85", 176798),
    "29678": ("80727a1e760ecee07214bfec0448fda106fc07f3c849a4248b347514b0fcfa91", 591377),
    "29679": ("80e29692260e0ade581839baec5d5a4550e5822cab0aa15115cd3c96a1569f3e", 33215),
    "29680": ("7c836719c64e5d97ba83b99a6fba966062a180a1e3aaffa30dde4d2117daef89", 373575),
    "29681": ("2416fea174cfd4522953171bbcaa38900aa38d3e9671c6b4ff9dbe314caa5f13", 293047),
    "29682": ("05c5971c974f1c9843ce2bb0d08c1500e380596107ac76a24b057b0efa658573", 466144),
    "29683": ("f3f6e73a1d1dd7ba7449ba0634b09decfd2350dfbe8f3fdcf3fbbdbe2719b5b7", 185222),
    "29684": ("92a3f9f7bcd352a6f1c03d305f98aec1f1d3cc315f3d6c2bab46f13bca7817b6", 325758),
    "29685": ("2db1c9bc24a7a12af81c2d8a5539e422d210e03eda0ecc0af6944e98bdea1305", 304402),
    "29686": ("fd3ebb92d81317be6db45ad6326d3227f095edd2aa93c2f3e571806b5a4301b9", 379396),
    "29687": ("8be2b3264792a2ac22abc9a613d5759e9f31e79b3659f78039a70c08d449860a", 233109),
    "29688": ("d0862ed97636b9cb77577c5ba6a82cb75176548231568a1a50c00bb6c47700d1", 301273),
    "29689": ("31b332bfc7af3903b1f6091cc92627ca299dbe9c2e2330f730228e505e573296", 129869),
    "29690": ("d39dbd910bd9fbe3b36faa99b6580cf371b1aff87230066f91f46d1a55dd124e", 224998),
    "29691": ("3012921370798079afe212db4bc22cfbf284bd7584e74cf33297956bd9189f8e", 385230),
    "29692": ("903718f7a211747b1677bc680f72facb3a4d582abd03131dd98c88ece06d0c54", 280343),
    "29693": ("d44c2877651a8dba7cf2943265f0f6353ff38697cf921fc14c33a5a1afcbcd73", 196010),
    "29694": ("0a20818fe43942fd653591cbed494c38a2550e27e3892753c278142b462f900d", 165487),
    "29695": ("98650214fe93da1f94e6ad9d0fc19bb79e2485223e706e1ba292a50af85c988f", 63581),
    "29696": ("ddfd96ca16ada6259061247926d6bda0d302b3ec7adbda44e66f0a31c27baf11", 490864),
    "29697": ("7e68d03f88d8e6a65ed883ccf45f5b2e1c7b1668e5b168355650311601fb3979", 276114),
    "29698": ("dc8bf976fc17cebf01fbf6779c8400ccf27a12ac7124cf7490a560051c18b50b", 520196),
    "29699": ("54e6417e77612a9b8444e4febb025e12e1488b05aa72fbcf63e8bd69a8bf6652", 52962),
    "29700": ("b03833951a0c568edc3a1568ad14769b58615b57a32d50b297fe12df05989128", 178206),
    "29701": ("d1d5c7795f4c794645bc728e02efe447bc8a4c896aa397fb843925d510262e70", 204765),
    "29702": ("d175d905034ca667379229035051f90abcd49d42d9aedd17cfc19bf6dc4ed7db", 329707),
    "29703": ("429e3c4a9c822847f9670251c8027485505686724caa0cf467fecb4b46afd7e7", 98889),
    "29704": ("1302773b4d1d13c8242e9b6fe283f85bdc6331b01de77f795eded9d2a399d334", 208973),
    "29705": ("beb66040a0e41e4f3ae26db0cb2d78321bfcb71b6f2fe3198671ceda10e3ffa6", 305340),
    "29706": ("ac55c0fd7e334f1a5cdf41006c33cd1c699ac71a9a95556e27f6ea564ae8a587", 260218),
    "29707": ("f86d9785862905e412f37cc2dbd49836c5aa6ead03c586a4f9a76555f1894305", 397788),
    "29708": ("f69d115f7a11814d05ab5e8e2b90070938a71edd890dbc4ae6eb4c62a96d6b8c", 339030),
    "29709": ("dd65b40eee8a23e0bdf47b67d5a3be54b31913ba9918fabe4a9d4aaf068f63aa", 218005),
    "29710": ("741594d8849df32c38f1898498573c18da785e19dea9481723d770b7d47bfad0", 480990),
    "29711": ("b4cb496aa54857053256ea084a4390bc9a157c923ac601c77057ef257439b7d7", 429787),
    "29712": ("8473b6193e761171dd28d3c2cb4894dfeee51b014330c2289cfd40b8d8a53c92", 315871),
    "29713": ("569fa73a25181a278b420556c8bfcce2e2abeb8e36e844c53c977ac8c3aa3e23", 254682),
    "29714": ("c55af0d333f124114fe440ca756b5d39ead1c14b07ebacedbd52300622e46454", 322112),
    "29715": ("d3007697c34a7ef94452e5ef3fd260c549fbfe37473d015c25f880e83f8bd284", 208474),
    "29716": ("a53d0db806788cc105ebd5ab4749eb7a31617a95b0394cc30a808bc5d9f79667", 375964),
    "29717": ("37d835e42fc5693772402122065dbbf8bb7da4642e63c26a97609146dfe64cb2", 30875),
    "29718": ("1682aeae89a9d92c6484dd7ba787922846f463103c4b58fcf3251e91c5190a16", 189651),
    "29719": ("ad695d09ad35c7c2d528d26b30f602e7883e5018007a4d6109c18d3020d3c382", 279840),
    "29720": ("4af2ab447a764814d215712afcbfb12b42de8e9ba29a1aa9b02ef1550240d2d3", 188421),
    "29721": ("574d2175f929da7be1a8d1c2849939be185945aa0565670d8036854e553dd718", 504332),
    "29722": ("88136aa35f1b1bf1ba2caf69fd12c466364712822bace72e5d188f7f0b41ce0b", 234812),
    "29723": ("91f501ed03e17b9860515fb8602956a17d569f927da5b4f61228116b5e282578", 22300),
    "29724": ("81db3e3d3276b80cc910cd207bbe43fe347cd300f52e32f6659cd965590716a7", 173636),
    "29725": ("8901239345b4524db707c92fecb542042d2c44c293ba35efb6b86c60d1feceeb", 267482),
    "29726": ("917694384bbd0debc698d56d6cdd38d18e9a6b4b8d1adef6faf2f4dd3858f83a", 227891),
    "29727": ("864c8141cca16977f599cdcef0d4e1313b96d51044c116f765c7bf226b08468a", 289964),
    "29728": ("2ea6ece248a3fbb0b7a8c0a2143daf36a429c4934b8ee72be014c95a8cecd868", 407475),
    "29729": ("480630ab0b3bc6ebade59d97f2c0f37a99df9b76a88dcf699c22cbfa6b5098eb", 525377),
    "29730": ("ca6c42ddd9afe5d55fdd8f47ea6feb26ba4a5a6600afdf6130b212b352aaf9b6", 96112),
    "29731": ("1cff0ee25e06c7c68ca657cca15d383cc831bcb4734577611afc676eb264d393", 325968),
    "29732": ("f44b95a2c2e643066d75410a7d860d2c12fcd7a890e06d4f9a226044554ceee7", 416985),
    "29733": ("7d72c78e4ca19ac4b71455aa5c7445f1b4f46ceced6580ee57f63345514d7ea4", 456935),
    "29734": ("abee412faa3a0c7c9ec64a65ee533cc717ff5d5aba34532fa1a71c091693dd15", 364966),
    "29735": ("041c0e34e313709ed82a7183eb122e04f588379d70e1688ce4f0fae78748bdca", 5674),
    "29736": ("3837381d2a37dca6c1aff7916a20108ab791a3c609c25adcaaac0ddc4eaed079", 254531),
    "29737": ("a4f9e99838d4cae4fedd6c7327f1f056d0c766ad3b71b3aafd13f1aa2ebd633e", 279112),
    "29738": ("e437e827d89a0e7f948b1130c59e14498b48d794542e6a290f156cdc75e6c8a2", 438833),
    "29739": ("eca903ff72aaa2e6d28f8ba09ebdc11a8870c04ba69d358e4bbdd9c9295cf2e5", 462594),
    "29740": ("aa23104e6d3451495da8f4c4b4697e14195ce91f534b05a07e16df13abca50a8", 209903),
    "29741": ("66b435428d6f56531272b5126d8253d8bf3a5dcebbbe2fede4192a492d531e1a", 396402),
    "29742": ("e06a6f405a309164c69e5b1efe16308cc6a73dd730156f25bf6dcccdde0d40ac", 291934),
    "29743": ("5f500f75a9b89f5046bfe71fac33d3166198f0f54e4fcc8783fbf210c06786cf", 199415),
    "29744": ("370e3f0dfb6fcb8ca4eb460a3f3bbf1a88e6b49ed6e12080043504c1bb583e75", 94905),
    "29745": ("90bc34c278369315792bbdd555b6a28b59c2c05577268cfa8c06fa62634dd61c", 278846),
    "29746": ("ddafbfff05e0bbf6fd951fa47ad23e82221e4bf967f8257f33c0608dd276eecf", 476441),
    "29747": ("fee198facb8b7c32f51d0a71dfedce1b9e5774cfc53292805d46ad07fbcc4c09", 240146),
    "29748": ("d45aeb61c6b1325d67d9e4bae1306b41757e2b6603df737fa4e706aac4c72717", 437285),
    "29749": ("d8d1bea0ad92686a2a81cb9e64b176fcbb42359764037a08e490639ebc0e9101", 378157),
    "29750": ("7c0104f71e089e9f6c3deec5931cc9914ee5cab5e20dadc521d6ab6a9f01d8dc", 153136),
    "29751": ("8110503bbb60bcd584dbb654257501dd629d2cb67ac0c5efba506ef359fd263e", 198878),
    "29752": ("06660e9b67661ddccd6e5f6e29c33b3f7ed277438beb2fe535dac8a4dbc95d75", 178611),
    "29753": ("70d821652187f89d4722c35f2a6fe242ce10fd5ba19d0a20659eb712cc36f7e0", 223117),
    "29754": ("ce8741810135d13631f820b05c6236b5cc59a8c4b32ceb15d954b6c903ed10db", 388709),
    "29755": ("4e48a4ccf5778b08f72e880d1ffcc75c55754e959793b449749f492a4b861fdd", 25809),
    "29756": ("3e8c187144179fe615c4d65f9552796df6db854bbaf452ee04bf2b79a6b46f64", 249967),
    "29757": ("d0c5c9c2841255304d132cfcd21dff3f46a4a1fd949cb46d13290b078c3fc6ad", 404469),
    "29758": ("c8da6129ba6741fe9ab3aec3d1f2a98295a1a3ac0e39e681e0f4c959c50ba57c", 557828),
    "29759": ("ff17439b1a4254ec0ac56e56eeb3a4a79efdab180d3aacb719cada411a86db88", 189077),
    "29760": ("a35bfb5aab5ccda3e08664a1144afc6fe4aacd8d50dadfaad1299176cdba2b81", 41846),
    "29761": ("b605f111974843799b6893c7a676edacedb9c4605a22e9b4bb2481df1685385d", 145351),
    "29762": ("2551862d131925ecc320ca175eb500a6b5f14f034689494521b8f4ad0788d100", 490759),
    "29763": ("a5a65eaa03b4ac5be3d19393de923a3a75a47c0b91e24286ae672d2c8d72410e", 128025),
    "29764": ("4899ac930c06390744285b8eb9d4561e9fdfed9600ff2282d8e88662dd24de57", 299517),
    "29765": ("de3aecd40c5e88d2631d4b6579a6bb05cc3d17a0ca78e4c14b731ce72da9aa7a", 49944),
    "29766": ("8e8192b45a95a651037fe46ba0a376aefc8c416fc613273b451441ddd77f16d8", 10213),
    "29767": ("e1549a8a8a4df8a7adde1d7eda4267141cfea21d500c59585e83633e3c72bfdc", 164821),
    "29768": ("d0df77108a4e1b4c6795ecc2a20b88b278fd2eb1bb1cb2832634ffe212c1601d", 44980),
    "29769": ("1272ee672e32c91ad19f9ccc88bcb0cdd669b4f0a0aa59208c47050f27815b46", 131144),
    "29770": ("1b4b19ebee58cfe4c47c5151d605b478cc25e67cd551d2eb5aec7e9a90850745", 393641),
    "29771": ("6caa00c059b3c0328a4548a205b3df0f48c73ea7b7b3a28481a6be5a133ffe29", 313637),
    "29772": ("58dbfcfd30b1430aa729c308e894f854d3b1760517e57e84fe066310db571806", 45750),
    "29773": ("1a88906df09163fda64f2ab6657d17f40a51bd607c45a54d5e7b1df0fc28f269", 377194),
    "29774": ("aa3f4192358443d4c1a1dafa9df61bbed5d0b4a9819f39666981e05959fc47f6", 311439),
    "29775": ("d9e74a559870f3e71306bf4b308e3b191089a53be2537cd1206091c7496485bb", 422151),
    "29776": ("62fba88862012fccdbe5a546af02ede17a8dae4ef5da0530fd8075b11f7a81ca", 170851),
    "29777": ("a25d5bccdef331fb7d7f4069d4aff6bcb3fd2b5978fcbc786cb63dfc69f35a2f", 192365),
    "29778": ("e6c63a4e9ab1d42cc99b828c401b2df40440f4b98017d73be591cacbbe6d15c6", 35073),
    "29779": ("de04222ff62a9c7d608661adb83048bfa480187d9d370be96ce0c868fb986c73", 353379),
    "29780": ("b18fddc7fb6781eee1362d66f7ebb0b46356a7126228c5a423a6dd2a31b0821e", 191319),
    "29781": ("4e5d80215bcee8645c5a2004fb4e9344b98a1a89eef2cea52b5a391c0471ad23", 7013),
    "29782": ("b810bf11ab589b948767fde0ca1fe359a0fd95cdf479ff562de5e1c2ee08ab91", 177385),
    "29783": ("6a76d12335bc9865e3e6a590e0680a348bed174fe77ee88c17f1b87cc23dc74b", 362709),
    "29784": ("a52552b4845de621251adcbef62ed34ff858bc348bb67957d1ee70a4d8af1c63", 643614),
    "29785": ("b5ba4b328f41ce244cae7d121ea793db6628911c5a19c0236b6600da27e808a0", 438950),
    "29786": ("adad10c2b55b075c88840b83a93d7d9d76191560d250ed50f76c5feea72dbebe", 311019),
    "29787": ("5f34951d0ba8669cee13b5041e3829a0d4add8558c35f286ed364b6645e03509", 408758),
    "29788": ("9dc12128ac87915441da9f6c8e77442b0f8cad7c711d1d292cfaea40fced0819", 510613),
    "29789": ("dc8abffe7b476f2c50800a8ce313a1f9b38590fe772094f7c9913fc00582a02a", 114300),
    "29790": ("87a5d02fd6cf49995ebc927b78165f9b094734a75ec67e33e1403c90d0e0a938", 208011),
    "29791": ("3d7157161b22d66f2f13fbf4bee430ccd0f2b96209ec3294bd81718702ea1ea1", 269717),
    "29792": ("a7b2b807b937331c404f47d7cddedc0294598dc31829c00616cf0859faa2fca3", 248637),
    "29793": ("55f462ab11c6bd2984ba207b28bb3991a613d342924020f36898524d22f42eac", 34816),
    "29794": ("13fe979b44d3dabc50acd949923e90fc92c8155fcb5862072a491b4bccb56702", 231414),
    "29795": ("b0b4e84d43bb1126c219207815189746e573ca79fc6dcd974cb919c98dac6114", 28979),
    "29796": ("e1d375fb269a2f7af483e9642006687bbf48b13de6449eba269ed1ea301ead0b", 309491),
    "29797": ("e06bfad920e6141e63b310614a675666ee208e7b107747d7a57b0f52d03b5ecf", 232256),
    "29798": ("100c4e5ef4a5754fd8a396fd91fa953703fbf6ea049d14ffa67212eec2079a84", 525169),
    "29799": ("718923abc7b871ec6a3283107c9433315051f3871aa4b25c04478bd250e2305c", 47642),
    "29800": ("3abbfd82dfc8dc604ce731bcb4edd93e676981bd13f0156a5b72df58c61ecca3", 181879),
    "29801": ("c752d233318384092edd28eb738f8b53a4cf02877847d23d8f8a10d48e0d5fc8", 275995),
    "29802": ("d04441eaa445a65be7c41b9300424b8b0e997424a8f1a2ad6784e55993c9ae1e", 158578),
    "29803": ("9a49b8fc4b1cf9da61adce45cadc6c009fe1711906f5e65ea574031b5586917d", 201658),
    "29804": ("e3b1536cc038ead8f04be37c5cf66eeafa86061ae1e8b05f600965ab32ef5444", 44618),
    "29805": ("1ba19fc1161fa0643d394e7bdf98b574fe26c1a09ae0e6d76856ee5fb47e168e", 217746),
    "29806": ("48760b1f9389f4b2b2b249abf76f748b8a21ff4481a68ad0c0b6861a0f33d379", 290530),
    "29807": ("4bd808167d6c60b90ecf542efe9a62c4805259e606c7b959243fd54a57b5145b", 406162),
    "29808": ("39171d5f029c46839b3b7d823a4acdc93d24170fed93ec77afbdea8ad9bef061", 377411),
    "29809": ("b9839b6409a51d502ff51af56e467b6c782d41df94cc09be6d28a5872d797aa4", 232030),
    "29810": ("f4d37575e2a94b37f18f2fb886d7ebfd545d18ebdc5bac75198d14e8bb562416", 311279),
    "29811": ("3738f8d7b3480ac63a0f58799e2ade217b107fac31f86b242ccc3e1769c2bf50", 36891),
    "29812": ("3d10acf022e0f2d31dec9058c29fbfe6ea27a32d43d69fc48b2aa2b5a8d5e12f", 344695),
    "29813": ("c1c11c48d2dd6aa79dc22b6af2814e9dee43853de682d4cc0b59e913dade0ca9", 338234),
    "29814": ("d6a59d15a9118b09db6e03727a068cdaab917b70ef1c9cc60f1950a7f8701909", 152910),
    "29815": ("4b97dc8d91ba86407d26b6f0dc7bd744203d654f6f9fb0d21204abe357de3cb8", 363993),
    "29816": ("d9a92bc5ae4876c0276bfa66858df931d2016753c22c369d9c617eadee3de400", 341561),
    "29817": ("9a43c390d00d9456a25302b58924e3d9e561247ffcbd1493511625731dc84105", 45291),
    "29818": ("a645d2d4cf8f14dddc8ac101087fa86d48eece80fd787bf489916153af526565", 66253),
    "29819": ("8d84a9fd5e104bda8726dd9438b2bee542560471ede28d829b2ecf12e66f515c", 42837),
    "29820": ("f7f0770e298746744e1a21651265c5cbb41bac1f48138ff6265b27162e495157", 70105),
    "29821": ("cff1b37a175fd949f3335220b0dc3fa7f2ca349a161f48eb7cb8ac5205c82a31", 373794),
    "29822": ("054fa8b6e3edd5441f6310a6fb1239f42a7a58c02eefcfdba5b639d9e52274ff", 300882),
    "29823": ("cc037b4c8f0229735412f689becbcb5aaf8f364dbee0d9ee689bc597e7a90086", 157341),
    "29824": ("c67f671153473749741612116e3face19345176d972fc62c5e6757c12fd6790c", 424572),
    "29825": ("1bd5b519e8a29fba5a2475b65e37a62cf6b1c7910a4a224775e1175ebebf8a00", 34935),
    "29826": ("9f6dfcdd509f2974dffbf152bcdb6104fe0095de2ef3a6f706f222007cb22673", 219081),
    "29827": ("e54d8a9765a3a3191e8bd92fd5a2307993f6ff06cac731c8d264458567ed737b", 151335),
    "29828": ("156754312c694379b1deef15f0403a60c8c28de6da21445d073fa92ba28da370", 135069),
    "29829": ("ce98997aefae475337dac34196db3f9d2120dea9b78d953da4afe91feb7b05f4", 312500),
    "29830": ("c140df5d30319a0c79f5c2eb6a07bb54499385e90a254e332189b243475ae548", 158682),
    "29831": ("e0c3a6d74e62cb1fe9d44d9183f52e77dc48c9cf3258da8981895a326c9c4d68", 124324),
    "29832": ("7dc3e23e9a6051058bf9cb5f3ccf30b8909028811727a89a8230b35695f6049a", 157619),
    "29833": ("a9d972896c339964e8e6ae847b4f0f8ed92927104dc06c00fd52cce46a14c140", 314791),
    "29834": ("bbda9f55ffecdab5e915a9ca121c8896afdd35a439dcb3b979379ba78ebb27ef", 112718),
    "29835": ("0dc449131efa13e262f3f6e53e67a8408c78f4caba1b5ad13de3a3ed92e2979a", 650725),
    "29836": ("6a72975ad1bc60225a0fa4ea2eadec0cd8e136625550d56d6d9c0205b73a7844", 56931),
    "29837": ("c6a9c7f61bf6b08c3d124a9cdcb833f3e63764f4cc9c948b9dc92de3223d8f7c", 9447),
    "29838": ("df029de24901a88df81a5d75a61638c978628fbbe0bdd52aaf795907af86175c", 351697),
    "29839": ("72256894647c036c6edd98503efd475f736eca87274fc6e092f39d78e1ca7ca8", 278412),
    "29840": ("063a7e05c87e9efebdfb31056553e7f8ccb2e3673cb6ad52bbdc5da38fbea535", 458885),
    "29841": ("91d641b2899d01febc290a363b0838e8013ccc1eecc480756fe223a453f8bceb", 415795),
    "29842": ("10a8c31503f56618f0e0f81792722b0cab77e815007a0cc23200e3559bd4e611", 235645),
    "29843": ("b95db2d4e7f4633047b7dd86fcb59e5b0f898ad75a9395257b41797c7831123c", 365666),
    "29844": ("d8d25e573eed3cba58725b7c473e5abdc44c0b09544cf5eb2a33bf724b0f9811", 41258),
    "29845": ("78399ca0a32c0b0e7b66f9ad68848964d7336c9564d7b4aaa88b4d5d3e655625", 31828),
    "29846": ("dd31f05d384e7379f4fa1c0f4a879076b5a25b924dc29836dcf486d2113be8bb", 296024),
    "29847": ("cc8ca7ef3f31dae8782a03e8cffae4de07a5aaea8f5ce359c3a4bd2fc7e26a65", 20607),
    "29848": ("63fccd507b10d898f7c4a1e2fa208c77aac87b8c97a6bcfdf66c041090283c69", 384334),
    "29849": ("9057d56158295da1508c65889db2f5aded71bd40c5c5a94f44d92f072774253f", 203331),
    "29850": ("8c620f36df7d85fa3451c743f1ae33376cfe413cc42e371abc9c95e515a89a90", 13873),
    "29851": ("8da0593ca8d40e5a8323ae8b5b22947557aa12dec00d21552806cafed9d2f15b", 157156),
    "29852": ("312027bd87f112ff69bc5fc4b3ed8640ec3a4c35ba27f9c06e16b54c367c950a", 293040),
    "29853": ("cee16b0f4a24739f8886d93ce33da9b119a149e2f723535f639e65560d8f82e1", 30224),
    "29854": ("e45dc7b39fe592e8fcc056577ae07883fdd291aae485ea61eed99d2378a91651", 202810),
    "29855": ("ea863bf064573d9027395bf66a8823bd4b77850cc41e32a07995036d939f35f4", 155005),
    "29856": ("ebe6fb96e634982d52e1de04ec7a892f650b32f9a898ebf4570a51670c7966cb", 321606),
    "29857": ("39cad3a848dd8770b9f0b811f3fe6f44bea6e8a4dd3f49f11e23a47684f62c8a", 442161),
    "29858": ("ac486c1dd789b23ce83893275d06c940420e49352cc1e0f94679efc45f1e2155", 44835),
    "29859": ("e6ecef0f813ead568921e94c1ab0cae1d65b7ac741652b13da6ce352f9b2a65b", 360024),
    "29860": ("cd1b20b16b1f1750a6094fb7a5295801f267146548f585c795ecdea9cf31d2c5", 112705),
    "29861": ("9547e001963fc5f40799dc525b4ee735abe2c3bed26ff50b14fdd89999d25484", 350187),
    "29862": ("4a95fe7afc5cbb350a2b7ab7aa54e893734e004193253dde6c68df941eff309a", 176608),
    "29863": ("f13482011b51bac98c51333191dc49ca77eddeb57a06401a4c34e41bb29584a3", 21920),
    "29864": ("75f3f8bcbf488a3e68638d2b2f7e42232d9b6938aa96361cdd95fe5ec6073fd3", 399486),
    "29865": ("e0b2882dd26ddd86c2a4a8c1cdc4766979f5fb3528c83091cde8710809a2b260", 181975),
    "29866": ("10b10a831107a71b081d23d0102470397b76fb7ec3451c9e45ffaa872ad20cb3", 387122),
    "29867": ("e581aaddea9210301d461b6fd949296981f2dc2f235501d591ae8749976e9488", 36464),
    "29868": ("519f3f44fd0b1072f358cfeb38c0b0094947b0af68d5bad90411dd0e092bba82", 470945),
    "29869": ("5c27308a717e13f28d752e0a1f37053f7e4361854f4aeb05cebf7819b18b4e08", 38444),
    "29870": ("bd5b2fcb7772118dd3dfb618cc7fb107ab4ea8147a2703aff6778372a276dfab", 255456),
    "29871": ("b20f3dd88d003dda575298fefd59beab1ce7c0bd0169681c5b350df699ddd82a", 136816),
    "29872": ("8854df3cc3d65677dddde8c34abb35be691bfb2cecaed8b4618ca353089e72cb", 26384),
    "29873": ("08d6a427a414fa439976321eebc132893bef9f41fff477ac771f61aba6dd62d8", 343919),
    "29874": ("20203cf3e30e598c5ad2715613bc430630b64789a1e7fdc5538c668a7bc99356", 26686),
    "29875": ("ddce3e2f191ca4029f9d3751aa4cac699e0b906472582399e2594868c6c80a8b", 433027),
    "29876": ("64e08f36a55fa80ac7556753bcd3f687861fdb4ca6951e81433501fa12935785", 146029),
    "29877": ("eae555f3e43c9f2d291c329d0b9be7ada3105ebf8b3df4de2ed82ebdfdde6d8d", 363644),
    "29878": ("2c412687c2ebddc26eac9f9d6ea026cd5a46ad1751e0545e5ca61f4b62904081", 158283),
    "29879": ("f0ae8aacd8aed12f0650dbacfd940fee193eefc974d7e578023851df8e94ff2f", 320358),
    "29880": ("ad69e5243f84affd6874579c4f4e40b96d81c78e3f7e257321b808bb49323a1b", 425143),
    "29881": ("7d6c80fe0530cfc6fd22d9b1d02f60219208ff55f3b052fec9d116fe4493eea3", 355016),
    "29882": ("d7806ec349d8c4032b4cdc768f65dfc47a0cb89bc9f099bebdf301f3df52b9ba", 217695),
    "29883": ("0e54abb0080e25d7fd7b4ec403acfa7d801c96d37d1a45b1e329e2b0e8a77e09", 401414),
    "29884": ("b247ba5a815df0e394d90c67454b8f066019272d64852e8d2d1807a5e806209c", 266379),
    "29885": ("565796d60d5346a57b019f46473ade092378d9a39d968c49456587b721e1e5ae", 45163),
    "29886": ("51e9732afaf804f16a96683010680b532a4660db16a1e30c6d900580739b12ef", 423880),
    "29887": ("3b3685f5932f166e1b5aaf9609ff88d3c8289cae53b06f32463fd90813537333", 343056),
    "29888": ("36ee713e2670edee4ce73d7321fc0bb9486c7754f80b99975515766a71bc4224", 176362),
    "29889": ("1dc949bf65a6df8b790784cb519710a0aece208413875656bb972e4be8aa6f14", 486220),
    "29890": ("3938eb523b39444c900c2d1e579a8f201e43b5e58ff315b6f87567b044ca0397", 256699),
    "29891": ("6e285eea02e05516fad2b8b2416d21c3cfbb99b8a36071c04edff4cc5f01443c", 250683),
    "29892": ("adfa33e978312772159d20f59992451705bac376662a5e82dd2f1ff096bf8b4a", 221409),
    "29893": ("79ceae4e26eac4471eae61b153fa82c041c2b55e3cdb20de30bf8055fb5783c4", 30338),
    "29894": ("80b60ab1b670764121eb7cfe9527b21d742d65fdd60ed8d0a3532708b4b6a411", 171884),
    "29895": ("a6c0d677758a319fad37d82ff4e72ae7cd6c082ee3ce0f5f3c4e070904c87763", 329105),
    "29896": ("c68f5844a7bb9eef2a58349d9aaf413d5e08a5c32bcbf07a6e571e2992201155", 54418),
    "29897": ("7b84e61344bdac82853bd92b078f40a66aafef7feafd616d8aaf982b007d6a3e", 210137),
    "29898": ("e5d9e2bfdc9b834048b27482465de42eaceafbca929ccd8737241ca04322035f", 47292),
    "29899": ("f5daf4e2a9801bc398ab3969e45f2c06e17e4eca27773a233b81e4686af649c2", 193799),
    "29900": ("d37a8cb22ca826b2f93065fdb6197417745b3e4d621f3d63d215c31d52d43b9b", 269812),
    "29901": ("26ae57492fc829277d1c563592546613bb8654f2c316a2864c8995833ead5e9f", 354840),
    "29902": ("d6b51939865c7d6447282d00fff885d64499ea45667e32ec728928ffd34593e6", 308796),
    "29903": ("1001c263bd5f61fc77ba29f23fc5a6110302b915778570f0e56265d0d80bc4bc", 327179),
    "29904": ("be8572affcc124958453fe2dd9563a8dddc3f26c7eefb835c7588edfc0ca4150", 350369),
    "29905": ("e9a7c429b75feb89f00bcbc04890d7ab4ebb1c899fb529cb1bb0dd4fa2058276", 666604),
    "29906": ("50bc6e01f65c13f1ba5be935dabb08b594ce0b7b0792b2da18c1ca58fd31f74c", 315611),
    "29907": ("492557dcffe67f2f77bae127aab297b8fb1377ffe23479ecadfbaab0eff7e842", 50758),
    "29908": ("d3f4ad2154d2e561d904b9b40e60d4170fa5e96ae82c0d17408d20a7cb4e18d8", 391628),
    "29909": ("77cfe3c7ace2dbf5dbc11eb161851a623795d9c8863b812b3a55054bd3e84ca9", 33810),
    "29910": ("773d24c3642af73fd594a94aea0a2279fafd7703e6158f8b57990921fa29a8a6", 143512),
    "29911": ("35f622289ea354b103d06b32faf7f246b4dcda578a750863fd280f1c9ccc2ad0", 588105),
    "29912": ("cda1b5197a0b4c4ae52e294f874d7d58b4134cb7212ae9c322c32e129e661d62", 95025),
    "29913": ("489740b2b6591287ee9c59144611df6d6b6742d2b970c9a7ef02a74b374bf7bd", 369235),
    "29914": ("2fb2b04c3509315246ae70492258bc64d9613623d3792f046e6aae8979625af6", 433789),
    "29915": ("1fea659557baed69f07cba2635d02ebe76d2570a6e9f93aa51daa9d1fc1ad3a0", 356727),
    "29916": ("fb7789e515f353d2de64fae91c73e809c57bc96e4078a8c5e120d384157ed55e", 320303),
    "29917": ("289b1ce17b4ad8c960f1da809285bfc3034d82d674e3dbedda403e48272aac3a", 17564),
    "29918": ("6725fb1e23e0e72cbf9bc450d0dd3cb175c3be1444fda811332bcc5309fe62fc", 408785),
    "29919": ("894dd00881dd040fa05e5074277ee2c34ef55e9eda2d4ad1e903d0d92f6312af", 51338),
    "29920": ("8f05a064c40d7c4d3b81dae0b0b21c9be66f09f035e8e96c5280e39aba9a2ddb", 340034),
    "29921": ("be1b76828fb49acac70aa634ee09b59ab78a22a4f357df655aa19d5fd976cc84", 216071),
    "29922": ("6d39c79bb7902a6dfebf79d080386e15c9af4be9bde68ed123b296d923af1309", 314600),
    "29923": ("588cbf95b3306e3cad45bd1a933cf0f23b1e2d7090c4db2b73e788bff43c3d06", 228564),
    "29924": ("7124eab0c351c49769c9aadbc371b5554b1baba046e175d5e5f877de7606d1fb", 193119),
    "29925": ("c211e7abdc83cf9ddaabddb5905147620805b378d4429b70cef0d9398fe2d4e7", 394759),
    "29926": ("76b3d18ab768faba062faf82e940fd69f844551eed2e8eb9db03cc4f98ba1ec6", 28284),
    "29927": ("f9e9947c7ec95d28dcebcc29f80bee61463ef61f5ecf77de5037c8ef3eec7f42", 14048),
    "29928": ("881864d6d1bde3ab2934d1ab8f66cac3d206d5229f9d7acd4fb4a7748fb69018", 215683),
    "29929": ("607b254ff8a6b18152f7a13db476ed0a969ffca5c06c20f21c47b9f2d6c78640", 251318),
    "29930": ("25b665a789ea24b1764b20e1c10bd37a86f5cb5b7ba9e31ab9c3493c003c2c8a", 255222),
    "29931": ("a8cd3f2809c4a73180b9e182408e22cfd68ca66f0ad343f366458f3595a91d37", 88801),
    "29932": ("a81e1534adbdcbda38a5b33998387d1d09d3780be37a521e80be6feb25a80de6", 178374),
    "29933": ("a427850dc4e04ae512ac8f643c8836e839e61124c20bb47c18156fc332b27cad", 438440),
    "29934": ("d50b65f99b28f2a37a875ec5bd2681c00a593d469c6cde01959b9c84acf5f7b4", 102557),
    "29935": ("61e6a46b4196e4ca8d4025c498802eb07bcf2e3b13e65acc8444c301cb1b1b13", 271692),
    "29936": ("5e1c02e8a715c4fab1316409550d402be74ef05e371bf2b3c75b8ad85f9ab4e7", 45024),
    "29937": ("9c7ebeb9cba5790c1a90edb4f6fb713db68709c9fcfdfa934454bda0760744db", 88587),
    "29938": ("513bcef293345224d54b67d964b7ae9b4227a4b1e06dc2d916b0144103bff839", 19210),
    "29939": ("613d13d83aa482adda4e812c2cf68234d2f3ff749589bc50a32603f181fafa7b", 349382),
    "29940": ("fc4c8864e9f88b01206fc2b83b1eeb6ba4c60a2146d5c26c1caf1bb8c85f54b1", 294794),
    "29941": ("b301240f69917690ab11066031fafe6021e9c837b7bd9bad913dadb97e239cf8", 428491),
    "29942": ("d4fa60eae71db77cde7e1adefb908ff26dd2ef4b64f840d5f6cd2d17ad3ff19e", 472554),
    "29943": ("e83a270164e3f37761adf3bc12362e9d1d5a8a4389f666233a967f1b28a8112a", 509787),
    "29944": ("92f6e9353634b2eb411affb00445dd4c7bbdbe3d83eb40fe777c5d387dfa9d31", 234212),
    "29945": ("ce237c3bec5cc7cf2bc16fbdcaa9b35e35267f350698c338fee6dcaa0a3374ed", 144498),
    "29946": ("f41d3c5b282d309860be51d9dc6f69c26eaae73bb7c042428b006e8fde8f536a", 213113),
    "29947": ("1c2725952f9119319b19403a3208a409925d97e077caae3fa6e4856ad2360280", 189683),
    "29948": ("83ec0cda3a11e39eb7ac9b326c5f5315e22c246417d7767ab9dfb49d71c19934", 329393),
    "29949": ("201e9f182576d6050a29a8a3a7604a4c8075b38d1dcd288221284cdce073fe47", 250680),
    "29950": ("d6acd6861639dfda3e06c85ad0cd8991548d18216ca8f062b425c2f64962a701", 215597),
    "29951": ("ba278857bebb0431a3423a63e77430ae6df6e13eab15fef61edbd06152a5c64b", 162039),
    "29952": ("44e808a799fbd557410d8abe27d5fc09a3fb21b80313567ce5769c95d027355a", 169532),
    "29953": ("f8895274b0ce60e90dc0ef04e43d343376a25bd37690812c1af2127ed5b3dd58", 180269),
    "29954": ("657c4225719cc8c4114db0629c7e572daa533e9081934a697fd6461a7630f1ec", 354907),
    "29955": ("7ec84d11b3ff20f52cb0e67bfc1e95961e06a673614e641b37425ec1945900d4", 238628),
    "29956": ("340f80df2b31c41399f65ec4a63fe352942604c61472537b2018e296e5d513f1", 59146),
    "29957": ("3c3eaae89011a58996420067f8e9d54ef3add5953e02b7946fc853281d849c79", 38775),
    "29958": ("0631ba0a1efca02b1ab476450e9fe8a9694a56d7c15683a9e60d11609821782b", 625637),
    "29959": ("72cc08bf673ebf2ed661130114b9c68f2e118bb9f09119415288a383166a4e76", 250745),
    "29960": ("5e70073f05770966965359cf8ccd0003422734e5f81285401d75539d52e52644", 297808),
    "29961": ("90f12d2e00663cfe64f1d2c8a74b991631eb21867e7c2c743111ba139b154530", 254933),
    "29962": ("6e6e0ea8a9a0c07cda332e0504bda15681305f152df65f3027edca7793202685", 283494),
    "29963": ("6ba4719e1b0c9a9b1ea5445c9864ab7de000f58346e42e200a2caf70b232c2d6", 175496),
    "29964": ("27954bbf156be5dc3c555484c4baa405f31a4b5b57ed33a7ecce330f33a060c0", 39077),
    "29965": ("857520da4ed5efa089022beee3bf9c606185d2a8f332efb681344f26930ba035", 11193),
    "29966": ("dcfc5bed185a039c4c1d5a2df4bab33aa671cf147f6c31c209be737432b886e0", 489420),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of the decoded text columns as a canonical JSON list of
    `[id, captions, question_type, text_detected]`."""
    payload = [
        [str(r["id"]), [str(c) for c in r["captions"]], str(r["question_type"]), bool(r["text_detected"])]
        for r in rows
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to
    read a parquet footer and a few column chunks without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _open_shard() -> Any:
    """The pinned shard as a `pyarrow.parquet.ParquetFile` over HTTPS range requests, after the file's
    declared size and LFS SHA-256 have been checked against the pins."""
    import pyarrow.parquet as pq
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(CORPUS_REPO, CORPUS_FILE["path"], repo_type="dataset", revision=CORPUS_REVISION)
    metadata = get_hf_file_metadata(url)
    declared = (metadata.etag or "").strip('"')
    if metadata.size != CORPUS_FILE["bytes"] or declared != CORPUS_FILE["sha256"]:
        raise ValueError(
            f"caption shard: the Hub declares {metadata.size} bytes / sha256 {declared[:16]}…, "
            f"pinned {CORPUS_FILE['bytes']} / {CORPUS_FILE['sha256'][:16]}…"
        )
    return pq.ParquetFile(_HttpRangeFile(url, CORPUS_FILE["bytes"]))


def _hub_text() -> list[dict[str, Any]]:
    rows = _open_shard().read(columns=list(CORPUS_TEXT_COLUMNS)).to_pylist()
    return [
        {
            "id": str(r["id"]),
            "captions": [str(c) for c in r["answer"]],
            "question_type": str(r["question_type"]),
            "text_detected": bool(r["text_detected"]),
        }
        for r in rows
    ]


def _hub_images() -> dict[str, bytes]:
    """The `media` column of the pinned row group only: image id -> JPEG bytes."""
    table = _open_shard().read_row_group(CORPUS_FILE["row_group"], columns=["id", CORPUS_IMAGE_COLUMN])
    out: dict[str, bytes] = {}
    for row in table.to_pylist():
        media = row[CORPUS_IMAGE_COLUMN]
        if not isinstance(media, list) or len(media) != 1:
            raise ValueError(
                f"row {row['id']}: expected exactly one image, found {len(media) if media else 0}"
            )
        out[str(row["id"])] = bytes(media[0]["bytes"])
    return out


def fetch_annotations(
    *, cache_dir: str | Path | None = None, fetcher: Callable[[], Sequence[Mapping[str, Any]]] | None = None
) -> list[dict[str, Any]]:
    """Return the shard's 1,550 caption rows from the cache or the Hub, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "annotations.json"
    rows: list[dict[str, Any]] | None = None
    if local.is_file():
        cached = json.loads(local.read_text(encoding="utf-8"))
        if isinstance(cached, list) and text_digest(cached) == CORPUS_FILE["text_sha256"]:
            rows = cached
    if rows is None:
        raw = fetcher() if fetcher is not None else _hub_text()
        rows = [
            {
                "id": str(r["id"]),
                "captions": [str(c) for c in r["captions"]],
                "question_type": str(r["question_type"]),
                "text_detected": bool(r["text_detected"]),
            }
            for r in raw
        ]
        if len(rows) != CORPUS_FILE["rows"] or text_digest(rows) != CORPUS_FILE["text_sha256"]:
            raise ValueError(
                f"caption shard: fetched {len(rows)} rows with text sha256 {text_digest(rows)[:16]}…, "
                f"pinned {CORPUS_FILE['rows']} / {CORPUS_FILE['text_sha256'][:16]}…"
            )
        local.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def fetch_images(
    image_ids: Sequence[str],
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[], Mapping[str, bytes]] | None = None,
) -> dict[str, Path]:
    """Stage the pinned photographs into the cache: cached files are re-hashed; anything missing or drifted is
    read from the pinned row group (one range read of its image column) and refused on any size or SHA-256
    mismatch. Returns image id -> path."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    (cache / "images").mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    missing: list[str] = []
    for image_id in image_ids:
        if image_id not in IMAGE_PINS:
            raise ValueError(f"{image_id} is not one of the {len(IMAGE_PINS)} pinned photographs")
        digest, size = IMAGE_PINS[image_id]
        dest = cache / "images" / f"{image_id}.jpg"
        data = dest.read_bytes() if dest.is_file() else None
        if data is None or len(data) != size or _sha256_bytes(data) != digest:
            missing.append(image_id)
        out[image_id] = dest
    if missing:
        payload = fetcher() if fetcher is not None else _hub_images()
        for image_id in missing:
            digest, size = IMAGE_PINS[image_id]
            data = payload.get(image_id)
            if data is None or len(data) != size or _sha256_bytes(data) != digest:
                got = (
                    f"{len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…"
                    if data is not None
                    else "no bytes"
                )
                raise ValueError(f"{image_id}.jpg: read {got}, pinned {size} / {digest[:16]}…")
            (cache / "images" / f"{image_id}.jpg").write_bytes(data)
    return out


def build_sample_dataset(
    annotations: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    image_paths: Mapping[str, Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The pinned photographs that carry at least one reference caption, shuffled with `seed` and cut **by
    image** into `sizes`; `image_paths` (from `fetch_images`) fills each record's `image`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    by_id = {str(r["id"]): r for r in annotations}
    missing = [image_id for image_id in IMAGE_PINS if image_id not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} pinned photograph(s) are absent from the annotations: {missing[:3]}"
        )
    pool = sorted(image_id for image_id in IMAGE_PINS if len(by_id[image_id]["captions"]) >= MIN_CAPTIONS)
    if sum(sizes.values()) > len(pool):
        raise ValueError(
            f"{len(pool)} pinned photographs carry a caption; the split sizes need {sum(sizes.values())}"
        )
    random.Random(seed).shuffle(pool)
    out: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name in ("train", "validation", "test"):
        chosen = pool[offset : offset + sizes[name]]
        offset += sizes[name]
        out[name] = [
            {
                "id": f"{name}-{index:04d}",
                "image_id": image_id,
                "image": str(image_paths[image_id])
                if image_paths is not None and image_id in image_paths
                else f"{image_id}.jpg",
                "captions": [str(c) for c in by_id[image_id]["captions"]],
                "category": "text" if by_id[image_id]["text_detected"] else "no-text",
            }
            for index, image_id in enumerate(chosen)
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    annotation_fetcher: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    image_fetcher: Callable[[], Mapping[str, bytes]] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned annotations and photographs."""
    annotations = fetch_annotations(cache_dir=cache_dir, fetcher=annotation_fetcher)
    paths = fetch_images(sorted(IMAGE_PINS), cache_dir=cache_dir, fetcher=image_fetcher)
    return build_sample_dataset(annotations, seed=seed, sizes=sizes, image_paths=paths)


def _check_record(record: Any, index: int, *, base_dir: Path | None) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/captions")
    for key in ("id", "image", "captions"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    image_ref = record["image"]
    if not isinstance(image_ref, (str, Path)) or not str(image_ref).strip():
        raise ValueError(f"{label}: image must be a file path")
    path = Path(image_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label}: image file not found: {path}")
    try:
        with Image.open(path) as handle:
            handle.load()
            validate_image(handle)
            width, height = handle.size
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: image cannot be decoded: {exc}") from exc
    captions = record["captions"]
    if isinstance(captions, str) or not isinstance(captions, Sequence) or len(captions) < MIN_CAPTIONS:
        raise ValueError(f"{label}: captions must be a list of at least {MIN_CAPTIONS} reference caption(s)")
    checked_captions = [" ".join(str(c).split()) for c in captions]
    if not all(checked_captions):
        raise ValueError(f"{label}: every reference caption must be a non-empty string")
    if any(len(c) > MAX_CAPTION_CHARS for c in checked_captions):
        raise ValueError(f"{label}: a reference caption exceeds MAX_CAPTION_CHARS={MAX_CAPTION_CHARS}")
    prefix = record.get("prefix")
    if prefix is not None and (not isinstance(prefix, str) or len(prefix) > MAX_PREFIX_CHARS):
        raise ValueError(f"{label}: prefix must be a str of at most MAX_PREFIX_CHARS={MAX_PREFIX_CHARS}")
    return {
        "id": rid,
        "image_id": str(record.get("image_id", rid)),
        "image": str(path),
        "image_size": [width, height],
        "captions": checked_captions,
        "category": str(record.get("category", "other")),
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Structural validation of a captioning dataset (every image opened and decoded); raises ValueError
    before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, captions} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    base = Path(base_dir) if base_dir is not None else None
    checked = []
    ids: set[str] = set()
    images: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index, base_dir=base)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        images.add(item["image_id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_images": len(images),
        "categories": dict(Counter(r["category"] for r in checked)),
        "captions_per_image": {
            "min": min(len(r["captions"]) for r in checked),
            "max": max(len(r["captions"]) for r in checked),
        },
        "caption_words": {
            "min": min(len(c.split()) for r in checked for c in r["captions"]),
            "max": max(len(c.split()) for r in checked for c in r["captions"]),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["image_id"], list(r["captions"]), r.get("category", "")] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def reference_captions(record: Mapping[str, Any]) -> list[str]:
    """The reference captions of a record."""
    return [str(c) for c in record["captions"]]


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image id appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record.get("image_id", record["id"]))
            if key in seen and seen[key] != name:
                raise ValueError(f"image {key!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    base_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by image**: every record on the same image
    lands in the same split, so a test image is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, base_dir=base_dir)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["image_id"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of ``{id, image, captions}`` objects; `image` paths are
    resolved relative to the file's directory by `validate_dataset(..., base_dir=...)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back (image paths as given)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("id", "image_id", "image", "captions", "category")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
